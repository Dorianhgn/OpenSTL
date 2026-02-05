# file: models/stvmamba/stvmamba_modules.py
# Corrected STDSConv and STVMambaModule implementations compared to stvmamba_model_unet.py
# Make sure to install mamba: pip install mamba-ssm[causal-conv1d]
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

# get MAMBA_VERSION from the environment, default to '1' if not set
import os
MAMBA_VERSION = os.getenv("MAMBA_VERSION", "2")

if MAMBA_VERSION == "1":
    from mamba_ssm import Mamba
    print("🐍 Using Mamba Version 1 🚜")
elif MAMBA_VERSION == "2":
    from mamba_ssm import Mamba2 as Mamba
    print("🐍 Using Mamba Version 2 🚀")
else:
    raise ValueError(f"Unsupported MAMBA_VERSION: {MAMBA_VERSION}")

from timm.layers import DropPath
from typing import Literal, Optional, Iterable


# =============================================================================
# STVMamba Building Blocks
# =============================================================================
class STDSConv(nn.Module):
    """Spatial-Temporal Depthwise Separable Convolution
    Fidèle à la Figure 1(c) : TPC -> (SDC + Residual) -> TPC
    """
    def __init__(self, dim: int, temporal_kernel: int = 3, spatial_kernel: int = 3):
        super().__init__()
        
        # 1. Temporal Pointwise Conv (Bas du schéma)
        # "Pointwise" suggère souvent un mélange de canaux (groups=1).
        # Ici on fait une conv sur le temps (T) en mélangeant les canaux.
        self.temporal_conv1 = nn.Conv2d(
            dim, dim,
            kernel_size=(temporal_kernel, 1),
            padding=(temporal_kernel // 2, 0),
            groups=1 # Mélange les canaux (Pointwise logic)
        )
        
        # 2. Spatial Depthwise Conv (Milieu du schéma)
        # "Depthwise" = groups=dim (pas de mélange de canaux, juste du spatial)
        self.spatial_conv = nn.Conv2d(
            dim, dim,
            kernel_size=spatial_kernel, # Carré (K, K)
            padding=spatial_kernel // 2,
            groups=dim # Depthwise pur
        )
        
        # 3. Temporal Pointwise Conv (Haut du schéma)
        self.temporal_conv2 = nn.Conv2d(
            dim, dim,
            kernel_size=(temporal_kernel, 1),
            padding=(temporal_kernel // 2, 0),
            groups=1 # Mélange les canaux
        )
        
        # Optionnel : Normalisation/Activation entre les couches ?
        # Le schéma montre juste des boîtes, mais souvent on met un SiLU/GELU entre.
        self.act = nn.SiLU() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (b, t, c, h, w)
        B, T, C, H, W = x.shape

        # ───── BLOC 1 : Temporal Pointwise ─────
        # On passe en (B*H*W, C, T, 1) pour convoler sur T
        x = rearrange(x, 'b t c h w -> (b h w) c t 1')
        x = self.temporal_conv1(x)
        # On peut ajouter une activation ici si on veut faire "Sandwich MLP"
        x = self.act(x) 
        x = rearrange(x, '(b h w) c t 1 -> b t c h w', h=H, w=W)

        # ───── BLOC 2 : Spatial Depthwise avec Résiduel ─────
        # C'est ici que le schéma montre la boucle résiduelle
        spatial_input = x 
        
        # Préparation pour conv spatiale (B*T, C, H, W)
        x_s = rearrange(x, 'b t c h w -> (b t) c h w')
        x_s = self.spatial_conv(x_s)
        x_s = rearrange(x_s, '(b t) c h w -> b t c h w', b=B)
        
        # Application du résiduel local (Fig 1c : la somme au milieu)
        x = spatial_input + x_s

        # ───── BLOC 3 : Temporal Pointwise ─────
        x = rearrange(x, 'b t c h w -> (b h w) c t 1')
        x = self.temporal_conv2(x)
        x = rearrange(x, '(b h w) c t 1 -> b t c h w', h=H, w=W)

        return x


class STSS(nn.Module):
    """Spatial-Temporal Selective Scan
    4 directions de scan (comme VMamba Cross-Scan, mais adapté au temporel)
    einops rend les transpositions et scans ultra clairs.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.s6_modules = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(4)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (b, t, c, h, w)
        B, T, C, H, W = x.shape

        # Vue row-major (h w devient une séquence spatiale)
        x_hw = rearrange(x, 'b t c h w -> b t (h w) c')
        # Vue column-major
        x_wh = rearrange(x, 'b t c h w -> b t (w h) c')

        # Les 4 scans
        scans = [
            x_hw,                          # row forward
            torch.flip(x_hw, dims=[2]),    # row reverse (flip sur la dim spatiale)
            x_wh,                          # col forward
            torch.flip(x_wh, dims=[2]),    # col reverse
        ]

        # Concat temporel → longue séquence (t * spatial)
        scans_flat = [rearrange(s, 'b t l c -> b (t l) c') for s in scans]

        # Passage dans les 4 Mamba indépendants
        outs = [mamba(scan) for mamba, scan in zip(self.s6_modules, scans_flat)]

        # Retour en (b, t, l, c)
        outs = [rearrange(o, 'b (t l) c -> b t l c', t=T) for o in outs]

        # Reverse les flips + transpose col → row
        out_hw_fwd = outs[0]
        out_hw_rev = torch.flip(outs[1], dims=[2])
        out_wh_fwd = rearrange(outs[2], 'b t (w h) c -> b t (h w) c', h=H)
        out_wh_rev = rearrange(torch.flip(outs[3], dims=[2]), 'b t (w h) c -> b t (h w) c', h=H)

        # Somme des 4 directions
        out = out_hw_fwd + out_hw_rev + out_wh_fwd + out_wh_rev

        # Retour en format spatial original
        out = rearrange(out, 'b t (h w) c -> b t c h w', h=H, w=W)

        return out

class SwiGLU(nn.Module):
    """SwiGLU classique (meilleur que GLU standard)"""
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(4 * dim)  # classique : ~4x comme dans FFN Transformer
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, dim)
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.dropout(self.w3(hidden))


class STVMambaModule(nn.Module):
    """Core STVMamba block — fidèle au papier + améliorations standard"""
    def __init__(self, dim: int, d_state: int = 16, expand: int = 2, drop_path: float = 0.0, layer_scale_init: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.inner_dim = expand * dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Norms (sur canal)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Branche SS3D (gating style Mamba standard + STDSConv avant STSS)
        self.ss3d_in_proj = nn.Linear(dim, 2 * self.inner_dim, bias=False)
        self.stds_conv = STDSConv(self.inner_dim)
        self.stss = STSS(d_model=self.inner_dim, d_state=d_state)
        self.ss3d_out_proj = nn.Linear(self.inner_dim, dim, bias=False)

        # Branche FFN : SwiGLU sur canaux (remplace ConvGLU — meilleure perf)
        self.ffn = SwiGLU(dim)

        # Layer scale (comme dans le papier)
        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        pos = (B, T, H, W)  # pour rearrange

        # === Branche 1 : SS3D (global spatiotemporal) ===
        residual = x
        x1 = rearrange(x, 'b t c h w -> (b t h w) c')
        x1 = self.norm1(x1)
        proj = self.ss3d_in_proj(x1)                              # → 2 * inner_dim
        main, gate = proj.chunk(2, dim=-1)                         # main & gate : inner_dim
        gate = F.silu(gate)

        main = rearrange(main, '(b t h w) i -> b t i h w', b=B, t=T, h=H, w=W)
        main = self.stds_conv(main)
        main = self.stss(main)
        main = rearrange(main, 'b t i h w -> (b t h w) i')

        main = main * gate                                        # gating ici
        main = self.ss3d_out_proj(main)
        main = rearrange(main, '(b t h w) c -> b t c h w', b=B, t=T, h=H, w=W)

        x = residual + self.drop_path(self.gamma1 * main)

        # === Branche 2 : SwiGLU (intra-token / FFN) ===
        residual = x
        x2 = rearrange(x, 'b t c h w -> (b t h w) c')
        x2 = self.norm2(x2)
        x2 = self.ffn(x2)
        x2 = rearrange(x2, '(b t h w) c -> b t c h w', b=B, t=T, h=H, w=W)

        x = residual + self.drop_path(self.gamma2 * x2)

        return x