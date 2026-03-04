# Copyright (c) CAIRI AI Lab. All rights reserved
# STVMamba modules for Spatial-Temporal Video Mamba
# Ported from MambaFlow/JvM implementation

"""
STVMamba Building Blocks for Spatiotemporal Video Processing.

Contains:
- STDSConv: Spatial-Temporal Depthwise Separable Convolution
- STSS: Spatial-Temporal Selective Scan (4-direction Mamba scan)
- SwiGLU: Gated Linear Unit with Swish activation
- STVMambaModule: Core STVMamba block
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_
from typing import Optional

# Mamba version selection from environment
MAMBA_VERSION = os.getenv("MAMBA_VERSION", "2")

try:
    if MAMBA_VERSION == "1":
        from mamba_ssm import Mamba
    elif MAMBA_VERSION == "2":
        from mamba_ssm import Mamba2 as Mamba
    else:
        raise ValueError(f"Unsupported MAMBA_VERSION: {MAMBA_VERSION}")
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba = None


# =============================================================================
# STVMamba Building Blocks
# =============================================================================

class STDSConv(nn.Module):
    """
    Spatial-Temporal Depthwise Separable Convolution.
    
    Architecture: TPC -> (SDC + Residual) -> TPC
    Where TPC = Temporal Pointwise Conv, SDC = Spatial Depthwise Conv
    
    Args:
        dim: Number of channels
        temporal_kernel: Kernel size for temporal convolution
        spatial_kernel: Kernel size for spatial convolution
    """
    
    def __init__(self, dim: int, temporal_kernel: int = 3, spatial_kernel: int = 3):
        super().__init__()
        
        # 1. Temporal Pointwise Conv (channel mixing)
        self.temporal_conv1 = nn.Conv2d(
            dim, dim,
            kernel_size=(temporal_kernel, 1),
            padding=(temporal_kernel // 2, 0),
            groups=1
        )
        
        # 2. Spatial Depthwise Conv (spatial processing)
        self.spatial_conv = nn.Conv2d(
            dim, dim,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            groups=dim
        )
        
        # 3. Temporal Pointwise Conv (channel mixing)
        self.temporal_conv2 = nn.Conv2d(
            dim, dim,
            kernel_size=(temporal_kernel, 1),
            padding=(temporal_kernel // 2, 0),
            groups=1
        )
        
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, C, H, W)
            
        Returns:
            Output tensor (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        # Block 1: Temporal Pointwise
        x = rearrange(x, 'b t c h w -> (b h w) c t 1')
        x = self.temporal_conv1(x)
        x = self.act(x)
        x = rearrange(x, '(b h w) c t 1 -> b t c h w', h=H, w=W)

        # Block 2: Spatial Depthwise with Residual
        spatial_input = x
        x_s = rearrange(x, 'b t c h w -> (b t) c h w')
        x_s = self.spatial_conv(x_s)
        x_s = rearrange(x_s, '(b t) c h w -> b t c h w', b=B)
        x = spatial_input + x_s

        # Block 3: Temporal Pointwise
        x = rearrange(x, 'b t c h w -> (b h w) c t 1')
        x = self.temporal_conv2(x)
        x = rearrange(x, '(b h w) c t 1 -> b t c h w', h=H, w=W)

        return x


class STSS(nn.Module):
    """
    Spatial-Temporal Selective Scan.
    
    4-direction scan (VMamba Cross-Scan adapted for temporal):
    - Row forward/reverse
    - Column forward/reverse
    
    Args:
        d_model: Model dimension
        d_state: Mamba state dimension
        d_conv: Mamba convolution size
        expand: Mamba expansion factor
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "Mamba is not installed. Please install with: pip install mamba-ssm[causal-conv1d]"
            )
        
        self.s6_modules = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(4)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, C, H, W)
            
        Returns:
            Output tensor (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        # Row-major view (h w -> sequence)
        x_hw = rearrange(x, 'b t c h w -> b t (h w) c')
        # Column-major view
        x_wh = rearrange(x, 'b t c h w -> b t (w h) c')

        # 4 scan directions
        scans = [
            x_hw,                           # row forward
            torch.flip(x_hw, dims=[2]),     # row reverse
            x_wh,                           # col forward
            torch.flip(x_wh, dims=[2]),     # col reverse
        ]

        # Flatten temporal + spatial for Mamba
        scans_flat = [rearrange(s, 'b t l c -> b (t l) c') for s in scans]

        # Pass through 4 independent Mamba modules
        outs = [mamba(scan) for mamba, scan in zip(self.s6_modules, scans_flat)]

        # Reshape back to (b, t, l, c)
        outs = [rearrange(o, 'b (t l) c -> b t l c', t=T) for o in outs]

        # Reverse flips and transpose col -> row
        out_hw_fwd = outs[0]
        out_hw_rev = torch.flip(outs[1], dims=[2])
        out_wh_fwd = rearrange(outs[2], 'b t (w h) c -> b t (h w) c', h=H)
        out_wh_rev = rearrange(torch.flip(outs[3], dims=[2]), 'b t (w h) c -> b t (h w) c', h=H)

        # Sum all 4 directions
        out = out_hw_fwd + out_hw_rev + out_wh_fwd + out_wh_rev

        # Back to spatial format
        out = rearrange(out, 'b t (h w) c -> b t c h w', h=H, w=W)

        return out


class SwiGLU(nn.Module):
    """
    SwiGLU activation (better than standard GLU).
    
    Formula: SwiGLU(x) = (xW1 ⊙ SiLU(xW2))W3
    
    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension (default: 4 * dim)
        dropout: Dropout rate
    """
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(4 * dim)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.dropout(self.w3(hidden))


class STVMambaModule(nn.Module):
    """
    Core STVMamba block.
    
    Architecture:
    1. SS3D branch: Norm -> Gated STDSConv + STSS -> Residual
    2. FFN branch: Norm -> SwiGLU -> Residual
    
    Args:
        dim: Model dimension
        d_state: Mamba state dimension
        expand: Mamba expansion factor
        drop_path: Drop path rate
        layer_scale_init: Initial value for layer scale
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        expand: int = 2,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-4
    ):
        super().__init__()
        self.dim = dim
        self.inner_dim = expand * dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # SS3D branch (gating + STDSConv + STSS)
        self.ss3d_in_proj = nn.Linear(dim, 2 * self.inner_dim, bias=False)
        self.stds_conv = STDSConv(self.inner_dim)
        self.stss = STSS(d_model=self.inner_dim, d_state=d_state)
        self.ss3d_out_proj = nn.Linear(self.inner_dim, dim, bias=False)

        # FFN branch: SwiGLU
        self.ffn = SwiGLU(dim)

        # Layer scale
        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, C, H, W)
            
        Returns:
            Output tensor (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        # Branch 1: SS3D (global spatiotemporal)
        residual = x
        x1 = rearrange(x, 'b t c h w -> (b t h w) c')
        x1 = self.norm1(x1)
        proj = self.ss3d_in_proj(x1)
        main, gate = proj.chunk(2, dim=-1)
        gate = F.silu(gate)

        main = rearrange(main, '(b t h w) i -> b t i h w', b=B, t=T, h=H, w=W)
        main = self.stds_conv(main)
        main = self.stss(main)
        main = rearrange(main, 'b t i h w -> (b t h w) i')

        main = main * gate
        main = self.ss3d_out_proj(main)
        main = rearrange(main, '(b t h w) c -> b t c h w', b=B, t=T, h=H, w=W)

        x = residual + self.drop_path(self.gamma1[None, None, :, None, None] * main)

        # Branch 2: SwiGLU FFN
        residual = x
        x2 = rearrange(x, 'b t c h w -> (b t h w) c')
        x2 = self.norm2(x2)
        x2 = self.ffn(x2)
        x2 = rearrange(x2, '(b t h w) c -> b t c h w', b=B, t=T, h=H, w=W)

        x = residual + self.drop_path(self.gamma2[None, None, :, None, None] * x2)

        return x


# =============================================================================
# Helper
# =============================================================================

def _safe_num_groups(channels: int, requested: int = 8) -> int:
    """Return the largest divisor of `channels` that is <= `requested`."""
    for g in range(min(requested, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# =============================================================================
# MidSTVMamba: Multi-Resolution Bottleneck for SimVP
# =============================================================================

class MidSTVMamba(nn.Module):
    """
    Multi-resolution STVMamba bottleneck that plugs into SimVP as the ``hid``
    temporal translator (same interface as ``MidMetaNet``).

    Architecture (faithfully replicates STVMambaFusionModel bottleneck):
        bottleneck_proj_in
            → initial_block
            → [Branch A1: n_t1 blocks @hid_c1]  ‖  [Branch A2: down→n_t2 blocks @hid_c2→up]
            → fusion_conv → final_block
        bottleneck_proj_out

    Drop-path is applied with a linear schedule from 0 → drop_path_rate
    across all STVMamba blocks (initial, A1, A2, final).

    Args:
        channel_in:       Input/output channel count (= hid_S from SimVP encoder).
        hid_c1:           Tier-1 bottleneck channels.  Must satisfy
                          hid_c1 * expand * 2 divisible by Mamba2 headdim (64)
                          with nheads >= 8 → hid_c1 >= 128 @ expand=2.
        hid_c2:           Tier-2 bottleneck channels (same constraint).
        n_t1:             Number of Tier-1 STVMamba blocks (Branch A1).
        n_t2:             Number of Tier-2 STVMamba blocks (Branch A2).
        d_state:          Mamba state dimension.
        drop_path_rate:   Max drop-path rate (linearly scheduled over blocks).
        layer_scale_init: Initial LayerScale parameter value.
    """

    def __init__(
        self,
        channel_in: int,
        hid_c1: int = 128,
        hid_c2: int = 128,
        n_t1: int = 1,
        n_t2: int = 2,
        d_state: int = 16,
        drop_path_rate: float = 0.1,
        layer_scale_init: float = 1e-4,
        **kwargs
    ):
        super().__init__()
        # Lazy import avoids circular dependency at module-load time
        from .simvp_modules import ConvSC

        self.channel_in = channel_in
        self.hid_c1 = hid_c1

        # Projections: channel_in <-> hid_c1  (per-spatial-frame with 1×1 conv)
        self.bottleneck_proj_in  = self._proj_block(channel_in, hid_c1)
        self.bottleneck_proj_out = self._proj_block(hid_c1, channel_in)

        # Initial STVMamba block
        self.initial_block = STVMambaModule(
            dim=hid_c1, d_state=d_state, drop_path=0.0,
            layer_scale_init=layer_scale_init)

        # Branch A1: n_t1 blocks at hid_c1, full bottleneck resolution
        self.a1_blocks = nn.ModuleList([
            STVMambaModule(dim=hid_c1, d_state=d_state, drop_path=0.0,
                           layer_scale_init=layer_scale_init)
            for _ in range(n_t1)
        ])

        # Branch A2: spatial down → n_t2 blocks at hid_c2 → up
        self.a2_downsample = ConvSC(hid_c1, hid_c2, kernel_size=3,
                                    downsampling=True, act_norm=True)
        self.a2_blocks = nn.ModuleList([
            STVMambaModule(dim=hid_c2, d_state=d_state, drop_path=0.0,
                           layer_scale_init=layer_scale_init)
            for _ in range(n_t2)
        ])
        self.a2_upsample = ConvSC(hid_c2, hid_c1, kernel_size=3,
                                  upsampling=True, act_norm=True)

        # Fusion: concat(A1, A2) → hid_c1 → final block
        self.fusion_conv = self._proj_block(2 * hid_c1, hid_c1)
        self.final_block  = STVMambaModule(
            dim=hid_c1, d_state=d_state, drop_path=0.0,
            layer_scale_init=layer_scale_init)

        # Weight initialisation
        self.apply(self._init_weights)

        # Linear drop-path schedule across all STVMamba blocks
        self._apply_drop_path_schedule(drop_path_rate)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _proj_block(in_ch: int, out_ch: int, groups: int = 8) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(_safe_num_groups(out_ch, groups), out_ch),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _apply_drop_path_schedule(self, drop_path_rate: float):
        blocks = (
            [self.initial_block]
            + list(self.a1_blocks)
            + list(self.a2_blocks)
            + [self.final_block]
        )
        total = len(blocks)
        if total == 0 or drop_path_rate <= 0:
            return
        dp_rates = torch.linspace(0, drop_path_rate, steps=total).tolist()
        for m, p in zip(blocks, dp_rates):
            m.drop_path = DropPath(p) if p > 0 else nn.Identity()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)  — C == channel_in (SimVP hid_S post-encoder)
        Returns:
            (B, T, C, H, W)     — same shape
        """
        B, T, C, H, W = x.shape

        # Project to bottleneck dimension (flatten T into batch)
        x_flat = x.reshape(B * T, C, H, W)
        x_flat = self.bottleneck_proj_in(x_flat)       # (B*T, hid_c1, H, W)
        _, C1, H_b, W_b = x_flat.shape
        x = x_flat.view(B, T, C1, H_b, W_b)

        # Initial block
        x = self.initial_block(x)                      # (B, T, hid_c1, H_b, W_b)

        # Snapshot for both branches
        x_a1 = x
        x_a2 = x

        # --- Branch A1: n_t1 blocks at full bottleneck resolution ---
        for blk in self.a1_blocks:
            x_a1 = blk(x_a1)

        # --- Branch A2: downsample → n_t2 blocks → upsample ---
        x_a2_flat = x_a2.reshape(B * T, C1, H_b, W_b)
        x_a2_flat = self.a2_downsample(x_a2_flat)      # (B*T, hid_c2, H_b/2, W_b/2)
        _, C2, H2, W2 = x_a2_flat.shape
        x_a2_down = x_a2_flat.view(B, T, C2, H2, W2)

        for blk in self.a2_blocks:
            x_a2_down = blk(x_a2_down)

        x_a2_up_flat = x_a2_down.reshape(B * T, C2, H2, W2)
        x_a2_up_flat = self.a2_upsample(x_a2_up_flat)  # (B*T, hid_c1, H_b, W_b)
        x_a2_up = x_a2_up_flat.view(B, T, C1, H_b, W_b)

        # --- Fusion: concatenate A1 + A2-up, project, final block ---
        x_cat = torch.cat(
            [x_a1.reshape(B * T, C1, H_b, W_b),
             x_a2_up.reshape(B * T, C1, H_b, W_b)], dim=1
        )                                               # (B*T, 2*hid_c1, H_b, W_b)
        x_fused = self.fusion_conv(x_cat)              # (B*T, hid_c1, H_b, W_b)
        x_fused = x_fused.view(B, T, C1, H_b, W_b)
        x = self.final_block(x_fused)                  # (B, T, hid_c1, H_b, W_b)

        # Project back to original channel_in
        x_out = x.reshape(B * T, C1, H_b, W_b)
        x_out = self.bottleneck_proj_out(x_out)        # (B*T, channel_in, H_b, W_b)
        x_out = x_out.view(B, T, C, H_b, W_b)
        return x_out


__all__ = ['STDSConv', 'STSS', 'SwiGLU', 'STVMambaModule', 'MidSTVMamba', 'MAMBA_AVAILABLE']
