# Copyright (c) CAIRI AI Lab. All rights reserved
# SPADEJvM Model for Flow Matching with SPADE conditioning
# Ported from MambaFlow/JvM implementation

"""
SPADEJvM (Spatial Adaptive Just video Mamba) - SPADE-conditioned video generation.

Philosophy: Combines "Just Image" prediction with SPADE conditioning for strong spatial control.
- Direct x prediction (Just Image approach)
- SPADE (Spatial Adaptive Normalization) for spatial conditioning
- No Cross-Attention (replaced by multiplicative SPADE modulation)
- Gated residuals for controlled information flow

Architecture:
    Input (B, T, C, H, W) + Spatial Condition (B, Cond_C, H_cond, W_cond)
    → Patch Embed with Bottleneck
    → N × SPADEJvMBlock (SPADE + AdaLN + Backbone + SwiGLU)
    → Linear Unpatchify
    → Output (B, T, C, H, W) - Predicted clean image x

Supports backbone ablations: mamba, attention, conv
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from typing import Tuple, Optional, Literal

# Import STVMamba modules (with graceful fallback)
try:
    from openstl.modules.stvmamba_modules import STDSConv, STSS, SwiGLU, MAMBA_AVAILABLE
except ImportError:
    MAMBA_AVAILABLE = False
    STDSConv = None
    STSS = None
    from openstl.modules.stvmamba_modules import SwiGLU


# ==============================================================================
# Time Embeddings
# ==============================================================================

def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Sinusoidal time embedding.
    
    Args:
        t: Time tensor (B,) in [0, 1]
        dim: Embedding dimension
        max_period: Maximum period for frequencies
        
    Returns:
        Time embeddings (B, dim)
    """
    half_dim = dim // 2
    freq_const = math.log(max_period) / max(half_dim - 1, 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * (-freq_const))
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    """
    Time embedding module with MLP projection.
    
    Args:
        model_dim: Base dimension for sinusoidal embedding
        time_dim: Output dimension after MLP
        max_period: Maximum period for frequencies
    """
    
    def __init__(self, model_dim: int, time_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.model_dim = model_dim
        self.max_period = max_period
        
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding(t, self.model_dim, self.max_period)
        return self.mlp(emb)


# ==============================================================================
# Patch Embedding with Linear Bottleneck
# ==============================================================================

class PatchEmbedBottleneck(nn.Module):
    """
    3D Patch Embedding with linear bottleneck for noise reduction.
    
    Args:
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        patch_size: Tuple of (patch_t, patch_h, patch_w)
        bottleneck_dim: Bottleneck dimension
    """
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: Tuple[int, int, int] = (2, 16, 16),
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        patch_vol = patch_size[0] * patch_size[1] * patch_size[2]
        high_dim = in_channels * patch_vol
        
        self.proj = nn.Sequential(
            nn.Linear(high_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )
        
        self.norm = nn.RMSNorm(embed_dim, eps=1e-6)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Args:
            x: Input tensor (B, T, C, H, W)
            
        Returns:
            patches: (B, N, D) where N = num_patches
            grid_size: (t, h, w) dimensions after patching
        """
        B, T, C, H, W = x.shape
        pt, ph, pw = self.patch_size
        
        assert T % pt == 0 and H % ph == 0 and W % pw == 0, \
            f"Input size ({T}, {H}, {W}) not divisible by patch_size ({pt}, {ph}, {pw})"
        
        t = T // pt
        h = H // ph
        w = W // pw
        grid_size = (t, h, w)
        
        x = rearrange(
            x,
            'b (t pt) c (h ph) (w pw) -> b (t h w) (pt ph pw c)',
            pt=pt, ph=ph, pw=pw
        )
        
        x = self.proj(x)
        x = self.norm(x)
        
        return x, grid_size


# ==============================================================================
# Unpatchify (Output Layer)
# ==============================================================================

class LinearUnpatchify(nn.Module):
    """
    Linear unpatchify layer with final modulation.
    
    Args:
        model_dim: Model dimension
        time_dim: Time embedding dimension
        out_channels: Output channels
        patch_size: Tuple of (patch_t, patch_h, patch_w)
    """
    
    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        out_channels: int,
        patch_size: Tuple[int, int, int],
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        patch_vol = patch_size[0] * patch_size[1] * patch_size[2]
        output_dim = out_channels * patch_vol
        
        self.norm = nn.RMSNorm(model_dim, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2 * model_dim),
        )
        self.proj = nn.Linear(model_dim, output_dim)

        # Final smoothing Conv (light 3D conv to reduce block artifacts)
        self.final_smooth = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # Initialize modulation and smoothing to small values
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.dirac_(self.final_smooth.weight)
        nn.init.zeros_(self.final_smooth.bias)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Args:
            x: Input patches (B, N, D)
            t_emb: Time embedding (B, time_dim)
            grid_size: (t, h, w) grid dimensions
            
        Returns:
            Video tensor (B, T, C, H, W)
        """
        B, N, D = x.shape
        t, h, w = grid_size
        pt, ph, pw = self.patch_size
        
        x = self.norm(x)
        
        scale, shift = self.modulation(t_emb).chunk(2, dim=1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        x = x * (1 + scale) + shift
        
        x = self.proj(x)
        
        x = rearrange(
            x,
            'b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)',
            t=t, h=h, w=w, pt=pt, ph=ph, pw=pw, c=self.out_channels
        )

        x = self.final_smooth(x)
        x = rearrange(x, 'b c t h w -> b t c h w')
        
        return x


# ==============================================================================
# Modulation Layers (SPADE and AdaLN)
# ==============================================================================

class VideoAdaLN(nn.Module):
    """Video-compatible AdaLN modulation (time-based only, no spatial conditioning)."""
    
    def __init__(self, time_dim: int, channels: int):
        super().__init__()
        self.norm = nn.RMSNorm(channels, elementwise_affine=False, eps=1e-6)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * channels))
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
            t_emb: (B, time_dim)
        Returns:
            Modulated x (B, T, C, H, W)
        """
        emb = self.emb(t_emb)
        scale, shift = emb.chunk(2, dim=1)
        
        x = rearrange(x, "b t c h w -> b t h w c")
        x = self.norm(x)
        x = rearrange(x, "b t h w c -> b t c h w")
        
        scale = scale.view(-1, 1, scale.size(1), 1, 1)
        shift = shift.view(-1, 1, shift.size(1), 1, 1)
        return x * (1 + scale) + shift


class SPADENorm(nn.Module):
    """
    Spatial Adaptive Normalization (SPADE).
    
    Args:
        norm_channels: Number of channels to normalize
        cond_channels: Number of condition channels
        hidden_channels: Hidden channels in SPADE conv network
        kernel_size: Kernel size for SPADE convs
    """
    
    def __init__(
        self,
        norm_channels: int,
        cond_channels: int,
        hidden_channels: int = 128,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.norm_channels = norm_channels
        
        self.norm = nn.RMSNorm(norm_channels, elementwise_affine=False, eps=1e-6)
        
        padding = kernel_size // 2
        self.spade_net = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_channels, kernel_size, padding=padding),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, norm_channels * 2, kernel_size, padding=padding),
        )
        
        nn.init.normal_(self.spade_net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.spade_net[-1].bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map (B, C, H, W) or (B, T, C, H, W)
            cond: Condition map (B, Cond_C, H_cond, W_cond)
            
        Returns:
            Modulated features (same shape as x)
        """
        is_video = x.ndim == 5
        if is_video:
            B, T, C, H, W = x.shape
            x = rearrange(x, "b t c h w -> (b t) c h w")
            
            if cond.ndim == 5:
                cond = rearrange(cond, "b t c h w -> (b t) c h w")
            elif cond.ndim == 4:
                cond = cond.unsqueeze(1).repeat(1, T, 1, 1, 1)
                cond = rearrange(cond, "b t c h w -> (b t) c h w")
        else:
            B, C, H, W = x.shape
            if cond.ndim == 5:
                cond = cond.mean(dim=1)
        
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x_norm = rearrange(x, "b c h w -> b h w c")
        x_norm = self.norm(x_norm)
        x_norm = rearrange(x_norm, "b h w c -> b c h w")
        
        spade_params = self.spade_net(cond)
        gamma, beta = spade_params.chunk(2, dim=1)
        
        out = x_norm * (1 + gamma) + beta   # Formule pour démarrer l'entraînement en identité
        
        if is_video:
            out = rearrange(out, "(b t) c h w -> b t c h w", b=B, t=T)
        
        return out


class SPADEAdaLNModulation(nn.Module):
    """Combined SPADE + Time-based AdaLN modulation."""
    
    def __init__(self, time_dim: int, channels: int, cond_channels: int):
        super().__init__()
        self.spade = SPADENorm(channels, cond_channels, hidden_channels=128)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2 * channels),
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features (B, T, C, H, W)
            cond: Spatial condition (B, Cond_C, H_cond, W_cond)
            t_emb: Time embedding (B, time_dim)
            
        Returns:
            Modulated features (B, T, C, H, W)
        """
        x = self.spade(x, cond)
        
        time_params = self.time_mlp(t_emb)
        gamma_t, beta_t = time_params.chunk(2, dim=1)
        
        gamma_t = gamma_t.view(-1, 1, gamma_t.size(1), 1, 1)
        beta_t = beta_t.view(-1, 1, beta_t.size(1), 1, 1)
        
        x = x * (1 + gamma_t) + beta_t
        
        return x


# ==============================================================================
# 3D RoPE (from Kandinsky)
# ==============================================================================

def get_freqs(dim: int, max_period: float = 10000.0) -> torch.Tensor:
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)
    return freqs

def apply_rotary(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    # x:    (B, L, num_heads, head_dim)
    # rope: (L, 1, head_dim//2, 2, 2)  -- 2x2 rotation matrix per frequency
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2)  # (B, L, num_heads, head_dim//2, 1, 2)
    rope = rope.unsqueeze(0)                  # (1, L, 1,         head_dim//2, 2, 2)
    x_out = (rope * x_).sum(dim=-1)           # (B, L, num_heads, head_dim//2, 2)
    return x_out.reshape(*x.shape).type_as(x)

class RoPE3D(nn.Module):
    def __init__(self, axes_dims: Tuple[int, int, int], max_pos: Tuple[int, int, int] = (128, 128, 128), max_period: float = 10000.0):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    def forward(self, shape: Tuple[int, int, int], pos: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], scale_factor: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        duration, height, width = shape
        args_t = getattr(self, "args_0")[pos[0]] / scale_factor[0]
        args_h = getattr(self, "args_1")[pos[1]] / scale_factor[1]
        args_w = getattr(self, "args_2")[pos[2]] / scale_factor[2]

        args = torch.cat([
            args_t.view(duration, 1, 1, -1).repeat(1, height, width, 1),
            args_h.view(1, height, 1, -1).repeat(duration, 1, width, 1),
            args_w.view(1, 1, width, -1).repeat(duration, height, 1, 1),
        ], dim=-1)
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)  # (T, H, W, head_dim//2, 4)
        rope = rope.view(*rope.shape[:-1], 2, 2)                     # (T, H, W, head_dim//2, 2, 2)
        # flatten (T, H, W) and add head broadcast dim
        rope = rope.reshape(-1, rope.shape[-3], 2, 2)                # (T*H*W, head_dim//2, 2, 2)
        rope = rope.unsqueeze(1)                                     # (T*H*W, 1, head_dim//2, 2, 2)
        return rope


# ==============================================================================
# Alternative Backbone Blocks for Ablation
# ==============================================================================

class ConvBlock(nn.Module):
    """Convolutional backbone block (for ablation: block_type='conv')."""
    
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, T, C, H, W). Returns: (B, T, C, H, W)."""
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.conv(x)
        x = rearrange(x, 'b c t h w -> b t c h w')
        return x


class AttentionBlock(nn.Module):
    """Self-attention backbone block with RoPE3D and RMSNorm on Q/K (like Kandinsky)."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.norm = nn.LayerNorm(dim)
        
        self.to_query = nn.Linear(dim, dim, bias=True)
        self.to_key = nn.Linear(dim, dim, bias=True)
        self.to_value = nn.Linear(dim, dim, bias=True)
        
        # Kandinsky-style QK RMSNorm for improved stability
        self.query_norm = nn.RMSNorm(self.head_dim)
        self.key_norm = nn.RMSNorm(self.head_dim)
        
        self.use_rope = use_rope
        if self.use_rope:
            # Distribute head_dim across T, H, W (e.g. 1/3 each roughly, must be even)
            # Example heuristic: T gets 32%, H gets 34%, W gets 34%
            d_t = 2 * (int(self.head_dim * 0.33) // 2)
            d_h = 2 * (int(self.head_dim * 0.33) // 2)
            d_w = self.head_dim - (d_t + d_h)
            self.rope3d = RoPE3D(axes_dims=(d_t, d_h, d_w))
            
        self.out_layer = nn.Linear(dim, dim, bias=True)
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, T, C, H, W). Returns: (B, T, C, H, W)."""
        B, T, C, H, W = x.shape
        
        x_flat = rearrange(x, 'b t c h w -> b (t h w) c')
        x_norm = self.norm(x_flat)
        
        q = self.to_query(x_norm).view(B, -1, self.num_heads, self.head_dim)
        k = self.to_key(x_norm).view(B, -1, self.num_heads, self.head_dim)
        v = self.to_value(x_norm).view(B, -1, self.num_heads, self.head_dim)
        
        # Apply RMSNorm to Q and K
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        
        if self.use_rope:
            device = x.device
            pos = (
                torch.arange(T, device=device),
                torch.arange(H, device=device),
                torch.arange(W, device=device)
            )
            # rope shape: (T*H*W, 1, head_dim//2, 2)
            rope = self.rope3d(shape=(T, H, W), pos=pos).to(device)
            q = apply_rotary(q, rope)
            k = apply_rotary(k, rope)
            
        # Reshape for multi-head attention (B, num_heads, L, head_dim)
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')
        
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        attn_out = rearrange(attn_out, 'b h l d -> b l (h d)')
        
        x_out = self.out_layer(attn_out)
        x_out = rearrange(x_out, 'b (t h w) c -> b t c h w', t=T, h=H, w=W)
        return x_out


class MambaBlock(nn.Module):
    """Mamba backbone block (default: block_type='mamba')."""
    
    def __init__(self, dim: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("Mamba not available. Install with: pip install mamba-ssm[causal-conv1d]")
        self.st_conv = STDSConv(dim)
        self.st_ss = STSS(dim, d_state, d_conv, expand)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, T, C, H, W). Returns: (B, T, C, H, W)."""
        x = self.st_conv(x)
        x = self.st_ss(x)
        return x


# ==============================================================================
# SPADEJvM Block (Core Building Block)
# ==============================================================================

class SPADEJvMBlock(nn.Module):
    """
    Hybrid SPADEJvM block with configurable backbone.
    
    Args:
        dim: Model dimension
        cond_channels: Spatial condition channels (0 = VideoAdaLN, >0 = SPADE)
        time_dim: Time embedding dimension
        block_type: Backbone type ('mamba', 'attention', 'conv')
        d_state: Mamba state dimension (if block_type='mamba')
        d_conv: Mamba convolution size (if block_type='mamba')
        expand: Mamba expansion factor (if block_type='mamba')
        num_heads: Number of attention heads (if block_type='attention')
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        cond_channels: int,
        time_dim: int,
        block_type: Literal["mamba", "attention", "conv"] = "mamba",
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_spade = cond_channels > 0
        
        # Modulation layers
        if self.use_spade:
            self.modulation1 = SPADEAdaLNModulation(time_dim, dim, cond_channels)
            self.modulation2 = SPADEAdaLNModulation(time_dim, dim, cond_channels)
        else:
            self.modulation1 = VideoAdaLN(time_dim, dim)
            self.modulation2 = VideoAdaLN(time_dim, dim)
        
        # Configurable backbone for spatio-temporal mixing
        if block_type == "mamba":
            self.backbone = MambaBlock(dim, d_state, d_conv, expand)
        elif block_type == "attention":
            self.backbone = AttentionBlock(dim, num_heads, dropout, use_rope=use_rope)
        elif block_type == "conv":
            self.backbone = ConvBlock(dim)
        else:
            raise ValueError(f"Unknown block_type: {block_type}")
        
        # Channel Mixing (SwiGLU)
        self.mlp = SwiGLU(dim, dim * 4, dropout=dropout)
        
        # Gated residual parameters
        self.gate_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2 * dim),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        cond_spatial: Optional[torch.Tensor],
        t_emb: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Args:
            x: Patches (B, N, D) where N = t*h*w
            cond_spatial: Spatial condition (B, Cond_C, H_cond, W_cond) or None
            t_emb: Time embedding (B, time_dim)
            grid_size: (t, h, w) grid dimensions after patching
            
        Returns:
            Output patches (B, N, D)
        """
        B, N, D = x.shape
        t, h, w = grid_size
        
        gates = self.gate_mlp(t_emb)
        gate1, gate2 = gates.chunk(2, dim=1)
        
        # Branch 1: Spatio-Temporal Mixing
        residual = x
        
        x_video = rearrange(x, 'b (t h w) d -> b t d h w', t=t, h=h, w=w)
        if self.use_spade:
            x_mod = self.modulation1(x_video, cond_spatial, t_emb)
        else:
            x_mod = self.modulation1(x_video, t_emb)
        
        x_out = self.backbone(x_mod)
        x_out = rearrange(x_out, 'b t d h w -> b (t h w) d')
        
        x = residual + gate1.unsqueeze(1) * x_out
        
        # Branch 2: Channel Mixing
        residual = x
        
        x_video = rearrange(x, 'b (t h w) d -> b t d h w', t=t, h=h, w=w)
        if self.use_spade:
            x_mod = self.modulation2(x_video, cond_spatial, t_emb)
        else:
            x_mod = self.modulation2(x_video, t_emb)
        
        x_mod = rearrange(x_mod, 'b t d h w -> b (t h w) d')
        x_out = self.mlp(x_mod)
        
        x = residual + gate2.unsqueeze(1) * x_out
        
        return x


# ==============================================================================
# SPADEJvM Model (Full Architecture)
# ==============================================================================

class SPADEJvM_Model(nn.Module):
    """
    Hybrid SPADEJvM model for Flow Matching.
    
    Supports:
    - SPADE or VideoAdaLN conditioning (use_spade ablation)
    - Multiple backbone types: mamba, attention, conv (backbone ablation)
    - RoPE positional encoding (use_rope ablation - placeholder)
    
    Args:
        in_shape: Input shape (T_in, C, H, W) - for OpenSTL compatibility
        in_channels: Input/output channels
        model_dim: Model dimension
        cond_channels: Spatial condition channels (0 = no SPADE)
        time_dim: Time embedding dimension
        num_blocks: Number of SPADEJvM blocks
        patch_size: Patch size (T, H, W)
        bottleneck_dim: Bottleneck dimension for patch embedding
        pre_seq_length: Number of input frames (condition)
        aft_seq_length: Number of output frames (prediction)
        block_type: Backbone type ('mamba', 'attention', 'conv')
        use_spade: Whether to use SPADE conditioning (else concatenate)
        use_rope: Whether to use RoPE positional encoding (placeholder)
        d_state: Mamba state dimension
        d_conv: Mamba convolution size
        expand: Mamba expansion factor
        num_heads: Number of attention heads
        dropout: Dropout rate
        gradient_checkpointing: Enable gradient checkpointing
    """
    
    def __init__(
        self,
        in_shape: Tuple[int, int, int, int] = (10, 1, 64, 64),
        in_channels: int = 1,
        model_dim: int = 256,
        cond_channels: int = 10,
        time_dim: int = 512,
        num_blocks: int = 12,
        patch_size: Tuple[int, int, int] = (2, 8, 8),
        bottleneck_dim: int = 128,
        pre_seq_length: int = 10,
        aft_seq_length: int = 10,
        block_type: Literal["mamba", "attention", "conv"] = "mamba",
        use_spade: bool = True,
        use_rope: bool = True,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        gradient_checkpointing: bool = True,
        **kwargs,
    ):
        super().__init__()
        
        # Parse in_shape for OpenSTL compatibility
        if in_shape is not None:
            T_in, C, H, W = in_shape
            in_channels = C
            pre_seq_length = T_in
        
        self.in_channels = in_channels
        self.model_dim = model_dim
        self.cond_channels = cond_channels if use_spade else 0
        self.num_blocks = num_blocks
        self.patch_size = tuple(patch_size) if isinstance(patch_size, list) else patch_size
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.block_type = block_type
        self.use_spade = use_spade
        self.use_rope = use_rope
        self.gradient_checkpointing = gradient_checkpointing
        
        # Time embedding
        self.time_emb = TimeEmbedding(model_dim, time_dim)
        
        # Patch embedding with bottleneck
        self.patch_embed = PatchEmbedBottleneck(
            in_channels=in_channels,
            embed_dim=model_dim,
            patch_size=self.patch_size,
            bottleneck_dim=bottleneck_dim,
        )
        
        # SPADEJvM blocks
        self.blocks = nn.ModuleList([
            SPADEJvMBlock(
                dim=model_dim,
                cond_channels=self.cond_channels,
                time_dim=time_dim,
                block_type=block_type,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                num_heads=num_heads,
                dropout=dropout,
                use_rope=self.use_rope,
            )
            for _ in range(num_blocks)
        ])
        
        # Output unpatchify
        self.unpatchify = LinearUnpatchify(
            model_dim=model_dim,
            time_dim=time_dim,
            out_channels=in_channels,
            patch_size=self.patch_size,
        )

    def _pad_time_to_patch_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Pad the temporal dimension so it becomes divisible by the temporal patch size.

        Returns:
            padded_x: Tensor with temporal length divisible by patch_size[0]
            original_t: Original temporal length before padding
        """
        pt = self.patch_size[0]
        original_t = x.shape[1]
        pad_t = (-original_t) % pt
        if pad_t == 0:
            return x, original_t

        pad_frame = torch.zeros_like(x[:, -1:, ...])
        pad = pad_frame.repeat(1, pad_t, 1, 1, 1)
        x = torch.cat([x, pad], dim=1)
        return x, original_t

    def _crop_time(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Crop the temporal dimension back to length t."""
        return x[:, :t, ...]
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_spatial: Optional[torch.Tensor] = None,
        x_past: Optional[torch.Tensor] = None,
        label_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Flow Matching.
        
        Args:
            x_t: Noisy input at time t (B, T_out, C, H, W)
            t: Time (B,) in [0, 1]
            cond_spatial: Processed spatial condition (B, Cond_C, H, W) for SPADE.
                         Required if use_spade=True, ignored if use_spade=False.
            x_past: Clean past frames (B, T_in, C, H, W), optional.
                    - If use_spade=True: concatenated with x_t on time dim
                    - If use_spade=False: concatenated before x_t (TODO: [cond, x_past, x_t])
                label_emb: Optional global label embedding (B, time_dim) added to time embedding.
            
        Returns:
            Predicted clean image (B, T_out, C, H, W)
        """
        B, T_out, C, H, W = x_t.shape
        
        # Time embedding
        t_emb = self.time_emb(t)
        if label_emb is not None:
            t_emb = t_emb + label_emb
        
        # Validate SPADE condition
        if self.use_spade and cond_spatial is None:
            raise ValueError(
                "use_spade=True requires cond_spatial. "
                "Either provide cond_spatial or set use_spade=False."
            )
        
        # Concatenate past frames with noisy input on time dimension
        if x_past is not None:
            x_input = torch.cat([x_past, x_t], dim=1)  # (B, T_in + T_out, C, H, W)
        else:
            x_input = x_t

        # Pad temporal length to a multiple of the temporal patch size.
        # This is important for SEVIR, where 13 past + 12 future = 25 frames.
        x_input, original_t = self._pad_time_to_patch_multiple(x_input)
        
        # Patch embedding
        x, grid_size = self.patch_embed(x_input)
        
        # Apply SPADEJvM blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cond_spatial, t_emb, grid_size, use_reentrant=False)
            else:
                x = block(x, cond_spatial, t_emb, grid_size)
        
        # Unpatchify to output
        x_pred = self.unpatchify(x, t_emb, grid_size)

        # Remove any temporal padding that was added before patch embedding.
        x_pred = self._crop_time(x_pred, original_t)
        
        # Extract only the future portion if past was concatenated
        if x_past is not None:
            T_in = x_past.shape[1]
            x_pred = x_pred[:, T_in:, :, :, :]
        
        return x_pred
    
    def predict_x_from_xt(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        x_past: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for forward() - compatibility with JvM interface."""
        return self.forward(x_t, t, cond_spatial=cond, x_past=x_past)
    
    def compute_v_from_x_pred(
        self,
        x_pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity v from predicted clean image.
        
        CFM formula: v(x_t, t) = (x1 - x_t) / (1 - t)
        """
        t_expanded = t.view(-1, 1, 1, 1, 1)
        t_safe = torch.clamp(t_expanded, max=1.0 - 1e-5)
        v = (x_pred - x_t) / (1.0 - t_safe)
        return v


# ==============================================================================
# Factory Function
# ==============================================================================

def create_spade_jvm_model(
    config: str = "base",
    in_shape: Tuple[int, int, int, int] = (10, 1, 64, 64),
    cond_channels: int = 10,
    block_type: str = "mamba",
    use_spade: bool = True,
    **kwargs,
) -> SPADEJvM_Model:
    """
    Factory function to create SPADEJvM models with different configurations.
    
    Args:
        config: Configuration preset ("tiny", "small", "base", "large")
        in_shape: Input shape (T, C, H, W)
        cond_channels: Spatial condition channels
        block_type: Backbone type ('mamba', 'attention', 'conv')
        use_spade: Whether to use SPADE conditioning
        **kwargs: Additional arguments override config
        
    Returns:
        SPADEJvM_Model instance
    """
    configs = {
        "tiny": {
            "model_dim": 192,
            "time_dim": 384,
            "num_blocks": 8,
            "d_state": 32,
            "patch_size": (2, 8, 8),
        },
        "small": {
            "model_dim": 256,
            "time_dim": 512,
            "num_blocks": 12,
            "d_state": 64,
            "patch_size": (2, 8, 8),
        },
        "base": {
            "model_dim": 384,
            "time_dim": 768,
            "num_blocks": 16,
            "d_state": 64,
            "patch_size": (2, 8, 8),
        },
        "large": {
            "model_dim": 512,
            "time_dim": 1024,
            "num_blocks": 20,
            "d_state": 128,
            "patch_size": (2, 8, 8),
        },
    }
    
    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")
    
    cfg = configs[config]
    cfg.update(kwargs)
    
    return SPADEJvM_Model(
        in_shape=in_shape,
        cond_channels=cond_channels,
        block_type=block_type,
        use_spade=use_spade,
        **cfg
    )
