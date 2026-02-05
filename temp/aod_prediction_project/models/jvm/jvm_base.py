"""
JvM Base Modules - Shared components for Just video Mamba architectures.

Contains:
- Time embeddings
- Patch embedding with linear bottleneck
- AdaLN modulation utilities
- SwiGLU
- Utility functions for modulation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple


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
    
    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.max_period = max_period
        
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor (B,) in [0, 1]
            
        Returns:
            Time embeddings (B, time_dim)
        """
        emb = sinusoidal_embedding(t, self.model_dim, self.max_period)
        return self.mlp(emb)


# ==============================================================================
# Patch Embedding with Linear Bottleneck
# ==============================================================================

class PatchEmbedBottleneck(nn.Module):
    """
    3D Patch Embedding with linear bottleneck for noise reduction.
    
    Philosophy: Instead of direct linear projection, we use a bottleneck
    (high_dim -> bottleneck_dim -> embed_dim) to force compact feature learning.
    
    Args:
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        patch_size: Tuple of (patch_t, patch_h, patch_w)
        bottleneck_dim: Bottleneck dimension (default: 128)
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
        
        # Linear bottleneck: high_dim -> bottleneck -> embed_dim
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
        
        # Check divisibility
        assert T % pt == 0 and H % ph == 0 and W % pw == 0, \
            f"Input size ({T}, {H}, {W}) not divisible by patch_size ({pt}, {ph}, {pw})"
        
        # Calculate grid size
        t = T // pt
        h = H // ph
        w = W // pw
        grid_size = (t, h, w)
        
        # Reshape into patches
        x = rearrange(
            x,
            'b (t pt) c (h ph) (w pw) -> b (t h w) (pt ph pw c)',
            pt=pt, ph=ph, pw=pw
        )
        
        # Apply linear bottleneck
        x = self.proj(x)
        
        # Apply RMSNorm
        x = self.norm(x)
        
        return x, grid_size


# ==============================================================================
# AdaLN Modulation
# ==============================================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulates the input tensor x with scale and shift (AdaLN).
    
    Args:
        x: Input tensor (B, T, C, H, W)
        shift: Shift tensor (B, C)
        scale: Scale tensor (B, C)
        
    Returns:
        Modulated tensor (B, T, C, H, W)
    """
    B, T, C, H, W = x.shape
    scale = scale.view(B, 1, C, 1, 1)
    shift = shift.view(B, 1, C, 1, 1)
    return x * (1 + scale) + shift


class AdaLNModulation(nn.Module):
    """
    AdaLN modulation layer for DiT-style conditioning.
    
    Predicts scale, shift, and gate parameters from time embedding.
    For JvM/JvTM: 6 parameters per block (2 branches × 3 params each).
    
    Args:
        time_dim: Input time embedding dimension
        model_dim: Model dimension
        num_params: Number of parameters (6 for scale, shift, gate × 2 branches)
    """
    
    def __init__(self, time_dim: int, model_dim: int, num_params: int = 6):
        super().__init__()
        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, num_params * model_dim, bias=True),
        )
        
        # Initialize to zero (identity at start)
        nn.init.zeros_(self.linear[-1].weight)
        nn.init.zeros_(self.linear[-1].bias)
    
    def forward(self, t_emb: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            t_emb: Time embedding (B, time_dim)
            
        Returns:
            Tuple of modulation parameters (each of shape (B, model_dim))
        """
        return self.linear(t_emb).chunk(6, dim=1)


# ==============================================================================
# SwiGLU
# ==============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU activation (better than standard GLU).
    
    Formula: SwiGLU(x) = (xW1 ⊙ SiLU(xW2))W3
    
    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension (default: 4 * dim)
        dropout: Dropout rate
    """
    
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(4 * dim)
        
        # Project to 2 * hidden_dim for gate and value
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (..., dim)
            
        Returns:
            Output tensor (..., dim)
        """
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.dropout(self.w3(hidden))


# ==============================================================================
# Unpatchify (Output Layer)
# ==============================================================================

class LinearUnpatchify(nn.Module):
    """
    Linear unpatchify layer with final modulation.
    
    Converts patches back to video format with AdaLN modulation.
    
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

        # Final smoothing Conv
        # Really light 3D conv to reduce block artifacts
        # Initialize identity & zeros to avoid affecting the beginning of training
        self.final_smooth = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # Initialize modulation to zero
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

        # Initialize final smoothing conv to zero
        nn.init.dirac_(self.final_smooth.weight) # Init as identity
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
        
        # Norm
        x = self.norm(x)
        
        # Modulate
        scale, shift = self.modulation(t_emb).chunk(2, dim=1)
        scale = scale.unsqueeze(1)  # (B, 1, D)
        shift = shift.unsqueeze(1)  # (B, 1, D)
        x = x * (1 + scale) + shift
        
        # Project to patches
        x = self.proj(x)  # (B, N, pt*ph*pw*C)
        
        # Reshape to video for convolution (B, C, T, H, W)
        x = rearrange(
            x,
            'b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)',
            t=t, h=h, w=w, pt=pt, ph=ph, pw=pw, c=self.out_channels
        )

        # Apply Smoothing
        x = self.final_smooth(x)

        # Rearrange back to (B, T, C, H, W)
        x = rearrange(x, 'b c t h w -> b t c h w')
        
        return x
