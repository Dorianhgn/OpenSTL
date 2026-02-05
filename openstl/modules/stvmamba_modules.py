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
from timm.layers import DropPath
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

        x = residual + self.drop_path(self.gamma1 * main)

        # Branch 2: SwiGLU FFN
        residual = x
        x2 = rearrange(x, 'b t c h w -> (b t h w) c')
        x2 = self.norm2(x2)
        x2 = self.ffn(x2)
        x2 = rearrange(x2, '(b t h w) c -> b t c h w', b=B, t=T, h=H, w=W)

        x = residual + self.drop_path(self.gamma2 * x2)

        return x


__all__ = ['STDSConv', 'STSS', 'SwiGLU', 'STVMambaModule', 'MAMBA_AVAILABLE']
