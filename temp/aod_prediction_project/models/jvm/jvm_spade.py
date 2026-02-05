"""
SPADEJvM (Spatial Adaptive Just video Mamba) - SPADE-conditioned video generation.

Philosophy: Combines "Just Image" prediction with SPADE conditioning for strong spatial control.
- Direct x prediction (Just Image approach)
- SPADE (Spatial Adaptive Normalization) for spatial conditioning
- No Cross-Attention (replaced by multiplicative SPADE modulation)
- RoPE3D for geometric position encoding
- Gated residuals for controlled information flow

Architecture:
    Input (B, T, C, H, W) + Spatial Condition (B, Cond_C, H_cond, W_cond)
    → Patch Embed with Bottleneck
    → N × SPADEJvMBlock (SPADE + AdaLN + STDSConv + STSS + SwiGLU)
    → Linear Unpatchify
    → Output (B, T, C, H, W) - Predicted clean image x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from typing import Tuple, Optional

from models.jvm.jvm_base import (
    TimeEmbedding,
    PatchEmbedBottleneck,
    SwiGLU,
    LinearUnpatchify,
)
from models.stvmamba_modules import STDSConv, STSS
from models.cfm.velocity_net_spade import SPADENorm, SPADEAdaLNModulation, VideoAdaLN


# ==============================================================================
# SPADEJvM Block (Core Building Block)
# ==============================================================================

class SPADEJvMBlock(nn.Module):
    """
    Hybrid SPADEJvM block - Supports SPADE or VideoAdaLN conditioning.
    
    Architecture:
    1. Branch 1 (Spatial-Temporal Mixing):
       - SPADE/VideoAdaLN modulation
       - STDSConv (local coherence)
       - STSS (global 4-direction scan)
       - Gated residual
    
    2. Branch 2 (Channel Mixing):
       - SPADE/VideoAdaLN modulation
       - SwiGLU (channel mixing)
       - Gated residual
    
    Args:
        dim: Model dimension
        cond_channels: Spatial condition channels (0 = VideoAdaLN, >0 = SPADE)
        time_dim: Time embedding dimension
        d_state: Mamba state dimension
        d_conv: Mamba convolution size
        expand: Mamba expansion factor
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        cond_channels: int,
        time_dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.use_spade = cond_channels > 0
        
        # Modulation layers (SPADE if cond available, else VideoAdaLN)
        if self.use_spade:
            self.modulation1 = SPADEAdaLNModulation(time_dim, dim, cond_channels)
            self.modulation2 = SPADEAdaLNModulation(time_dim, dim, cond_channels)
        else:
            self.modulation1 = VideoAdaLN(time_dim, dim)
            self.modulation2 = VideoAdaLN(time_dim, dim)
        
        # Spatio-Temporal Mixing (Mamba)
        self.st_conv = STDSConv(dim)
        self.st_ss = STSS(dim, d_state, d_conv, expand)
        
        # Channel Mixing (SwiGLU)
        self.mlp = SwiGLU(dim, dim * 4, dropout=dropout)
        
        # Gated residual parameters (from time embedding)
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
        
        # Get gate parameters
        gates = self.gate_mlp(t_emb)
        gate1, gate2 = gates.chunk(2, dim=1)  # Each (B, D)
        
        # Branch 1: Spatio-Temporal Mixing
        residual = x
        
        # Reshape to video format for modulation
        x_video = rearrange(x, 'b (t h w) d -> b t d h w', t=t, h=h, w=w)
        if self.use_spade:
            x_mod = self.modulation1(x_video, cond_spatial, t_emb)
        else:
            x_mod = self.modulation1(x_video, t_emb)
        
        # STDSConv + STSS
        x_out = self.st_conv(x_mod)
        x_out = self.st_ss(x_out)
        
        # Back to patches
        x_out = rearrange(x_out, 'b t d h w -> b (t h w) d')
        
        # Gated residual
        x = residual + gate1.unsqueeze(1) * x_out
        
        # Branch 2: Channel Mixing
        residual = x
        
        # Reshape for modulation
        x_video = rearrange(x, 'b (t h w) d -> b t d h w', t=t, h=h, w=w)
        if self.use_spade:
            x_mod = self.modulation2(x_video, cond_spatial, t_emb)
        else:
            x_mod = self.modulation2(x_video, t_emb)
        
        # Back to patches for MLP
        x_mod = rearrange(x_mod, 'b t d h w -> b (t h w) d')
        x_out = self.mlp(x_mod)
        
        # Gated residual
        x = residual + gate2.unsqueeze(1) * x_out
        
        return x


# ==============================================================================
# SPADEJvM Model (Full Architecture)
# ==============================================================================

class SPADEJvMModel(nn.Module):
    """
    Hybrid SPADEJvM - Supports SPADE or VideoAdaLN conditioning.
    
    Input Strategy:
    - Concatenates x_past (clean, T_in) + x_noisy (future, T_out) on time dim
    - Total input time: T = T_in + T_out
    - Loss computed only on future part (T_out)
    
    Conditioning:
    - If cond provided → SPADE modulation (spatial conditioning)
    - If cond = None → VideoAdaLN modulation (time-only)
    
    Args:
        in_channels: Input channels
        model_dim: Model dimension
        cond_channels: Spatial condition channels (0 = VideoAdaLN, >0 = SPADE)
        time_dim: Time embedding dimension
        num_blocks: Number of SPADEJvM blocks
        patch_size: Patch size (T, H, W)
        bottleneck_dim: Bottleneck dimension for patch embedding
        t_in: Number of past frames
        t_out: Number of future frames
        d_state: Mamba state dimension
        d_conv: Mamba convolution size
        expand: Mamba expansion factor
        dropout: Dropout rate
        gradient_checkpointing: Enable gradient checkpointing
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        model_dim: int = 256,
        cond_channels: int = 10,  # 0 for VideoAdaLN, >0 for SPADE
        time_dim: int = 512,
        num_blocks: int = 12,
        patch_size: Tuple[int, int, int] = (2, 16, 16),
        bottleneck_dim: int = 128,
        t_in: int = 8,  # Number of past frames
        t_out: int = 24,  # Number of future frames
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_dim = model_dim
        self.cond_channels = cond_channels
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.t_in = t_in
        self.t_out = t_out
        self.gradient_checkpointing = gradient_checkpointing
        
        # Time embedding
        self.time_emb = TimeEmbedding(model_dim, time_dim)
        
        # Patch embedding with bottleneck
        self.patch_embed = PatchEmbedBottleneck(
            in_channels=in_channels,
            embed_dim=model_dim,
            patch_size=patch_size,
            bottleneck_dim=bottleneck_dim,
        )
        
        # SPADEJvM blocks
        self.blocks = nn.ModuleList([
            SPADEJvMBlock(
                dim=model_dim,
                cond_channels=cond_channels,
                time_dim=time_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])
        
        # Output unpatchify
        self.unpatchify = LinearUnpatchify(
            model_dim=model_dim,
            time_dim=time_dim,
            out_channels=in_channels,
            patch_size=patch_size,
        )
    
    def forward(
        self,
        x_t: torch.Tensor,
        cond_spatial: Optional[torch.Tensor],
        t: torch.Tensor,
        x_past: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with past+future input strategy.
        
        Args:
            x_t: Noisy future input (B, T_out, C, H, W)
            cond_spatial: Spatial condition (B, Cond_C, H_cond, W_cond) or None
            t: Time (B,) in [0, 1]
            x_past: Clean past frames (B, T_in, C, H, W) or None
            
        Returns:
            Predicted clean image for future part only (B, T_out, C, H, W)
        """
        B, T_out, C, H, W = x_t.shape
        
        # Time embedding
        t_emb = self.time_emb(t)  # (B, time_dim)
        
        # Concatenate past + noisy future on time dimension
        if x_past is not None:
            x_input = torch.cat([x_past, x_t], dim=1)  # (B, T_in+T_out, C, H, W)
        else:
            x_input = x_t  # Fallback: only future
        
        # Patch embedding
        x, grid_size = self.patch_embed(x_input)  # (B, N, D), (t, h, w)
        t_patches, h_patches, w_patches = grid_size
        
        # Apply hybrid SPADEJvM blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cond_spatial, t_emb, grid_size, use_reentrant=False)
            else:
                x = block(x, cond_spatial, t_emb, grid_size)
        
        # Unpatchify to output
        x_pred = self.unpatchify(x, t_emb, grid_size)  # (B, T_total, C, H, W)
        
        # Extract future part only
        if x_past is not None:
            # Compute T_in_patches based on patch_size
            t_patch_size = self.patch_size[0]
            t_in_patches = self.t_in // t_patch_size
            # Split at frame level
            x_pred = x_pred[:, self.t_in:, :, :, :]  # Keep only T_out frames
        
        return x_pred
    
    def predict_x_from_xt(
        self,
        x_t: torch.Tensor,
        cond_spatial: Optional[torch.Tensor],
        t: torch.Tensor,
        x_past: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict clean image x from noisy x_t.
        Alias for forward() for compatibility with JvM interface.
        """
        return self.forward(x_t, cond_spatial, t, x_past)
    
    def compute_v_from_x_pred(
        self,
        x_pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity v from predicted clean image.
        
        CFM formula: v(x_t, t) = (x1 - x_t) / (1 - t)
        
        Args:
            x_pred: Predicted clean image (B, T, C, H, W)
            x_t: Noisy input (B, T, C, H, W)
            t: Time (B,) in [0, 1]
            
        Returns:
            Velocity (B, T, C, H, W)
        """
        t_expanded = t.view(-1, 1, 1, 1, 1)  # (B, 1, 1, 1, 1)
        t_safe = torch.clamp(t_expanded, min=1e-5)
        v = (x_pred - x_t) / (1.0 - t_safe)
        return v


# ==============================================================================
# Factory Function
# ==============================================================================

def create_spade_jvm_model(
    config: str = "base",
    in_channels: int = 4,
    cond_channels: int = 10,
    **kwargs,
) -> SPADEJvMModel:
    """
    Factory function to create SPADEJvM models with different configurations.
    
    Args:
        config: Configuration preset ("tiny", "small", "base", "large")
        in_channels: Input channels
        cond_channels: Spatial condition channels
        **kwargs: Additional arguments override config
        
    Returns:
        SPADEJvMModel instance
    """
    configs = {
        "tiny": {
            "model_dim": 192,
            "time_dim": 384,
            "num_blocks": 8,
            "d_state": 32,
            "patch_size": (2, 16, 16),
        },
        "small": {
            "model_dim": 256,
            "time_dim": 512,
            "num_blocks": 12,
            "d_state": 64,
            "patch_size": (2, 16, 16),
        },
        "base": {
            "model_dim": 384,
            "time_dim": 768,
            "num_blocks": 16,
            "d_state": 64,
            "patch_size": (2, 16, 16),
        },
        "large": {
            "model_dim": 512,
            "time_dim": 1024,
            "num_blocks": 20,
            "d_state": 128,
            "patch_size": (2, 16, 16),
        },
    }
    
    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")
    
    cfg = configs[config]
    cfg.update(kwargs)
    
    return SPADEJvMModel(in_channels=in_channels, cond_channels=cond_channels, **cfg)
