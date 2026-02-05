"""
SPADE-based Velocity Network for Conditional Flow Matching.

Replaces Cross-Attention with Spatial Adaptive Layer Normalization (SPADE)
for stronger, spatially-aware condition injection.

SPADE: "Semantic Image Synthesis with Spatially-Adaptive Normalization"
(Park et al., 2019) - https://arxiv.org/abs/1903.07291

Key Idea:
- Instead of global AdaLN (γ, β per channel), use spatial modulation (γ, β per pixel)
- Condition (weather maps) directly controls normalization parameters at each spatial location
- Multiplicative control = Model cannot ignore condition (unlike additive cross-attention)

Architecture:
    Input: (B, T, C, H, W) latent video + (B, Cond, H, W) spatial condition
    → SPADE-Mamba blocks (STDSConv + STSS + MLP with spatial modulation)
    → Output: (B, T, C, H, W) predicted velocity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
import math
from typing import Literal, Optional

from models.stvmamba_modules import STDSConv, STSS


# ==============================================================================
# Core SPADE Modules (with Hybrid SPADE/VideoAdaLN support)
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
        scale, shift = emb.chunk(2, dim=1)  # Each (B, C)
        
        # Apply norm on channels
        x = rearrange(x, "b t c h w -> b t h w c")
        x = self.norm(x)
        x = rearrange(x, "b t h w c -> b t c h w")
        
        # Modulate
        scale = scale.view(-1, 1, scale.size(1), 1, 1)  # (B, 1, C, 1, 1)
        shift = shift.view(-1, 1, shift.size(1), 1, 1)
        return x * (1 + scale) + shift


# ==============================================================================
# Core SPADE Modules
# ==============================================================================

class SPADENorm(nn.Module):
    """
    Spatial Adaptive Normalization (SPADE).
    
    Takes a spatial condition map and produces pixel-wise normalization parameters.
    
    Args:
        norm_channels: Number of channels to normalize
        cond_channels: Number of condition channels
        hidden_channels: Hidden channels in SPADE conv network (default: 128)
        kernel_size: Kernel size for SPADE convs (default: 3)
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
        
        # Normalization (without learnable affine parameters)
        self.norm = nn.RMSNorm(norm_channels, elementwise_affine=False, eps=1e-6)
        
        # SPADE network: condition → (γ, β)
        # Small CNN to process condition and output modulation parameters
        padding = kernel_size // 2
        self.spade_net = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_channels, kernel_size, padding=padding),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, norm_channels * 2, kernel_size, padding=padding),
        )
        
        # Initialize to produce small γ and β at start (near-identity modulation)
        # Using small values instead of zeros to allow gradient flow
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
        # Handle 5D input (video)
        is_video = x.ndim == 5
        if is_video:
            B, T, C, H, W = x.shape
            x = rearrange(x, "b t c h w -> (b t) c h w")
            
            # Handle condition shape - must align with (B*T) if video
            if cond.ndim == 5:
                # Video condition: flatten time to match x shape (B*T, C, H, W)
                cond = rearrange(cond, "b t c h w -> (b t) c h w")
            elif cond.ndim == 4:
                # Static condition: repeat for each frame
                cond = cond.unsqueeze(1).repeat(1, T, 1, 1, 1)  # (B, Cond_C, H, W) -> (B, T, Cond_C, H, W)
                cond = rearrange(cond, "b t c h w -> (b t) c h w")  # -> (B*T, Cond_C, H, W)
        else:
            B, C, H, W = x.shape
            # Handle 5D condition for 4D input (unlikely but handle it)
            if cond.ndim == 5:
                # Average over time or take first frame
                cond = cond.mean(dim=1)  # (B, T, C, H, W) -> (B, C, H, W)
        
        # Upsample condition to match feature resolution if needed
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize features (RMSNorm expects channels last)
        x_norm = rearrange(x, "b c h w -> b h w c")
        x_norm = self.norm(x_norm)
        x_norm = rearrange(x_norm, "b h w c -> b c h w")
        
        # Generate spatial modulation parameters
        spade_params = self.spade_net(cond)  # (B*T, 2*C, H, W)
        gamma, beta = spade_params.chunk(2, dim=1)  # Each (B*T, C, H, W)
        
        # Apply spatial modulation: y = γ(x) * x_norm + β(x)
        out = gamma * x_norm + beta
        
        # Restore video shape if needed
        if is_video:
            out = rearrange(out, "(b t) c h w -> b t c h w", b=B, t=T)
        
        return out


class SPADEAdaLNModulation(nn.Module):
    """
    Combined SPADE + Time-based AdaLN modulation.
    
    - Spatial modulation from condition (SPADE)
    - Global modulation from time embedding (AdaLN)
    
    Args:
        time_dim: Time embedding dimension
        channels: Feature channels
        cond_channels: Condition channels for SPADE
    """
    
    def __init__(self, time_dim: int, channels: int, cond_channels: int):
        super().__init__()
        # SPADE for spatial condition
        self.spade = SPADENorm(channels, cond_channels, hidden_channels=128)
        
        # Time-based global modulation (AdaLN style)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2 * channels),
        )
        # Initialize to small values instead of zero (identity at start but allows gradient flow)
        nn.init.normal_(self.time_mlp[-1].weight, mean=0.0, std=0.01)
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
        # Apply SPADE (spatial modulation)
        x = self.spade(x, cond)
        
        # Apply time-based global modulation
        time_params = self.time_mlp(t_emb)  # (B, 2*C)
        gamma_t, beta_t = time_params.chunk(2, dim=1)  # Each (B, C)
        
        # Reshape for broadcasting over (T, H, W)
        gamma_t = gamma_t.view(-1, 1, gamma_t.size(1), 1, 1)  # (B, 1, C, 1, 1)
        beta_t = beta_t.view(-1, 1, beta_t.size(1), 1, 1)
        
        x = x * (1 + gamma_t) + beta_t
        
        return x


# ==============================================================================
# SwiGLU (For MLP)
# ==============================================================================

class SwiGLU(nn.Module):
    """Gated Linear Unit with Swish activation."""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.dropout(self.w3(F.silu(x1) * x2))


# ==============================================================================
# SPADE-Mamba Block
# ==============================================================================

class SPADEMambaBlock(nn.Module):
    """
    Hybrid Spatio-Temporal Mamba Block with SPADE/VideoAdaLN conditioning.
    
    Supports two conditioning modes:
    - SPADE: If cond_channels > 0, uses spatial conditioning
    - VideoAdaLN: If cond_channels = 0, uses only time-based modulation
    
    Architecture:
    1. Modulated ST-Mamba (STDSConv + STSS)
    2. Modulated MLP
    
    Args:
        dim: Model dimension
        cond_channels: Condition channels (0 = VideoAdaLN only, >0 = SPADE)
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
        
        # Channel Mixing (MLP)
        self.mlp = SwiGLU(dim, dim * 4, dropout=dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor],
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Video latent (B, T, C, H, W)
            cond: Spatial condition (B, Cond_C, H_cond, W_cond) or None
            t_emb: Time embedding (B, time_dim)
            
        Returns:
            Output features (B, T, C, H, W)
        """
        # Branch 1: Spatio-Temporal Mamba
        residual = x
        if self.use_spade:
            x_mod = self.modulation1(x, cond, t_emb)
        else:
            x_mod = self.modulation1(x, t_emb)
        x_out = self.st_conv(x_mod)
        x_out = self.st_ss(x_out)
        x = residual + x_out
        
        # Branch 2: MLP
        residual = x
        if self.use_spade:
            x_mod = self.modulation2(x, cond, t_emb)
        else:
            x_mod = self.modulation2(x, t_emb)
        # MLP operates on channel dimension
        x_mod = rearrange(x_mod, "b t c h w -> b t h w c")
        x_out = self.mlp(x_mod)
        x = residual + rearrange(x_out, "b t h w c -> b t c h w")
        
        return x


# ==============================================================================
# SPADE Velocity Network
# ==============================================================================

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time embedding."""
    half_dim = dim // 2
    freq_const = math.log(10000.0) / max(half_dim - 1, 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * (-freq_const))
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class VideoSPADEFinalLayer(nn.Module):
    """Final SPADE/VideoAdaLN-modulated output layer (hybrid)."""
    
    def __init__(self, hidden_dim: int, cond_channels: int, time_dim: int, out_channels: int):
        super().__init__()
        self.use_spade = cond_channels > 0
        
        if self.use_spade:
            self.modulation = SPADEAdaLNModulation(time_dim, hidden_dim, cond_channels)
        else:
            self.modulation = VideoAdaLN(time_dim, hidden_dim)
        
        self.conv = nn.Conv3d(hidden_dim, out_channels, kernel_size=3, padding=1)
        # Initialize to small values for stability (not zero to avoid dead gradients)
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor], t_emb: torch.Tensor) -> torch.Tensor:
        if self.use_spade:
            x = self.modulation(x, cond, t_emb)
        else:
            x = self.modulation(x, t_emb)
        x = rearrange(x, "b t c h w -> b c t h w")
        x = self.conv(x)
        x = rearrange(x, "b c t h w -> b t c h w")
        return x


class SPADEMambaVelocityNet(nn.Module):
    """
    Hybrid Velocity Network with SPADE/VideoAdaLN conditioning for CFM.
    
    Input Strategy:
    - Concatenates x_past (clean, T_in) + x_noisy (future, T_out) on time dim
    - Total input time: T = T_in + T_out
    - Loss computed only on future part (T_out)
    
    Conditioning:
    - If cond provided → SPADE modulation (spatial conditioning)
    - If cond = None → VideoAdaLN modulation (time-only)
    
    Args:
        hidden_dim: Model hidden dimension
        cond_channels: Number of condition channels (0 = no spatial cond, >0 = SPADE)
        num_blocks: Number of hybrid Mamba blocks
        d_state: Mamba state dimension
        d_conv: Mamba convolution size
        expand: Mamba expansion factor
        dropout: Dropout rate
        latent_channels: Number of latent channels (input/output)
        latent_size: Latent spatial size (T_out, H, W) - future only
        t_in: Number of past frames
        t_out: Number of future frames to predict
        output_head: Type of output head ("conv3d" or "mlp")
        gradient_checkpointing: Enable gradient checkpointing
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        cond_channels: int = 27,  # 0 for VideoAdaLN, >0 for SPADE
        num_blocks: int = 8,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        latent_channels: int = 4,
        latent_size: tuple[int, int, int] = (24, 8, 19),  # (T_out, H, W)
        t_in: int = 8,  # Number of past frames
        t_out: int = 24,  # Number of future frames
        output_head: Literal["conv3d", "mlp"] = "conv3d",
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_channels = cond_channels
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.t_in = t_in
        self.t_out = t_out
        self.output_head_type = output_head
        self.gradient_checkpointing = gradient_checkpointing
        
        T_out, H, W = latent_size
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # Input projection (3D conv for local context)
        self.input_proj = nn.Conv3d(latent_channels, hidden_dim, kernel_size=3, padding=1)
        
        # Absolute positional embeddings (for full T_in + T_out)
        self.pos_emb = nn.Parameter(torch.zeros(1, hidden_dim, t_in + t_out, H, W))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        
        # SPADE-Mamba blocks
        self.blocks = nn.ModuleList([
            SPADEMambaBlock(
                dim=hidden_dim,
                cond_channels=cond_channels,
                time_dim=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])
        
        # Output head
        if output_head == "conv3d":
            self.output_layer = VideoSPADEFinalLayer(
                hidden_dim, cond_channels, hidden_dim, latent_channels
            )
        elif output_head == "mlp":
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, latent_channels),
            )
        else:
            raise ValueError(f"Unknown output_head: {output_head}")
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        x_past: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with past+future input strategy.
        
        Args:
            x_t: Noisy future latent at time t (B, T_out, C, H, W)
            t: Time step in [0, 1] (B,)
            cond: Spatial condition (B, Cond_C, H_cond, W_cond) or None
            x_past: Clean past latent (B, T_in, C, H, W) or None
            
        Returns:
            Predicted velocity for future part only (B, T_out, C, H, W)
        """
        B, T_out, C, H, W = x_t.shape
        
        # Time embedding
        t_emb = sinusoidal_embedding(t, self.hidden_dim)
        t_emb = self.time_mlp(t_emb)  # (B, hidden_dim)
        
        # Concatenate past + noisy future on time dimension
        if x_past is not None:
            x_input = torch.cat([x_past, x_t], dim=1)  # (B, T_in+T_out, C, H, W)
        else:
            x_input = x_t  # Fallback: only future
        
        # Input projection
        x = rearrange(x_input, "b t c h w -> b c t h w")
        x = self.input_proj(x)
        
        # Add positional embeddings (adjust if only future is passed)
        if x_past is not None:
            x = x + self.pos_emb
        else:
            # Use only future part of pos_emb
            x = x + self.pos_emb[:, :, self.t_in:, :, :]
        
        # Rearrange to (B, T, C, H, W) for blocks
        x = rearrange(x, "b c t h w -> b t c h w")
        
        # Apply hybrid SPADE/VideoAdaLN Mamba blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cond, t_emb, use_reentrant=False)
            else:
                x = block(x, cond, t_emb)
        
        # Extract future part only for output
        if x_past is not None:
            x = x[:, self.t_in:, :, :, :]  # Keep only T_out frames
        
        # Output head
        if self.output_head_type == "conv3d":
            out = self.output_layer(x, cond, t_emb)
        else:  # mlp
            out = rearrange(x, "b t c h w -> b t h w c")
            out = self.output_layer(out)
            out = rearrange(out, "b t h w c -> b t c h w")
        
        return out
