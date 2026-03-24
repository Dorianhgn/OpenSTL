# Copyright (c) CAIRI AI Lab. All rights reserved
# Context Encoders for conditioning in Flow Matching models

"""
Context Encoders for preparing spatial conditions.

These encoders transform raw input conditions into compact spatial representations
suitable for conditioning models like SPADEJvM.

Available encoders:
- ContextNet: Simple CNN encoder with residual blocks
- LabelConditioner: Lightweight spatial decoder for class / multi-hot labels
- More to come...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResidualBlock(nn.Module):
    """
    Simple residual block with GroupNorm.
    Maintains spatial resolution (stride=1).
    """
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h  # Skip connection


class ContextNet(nn.Module):
    """
    CNN-based context encoder for SPADE conditioning.
    Transforms (B, T_in, C_in, H, W) -> (B, out_channels, H, W).
    
    Philosophy:
    - Preserves spatial structure (no tokenization)
    - Mixes temporal and channel dimensions
    - Reduces to compact representation for conditioning
    - Fast and memory-efficient
    
    Args:
        in_channels: Number of input channels per frame
        t_in: Number of input frames
        out_channels: Output channels for conditioning (cond_channels)
        hidden_dim: Internal network width
        num_layers: Number of residual blocks
        dropout: Dropout rate
    """
    def __init__(
        self,
        in_channels: int,
        t_in: int,
        out_channels: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Flattening Time-to-Channel: C_in * T_in channels
        input_dim = in_channels * t_in
        
        # 1. Initial projection (Mix Time/Channels)
        self.initial_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        # 2. Network body (Deep feature extraction without resolution loss)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 3. Final projection to conditioning dimension
        self.final_norm = nn.GroupNorm(8, hidden_dim)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        
        # Initialization
        self.apply(self._init_weights)
        # Zero-out last layer of ResBlock for better gradient flow at start
        for block in self.blocks:
            nn.init.zeros_(block.conv2.weight)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) or (B, T*C, H, W)
            
        Returns:
            (B, out_channels, H, W)
        """
        # Flatten Time into Channels if needed
        if x.ndim == 5:
            x = rearrange(x, 'b t c h w -> b (t c) h w')
        
        x = self.initial_conv(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_act(self.final_norm(x))
        x = self.final_conv(x)
        
        return x  # (B, out_channels, H, W)


class LabelConditioner(nn.Module):
    """
    Lightweight conditioner for categorical / multi-hot labels.

    Transforms a label vector (B, num_classes) or class indices (B,) into a
    compact spatial tensor compatible with SPADE conditioning.

    Args:
        num_classes: Size of the label vocabulary.
        out_channels: Output channels for conditioning.
        hidden_dim: Internal projection width.
        dropout: Dropout rate on the label embedding.
    """

    def __init__(
        self,
        num_classes: int = 10,
        out_channels: int = 64,
        hidden_dim: int = 64,
        spatial_size: int = 16,
        base_size: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        self.base_size = base_size

        self.label_proj = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim * base_size * base_size),
            nn.SiLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
        )

        nn.init.normal_(self.decoder[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, labels):
        """
        Args:
            labels: (B,) integer class indices or (B, num_classes) multi-hot vectors.

        Returns:
            (B, out_channels, 1, 1) spatial conditioning tensor.
        """
        if labels.ndim == 1:
            labels = F.one_hot(labels.long(), num_classes=self.num_classes)
        labels = labels.float()
        cond = self.label_proj(labels)
        cond = cond.view(labels.shape[0], -1, self.base_size, self.base_size)
        cond = self.decoder(cond)

        if cond.shape[-1] != self.spatial_size or cond.shape[-2] != self.spatial_size:
            cond = F.interpolate(cond, size=(self.spatial_size, self.spatial_size), mode='bilinear', align_corners=False)

        return cond


# Registry for context encoders
CONTEXT_ENCODER_REGISTRY = {
    'contextnet': ContextNet,
    'ContextNet': ContextNet,
    'labelconditioner': LabelConditioner,
    'LabelConditioner': LabelConditioner,
    'SpatialLabelConditioner': LabelConditioner,
}


def build_context_encoder(encoder_type: str, **kwargs):
    """
    Build a context encoder from registry.
    
    Args:
        encoder_type: Type of encoder ('contextnet', etc.)
        **kwargs: Arguments for the encoder
        
    Returns:
        Context encoder module
    """
    if encoder_type not in CONTEXT_ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown context encoder type: {encoder_type}. "
            f"Available: {list(CONTEXT_ENCODER_REGISTRY.keys())}"
        )
    
    return CONTEXT_ENCODER_REGISTRY[encoder_type](**kwargs)
