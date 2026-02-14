"""
Context CNN Encoder for SPADE conditioning.

Simple and efficient encoder for preparing dense spatial conditions (weather, physics).
Replaces ViT encoder which is inefficient for pixel-aligned tasks.

Philosophy:
- Preserves spatial structure (no tokenization)
- Mixes temporal and channel dimensions
- Reduces to compact representation for SPADE
- Fast and memory-efficient
"""

import torch
import torch.nn as nn
from einops import rearrange

class ResidualBlock(nn.Module):
    """
    Bloc résiduel simple avec GroupNorm.
    Garde la résolution spatiale (stride=1).
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
    Encodeur contextuel "SimVP-style" pour SPADE.
    Transforme (B, T_in, C_in, H, W) -> (B, out_channels, H, W).
    """
    def __init__(
        self, 
        in_channels: int,        # Tes variables physiques (ex: 27)
        t_in: int,               # Nombre de frames de contexte (ex: 8)
        out_channels: int = 64,  # Ce que SPADE va recevoir (compact)
        hidden_dim: int = 128,   # Largeur interne de l'encodeur (cond_encoder_hidden_dim)
        num_layers: int = 3,     # Profondeur (nombre de ResBlocks)
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Flattening Time-to-Channel: 27 vars * 8 frames = 216 canaux d'entrée
        input_dim = in_channels * t_in
        
        # 1. Projection initiale (Mélange Temps/Variables)
        self.initial_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        # 2. Corps du réseau (Extraction de features profondes sans perte de résolution)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        # 3. Projection finale vers la dimension du SPADE
        self.final_norm = nn.GroupNorm(8, hidden_dim)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        
        # Initialisation soignée
        self.apply(self._init_weights)
        # Zero-out last layer of ResBlock for better gradient flow at start
        for block in self.blocks:
            nn.init.zeros_(block.conv2.weight)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, T, C, H, W)
        if x.ndim == 5:
            # Flatten Time into Channels: (B, T, C, H, W) -> (B, T*C, H, W)
            x = rearrange(x, 'b t c h w -> b (t c) h w')
            
        x = self.initial_conv(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_act(self.final_norm(x))
        x = self.final_conv(x)
        
        return x # (B, out_channels, H, W)