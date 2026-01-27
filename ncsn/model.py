"""
NCSN Model Architecture.

Implements RefineNet-style architecture for score estimation.

The model takes as input:
- x: Noisy image of shape (B, C, H, W)
- y: Noise level index of shape (B,)

And outputs:
- s: Estimated score ∇_x log p(x|σ) of shape (B, C, H, W)

References:
- Song & Ermon (2019): Original NCSN architecture
- Lin et al. (2017): RefineNet for semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .layers import (
    ConditionalInstanceNorm2d,
    ConditionalResidualBlock,
    RefineBlock,
)


class RefineNet(nn.Module):
    """
    RefineNet-style architecture for score estimation.

    Architecture overview:
    1. Initial convolution to expand channels
    2. Encoder: Progressively downsample while increasing channels
    3. Middle blocks with dilated convolutions
    4. Decoder: RefineNet blocks that fuse multi-scale features
    5. Output projection to score

    All normalization layers are conditioned on the noise level σ.

    Args:
        in_channels: Number of input image channels (3 for RGB)
        num_classes: Number of noise levels L (for conditioning)
        ngf: Base number of filters (multiplied at each level)
    """

    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 10,
            ngf: int = 128,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.ngf = ngf

        # Channel progression: ngf -> 2*ngf -> 4*ngf -> 4*ngf
        channels = [ngf, 2 * ngf, 4 * ngf, 4 * ngf]

        # ============ Encoder ============
        # Initial convolution: (B, 3, 32, 32) -> (B, ngf, 32, 32)
        self.conv_in = nn.Conv2d(in_channels, ngf, kernel_size=3, padding=1)

        # Encoder blocks (progressively downsample)
        # Level 0: 32x32, ngf channels
        self.enc0 = nn.ModuleList([
            ConditionalResidualBlock(ngf, ngf, num_classes),
            ConditionalResidualBlock(ngf, ngf, num_classes),
        ])

        # Level 1: 16x16, 2*ngf channels
        self.enc1 = nn.ModuleList([
            ConditionalResidualBlock(ngf, 2 * ngf, num_classes, resample='down'),
            ConditionalResidualBlock(2 * ngf, 2 * ngf, num_classes),
        ])

        # Level 2: 8x8, 4*ngf channels
        self.enc2 = nn.ModuleList([
            ConditionalResidualBlock(2 * ngf, 4 * ngf, num_classes, resample='down'),
            ConditionalResidualBlock(4 * ngf, 4 * ngf, num_classes),
        ])

        # Level 3: 4x4, 4*ngf channels (with dilation)
        self.enc3 = nn.ModuleList([
            ConditionalResidualBlock(4 * ngf, 4 * ngf, num_classes, resample='down'),
            ConditionalResidualBlock(4 * ngf, 4 * ngf, num_classes, dilation=2),
        ])

        # ============ Middle (Bottleneck) ============
        # Additional processing at lowest resolution with dilated convolutions
        self.middle = nn.ModuleList([
            ConditionalResidualBlock(4 * ngf, 4 * ngf, num_classes, dilation=2),
            ConditionalResidualBlock(4 * ngf, 4 * ngf, num_classes, dilation=4),
        ])

        # ============ Decoder (RefineNet blocks) ============
        # Progressively fuse features from different scales

        # Refine level 3: fuse middle output
        self.refine3 = RefineBlock([4 * ngf], 4 * ngf, num_classes)

        # Refine level 2: fuse with enc2
        self.refine2 = RefineBlock([4 * ngf, 4 * ngf], 2 * ngf, num_classes)

        # Refine level 1: fuse with enc1
        self.refine1 = RefineBlock([2 * ngf, 2 * ngf], ngf, num_classes)

        # Refine level 0: fuse with enc0
        self.refine0 = RefineBlock([ngf, ngf], ngf, num_classes)

        # ============ Output ============
        # Final normalization and projection to score
        self.norm_out = ConditionalInstanceNorm2d(ngf, num_classes)
        self.conv_out = nn.Conv2d(ngf, in_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Input conv
        nn.init.kaiming_normal_(self.conv_in.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv_in.bias)

        # Output conv (initialize to small values for score)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input noisy images of shape (B, C, H, W)
            y: Noise level indices of shape (B,) with values in [0, L-1]

        Returns:
            Estimated score of shape (B, C, H, W)
        """
        # Initial convolution
        h = self.conv_in(x)  # (B, ngf, 32, 32)

        # ============ Encoder ============
        # Level 0: 32x32
        for block in self.enc0:
            h = block(h, y)
        h0 = h  # Save for skip connection

        # Level 1: 16x16
        for block in self.enc1:
            h = block(h, y)
        h1 = h  # Save for skip connection

        # Level 2: 8x8
        for block in self.enc2:
            h = block(h, y)
        h2 = h  # Save for skip connection

        # Level 3: 4x4
        for block in self.enc3:
            h = block(h, y)
        h3 = h  # Save for skip connection

        # ============ Middle ============
        for block in self.middle:
            h = block(h, y)

        # ============ Decoder (RefineNet) ============
        # Refine level 3
        h = self.refine3([h], y)  # (B, 4*ngf, 4, 4)

        # Upsample and refine level 2
        h = F.interpolate(h, size=h2.shape[-2:], mode='bilinear', align_corners=False)
        h = self.refine2([h, h2], y)  # (B, 2*ngf, 8, 8)

        # Upsample and refine level 1
        h = F.interpolate(h, size=h1.shape[-2:], mode='bilinear', align_corners=False)
        h = self.refine1([h, h1], y)  # (B, ngf, 16, 16)

        # Upsample and refine level 0
        h = F.interpolate(h, size=h0.shape[-2:], mode='bilinear', align_corners=False)
        h = self.refine0([h, h0], y)  # (B, ngf, 32, 32)

        # ============ Output ============
        h = self.norm_out(h, y)
        h = F.relu(h)
        score = self.conv_out(h)  # (B, C, 32, 32)

        return score


class NCSN(nn.Module):
    """
    Noise Conditional Score Network wrapper.

    Wraps the RefineNet architecture and handles noise level computation.

    Args:
        config: Configuration dict with model parameters
        sigmas: Pre-computed noise levels tensor
    """

    def __init__(
            self,
            config: dict,
            sigmas: torch.Tensor,
    ):
        super().__init__()

        self.config = config
        self.register_buffer('sigmas', sigmas)

        # Build the score network
        self.score_net = RefineNet(
            in_channels=config['data']['channels'],
            num_classes=config['model']['num_classes'],
            ngf=config['model']['ngf'],
        )

    @property
    def num_classes(self) -> int:
        return self.config['model']['num_classes']

    def forward(
            self,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            sigmas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)
            y: Noise level indices (B,). If None, will be sampled randomly.
            sigmas: Actual sigma values (B,). Alternative to y for continuous conditioning.

        Returns:
            Estimated score (B, C, H, W)
        """
        if y is None and sigmas is None:
            # Sample random noise levels during training
            y = torch.randint(0, self.num_classes, (x.shape[0],), device=x.device)

        return self.score_net(x, y)

    def get_sigmas(self, y: torch.Tensor) -> torch.Tensor:
        """Get sigma values for given indices."""
        return self.sigmas[y]


def get_model(config: dict, sigmas: torch.Tensor) -> NCSN:
    """
    Factory function to create NCSN model.

    Args:
        config: Configuration dictionary
        sigmas: Pre-computed noise levels

    Returns:
        NCSN model instance
    """
    return NCSN(config, sigmas)


# For direct import
__all__ = ['RefineNet', 'NCSN', 'get_model']