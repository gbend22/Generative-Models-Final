"""
Custom layers for NCSN.

Implements:
- Conditional Instance Normalization (conditions on noise level σ)
- Residual blocks with dilated convolutions
- RefineNet blocks for multi-scale processing

References:
- Song & Ermon (2019): NCSN architecture details
- Conditional Instance Normalization from style transfer literature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConditionalInstanceNorm2d(nn.Module):
    """
    Conditional Instance Normalization.

    Applies instance normalization but with learnable scale (γ) and shift (β)
    parameters that are conditioned on the noise level index.

    This allows the network to adapt its normalization statistics based on
    which noise level σ_i the input was corrupted with.

    Math:
        y = γ(σ) * (x - μ) / σ_x + β(σ)

    where μ and σ_x are instance-wise mean and std, and γ(σ), β(σ) are
    learned embeddings indexed by the noise level.

    Args:
        num_features: Number of channels in the input
        num_classes: Number of noise levels (L)
    """

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Instance norm without learnable affine parameters
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)

        # Learnable scale and shift for each noise level
        # Shape: (num_classes, num_features)
        self.gamma = nn.Embedding(num_classes, num_features)
        self.beta = nn.Embedding(num_classes, num_features)

        # Initialize: γ=1, β=0 (identity transformation)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)
            y: Noise level indices of shape (B,) with values in [0, num_classes-1]

        Returns:
            Normalized and conditioned tensor of shape (B, C, H, W)
        """
        # Apply instance normalization
        out = self.instance_norm(x)

        # Get conditioning parameters
        gamma = self.gamma(y)  # (B, C)
        beta = self.beta(y)  # (B, C)

        # Reshape for broadcasting: (B, C) -> (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # Apply conditional affine transformation
        out = gamma * out + beta

        return out


class ConditionalResidualBlock(nn.Module):
    """
    Residual block with Conditional Instance Normalization.

    Structure:
        x -> Conv -> CondIN -> ReLU -> Conv -> CondIN -> + -> out
        |______________________________________________|
                            (skip)

    Optionally uses dilation for increased receptive field.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_classes: Number of noise levels for conditioning
        resample: 'up', 'down', or None for resolution change
        dilation: Dilation rate for convolutions
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_classes: int,
            resample: str = None,
            dilation: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resample = resample

        # First conv-norm-activation
        self.norm1 = ConditionalInstanceNorm2d(in_channels, num_classes)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=dilation, dilation=dilation
        )

        # Second conv-norm-activation
        self.norm2 = ConditionalInstanceNorm2d(out_channels, num_classes)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=dilation, dilation=dilation
        )

        # Skip connection
        if in_channels != out_channels or resample is not None:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Resampling layers
        if resample == 'up':
            self.resample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        elif resample == 'down':
            self.resample_layer = nn.AvgPool2d(2)
        else:
            self.resample_layer = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He initialization."""
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        if isinstance(self.shortcut, nn.Conv2d):
            nn.init.kaiming_normal_(self.shortcut.weight, mode='fan_out', nonlinearity='relu')
            if self.shortcut.bias is not None:
                nn.init.zeros_(self.shortcut.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)
            y: Noise level indices (B,)

        Returns:
            Output tensor (B, C', H', W')
        """
        # Main path
        h = self.norm1(x, y)
        h = F.relu(h)
        h = self.resample_layer(h)
        h = self.conv1(h)

        h = self.norm2(h, y)
        h = F.relu(h)
        h = self.conv2(h)

        # Skip connection
        skip = self.resample_layer(x)
        skip = self.shortcut(skip)

        return h + skip


class RefineBlock(nn.Module):
    """
    RefineNet block for multi-scale feature fusion.

    Takes features from multiple resolutions, processes them,
    and fuses them together.

    Args:
        in_channels: List of input channels for each resolution
        out_channels: Number of output channels
        num_classes: Number of noise levels
    """

    def __init__(
            self,
            in_channels: list,
            out_channels: int,
            num_classes: int,
    ):
        super().__init__()

        self.adapters = nn.ModuleList()
        for ch in in_channels:
            self.adapters.append(
                ConditionalResidualBlock(ch, out_channels, num_classes)
            )

        self.output_block = ConditionalResidualBlock(
            out_channels, out_channels, num_classes
        )

    def forward(self, xs: list, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            xs: List of feature maps at different resolutions
            y: Noise level indices (B,)

        Returns:
            Fused feature map
        """
        # Process each input and upsample to largest resolution
        hs = []
        target_size = xs[0].shape[-2:]  # Assume first is largest

        for x, adapter in zip(xs, self.adapters):
            h = adapter(x, y)
            if h.shape[-2:] != target_size:
                h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)
            hs.append(h)

        # Sum all features
        h = sum(hs)

        # Final processing
        h = self.output_block(h, y)

        return h


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for noise level conditioning.

    Alternative to embedding lookup - provides continuous representation
    of the noise level σ.

    Args:
        dim: Embedding dimension
        max_positions: Maximum number of positions (not used, kept for compatibility)
    """

    def __init__(self, dim: int, max_positions: int = 10000):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        """
        Generate positional encoding for noise levels.

        Args:
            noise_level: Tensor of noise levels (B,) or scalar values

        Returns:
            Positional encoding of shape (B, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=noise_level.device) * -emb)

        # Handle both index-based and continuous noise levels
        if noise_level.dim() == 0:
            noise_level = noise_level.unsqueeze(0)

        emb = noise_level.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode='constant')

        return emb