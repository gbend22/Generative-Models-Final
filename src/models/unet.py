"""
U-Net architecture for NCSN
Based on RefineNet architecture from Song & Ermon (2019)
Implements a conditional U-Net that takes noise level as input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConditionalInstanceNorm2d(nn.Module):
    """
    Conditional Instance Normalization
    Modulates the normalization parameters based on noise level
    """

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)

        # Learnable scale and bias for each noise level
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # gamma
        self.embed.weight.data[:, num_features:].zero_()  # beta

    def forward(self, x, y):
        """
        x: (B, C, H, W)
        y: (B,) noise level indices
        """
        h = self.instance_norm(x)
        gamma, beta = self.embed(y).chunk(2, dim=1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return gamma * h + beta


class ConditionalResidualBlock(nn.Module):
    """
    Residual block with conditional instance normalization
    """

    def __init__(self, in_channels, out_channels, num_classes, resample=None, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resample = resample

        # First conv block
        self.norm1 = ConditionalInstanceNorm2d(in_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Second conv block
        self.norm2 = ConditionalInstanceNorm2d(out_channels, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection
        if in_channels != out_channels or resample:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, y):
        """
        x: (B, C, H, W)
        y: (B,) noise level indices
        """
        h = self.norm1(x, y)
        h = F.relu(h)

        # Resample if needed
        if self.resample == 'down':
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)
        elif self.resample == 'up':
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            x = F.interpolate(x, scale_factor=2, mode='nearest')

        h = self.conv1(h)
        h = self.norm2(h, y)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class RefineBlock(nn.Module):
    """
    RefineNet block for multi-scale feature fusion
    """

    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.conv1 = ConditionalResidualBlock(in_channels, out_channels, num_classes)
        self.conv2 = ConditionalResidualBlock(out_channels, out_channels, num_classes)

    def forward(self, x, y):
        h = self.conv1(x, y)
        h = self.conv2(h, y)
        return h


class RefineNetUNet(nn.Module):
    """
    RefineNet-based U-Net for NCSN
    Architecture follows Song & Ermon (2019)
    """

    def __init__(self, image_channels=3, ngf=128, num_classes=10):
        super().__init__()
        self.ngf = ngf
        self.num_classes = num_classes

        # Initial convolution
        self.conv_first = nn.Conv2d(image_channels, ngf, 3, padding=1)

        # Encoder (downsampling path)
        self.down1 = ConditionalResidualBlock(ngf, ngf, num_classes)
        self.down2 = ConditionalResidualBlock(ngf, ngf * 2, num_classes, resample='down')
        self.down3 = ConditionalResidualBlock(ngf * 2, ngf * 2, num_classes, resample='down')

        # Middle (bottleneck)
        self.middle = ConditionalResidualBlock(ngf * 2, ngf * 2, num_classes)

        # Decoder (upsampling path)
        self.up1 = ConditionalResidualBlock(ngf * 2, ngf * 2, num_classes, resample='up')
        self.up2 = ConditionalResidualBlock(ngf * 2, ngf, num_classes, resample='up')
        self.up3 = ConditionalResidualBlock(ngf, ngf, num_classes)

        # RefineNet blocks for feature fusion
        self.refine1 = RefineBlock(ngf * 4, ngf * 2, num_classes)
        self.refine2 = RefineBlock(ngf * 2, ngf, num_classes)

        # Final normalization and output
        self.norm_final = ConditionalInstanceNorm2d(ngf, num_classes)
        self.conv_final = nn.Conv2d(ngf, image_channels, 3, padding=1)

    def forward(self, x, y):
        """
        x: (B, C, H, W) input images
        y: (B,) noise level indices
        Returns: (B, C, H, W) score estimates
        """
        # Initial features
        h = self.conv_first(x)

        # Encoder
        h1 = self.down1(h, y)  # (B, ngf, 32, 32)
        h2 = self.down2(h1, y)  # (B, ngf*2, 16, 16)
        h3 = self.down3(h2, y)  # (B, ngf*2, 8, 8)

        # Bottleneck
        h = self.middle(h3, y)  # (B, ngf*2, 8, 8)

        # Decoder with skip connections
        h = self.up1(h, y)  # (B, ngf*2, 16, 16)
        h = torch.cat([h, h2], dim=1)  # (B, ngf*4, 16, 16)
        h = self.refine1(h, y)  # (B, ngf*2, 16, 16)

        h = self.up2(h, y)  # (B, ngf, 32, 32)
        h = torch.cat([h, h1], dim=1)  # (B, ngf*2, 32, 32)
        h = self.refine2(h, y)  # (B, ngf, 32, 32)

        h = self.up3(h, y)  # (B, ngf, 32, 32)

        # Final output
        h = self.norm_final(h, y)
        h = F.relu(h)
        h = self.conv_final(h)

        return h


if __name__ == '__main__':
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RefineNetUNet(image_channels=3, ngf=128, num_classes=10).to(device)

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32).to(device)
    y = torch.randint(0, 10, (batch_size,)).to(device)

    output = model(x, y)
    print(f"Input shape: {x.shape}")
    print(f"Noise levels: {y}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")