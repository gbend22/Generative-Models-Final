import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConditionalInstanceNorm2d(nn.Module):
    """
    Conditional Instance Normalization
    Modulates normalization based on noise level
    """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
        
        # Learnable scale and bias for each noise level
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Scale ~ 1
        self.embed.weight.data[:, num_features:].zero_()  # Bias ~ 0
    
    def forward(self, x, y):
        """
        Args:
            x: Input features [B, C, H, W]
            y: Noise level index [B]
        """
        out = self.instance_norm(x)
        gamma, beta = self.embed(y).chunk(2, dim=1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return gamma * out + beta


class ResidualBlock(nn.Module):
    """
    Residual block with conditional normalization
    """
    def __init__(self, in_channels, out_channels, num_classes, stride=1, downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.norm1 = ConditionalInstanceNorm2d(out_channels, num_classes)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = ConditionalInstanceNorm2d(out_channels, num_classes)
        
        self.downsample = downsample
        self.act = nn.ELU()
    
    def forward(self, x, y):
        """
        Args:
            x: Input [B, C, H, W]
            y: Noise level [B]
        """
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out, y)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.norm2(out, y)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.act(out)
        
        return out


class RefineBlock(nn.Module):
    """
    RefineNet block - combines features from encoder path
    """
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        
        # Residual convolution units
        self.rcu1 = ResidualBlock(in_channels, out_channels, num_classes)
        self.rcu2 = ResidualBlock(out_channels, out_channels, num_classes)
        
        # Multi-resolution fusion (if needed)
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)
        
        # Chain pooling
        self.pool = nn.AvgPool2d(5, 1, 2)
        self.conv_pool = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    
    def forward(self, x, y):
        """
        Args:
            x: Input features
            y: Noise level
        """
        # Apply residual blocks
        out = self.rcu1(x, y)
        
        # Chained pooling for capturing context
        pool1 = self.pool(out)
        pool1 = self.conv_pool(pool1)
        
        out = out + pool1
        out = self.rcu2(out, y)
        
        return out


class RefineNet(nn.Module):
    """
    RefineNet architecture for NCSN
    Multi-scale feature processing with skip connections
    """
    def __init__(self, in_channels=3, base_channels=128, num_classes=10):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        
        # Encoder path (downsampling)
        self.enc1 = ResidualBlock(base_channels, base_channels * 2, num_classes, stride=2)
        self.enc2 = ResidualBlock(base_channels * 2, base_channels * 4, num_classes, stride=2)
        self.enc3 = ResidualBlock(base_channels * 4, base_channels * 8, num_classes, stride=2)
        
        # RefineNet blocks (decoder path)
        self.refine3 = RefineBlock(base_channels * 8, base_channels * 4, num_classes)
        self.refine2 = RefineBlock(base_channels * 4, base_channels * 2, num_classes)
        self.refine1 = RefineBlock(base_channels * 2, base_channels, num_classes)
        
        # Output convolution
        self.conv_out = nn.Conv2d(base_channels, in_channels, 3, 1, 1)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, y):
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, 32, 32]
            y: Noise level indices [B]
        
        Returns:
            Score estimate [B, 3, 32, 32]
        """
        # Initial features
        h = self.conv_in(x)
        
        # Encoder path
        h1 = self.enc1(h, y)      # 16x16
        h2 = self.enc2(h1, y)     # 8x8
        h3 = self.enc3(h2, y)     # 4x4
        
        # Decoder path with skip connections
        h = self.refine3(h3, y)
        h = self.upsample(h) + h2
        
        h = self.refine2(h, y)
        h = self.upsample(h) + h1
        
        h = self.refine1(h, y)
        h = self.upsample(h)
        
        # Output score
        score = self.conv_out(h)
        
        return score
