import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianFourierProjection(nn.Module):
    """
    Embeds the scalar noise level sigma into a vector.
    Inspired by 'Language Models are Unsupervised Multitask Learners' 
    and widely used in Score-based models / Diffusion.
    """
    def __init__(self, embed_dim=256, scale=30.0):
        super().__init__()
        # Randomly sampled weights for the projection, fixed during training
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        # x shape: [batch_size] (the sigmas)
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.dense(x)[..., None, None]

class ScoreNet(nn.Module):
    """
    A simple U-Net based Score Network for CIFAR-10 (32x32).
    Input: x (batch, 3, 32, 32), sigma (batch)
    Output: score (batch, 3, 32, 32)
    """
    def __init__(self, num_classes=10, channels=3, ch=128, ch_mult=[1, 2, 2, 2]):
        super().__init__()
        
        # 1. Noise Embedding
        self.embed = GaussianFourierProjection(embed_dim=ch)
        
        # 2. Downsampling (Encoder)
        self.conv_in = nn.Conv2d(channels, ch, kernel_size=3, stride=1, padding=1)
        
        self.down1 = nn.ModuleList([
            nn.Conv2d(ch, ch * ch_mult[0], 3, 2, 1),
            Dense(ch, ch * ch_mult[0]) # Projection for sigma
        ])
        
        self.down2 = nn.ModuleList([
            nn.Conv2d(ch * ch_mult[0], ch * ch_mult[1], 3, 2, 1),
            Dense(ch, ch * ch_mult[1])
        ])
        
        self.down3 = nn.ModuleList([
            nn.Conv2d(ch * ch_mult[1], ch * ch_mult[2], 3, 2, 1),
            Dense(ch, ch * ch_mult[2])
        ])
        
        # 3. Upsampling (Decoder) with Skip Connections
        self.up1 = nn.ModuleList([
            nn.ConvTranspose2d(ch * ch_mult[2], ch * ch_mult[1], 4, 2, 1),
            Dense(ch, ch * ch_mult[1])
        ])
        
        self.up2 = nn.ModuleList([
            nn.ConvTranspose2d(ch * ch_mult[1] * 2, ch * ch_mult[0], 4, 2, 1), # *2 for skip concat
            Dense(ch, ch * ch_mult[0])
        ])
        
        self.up3 = nn.ModuleList([
            nn.ConvTranspose2d(ch * ch_mult[0] * 2, ch, 4, 2, 1),
            Dense(ch, ch)
        ])
        
        self.conv_out = nn.Conv2d(ch * 2, channels, 3, 1, 1)
        self.act = nn.SiLU() # Swish activation (standard in modern generative models)

    def forward(self, x, sigma):
        # Embed noise level
        embed = self.act(self.embed(sigma))
        
        # --- Encoder ---
        h1 = self.conv_in(x)
        
        # Block 1
        h2 = self.act(self.down1[0](h1) + self.down1[1](embed))
        
        # Block 2
        h3 = self.act(self.down2[0](h2) + self.down2[1](embed))
        
        # Block 3
        h4 = self.act(self.down3[0](h3) + self.down3[1](embed))
        
        # --- Decoder ---
        # Up 1
        h_up1 = self.act(self.up1[0](h4) + self.up1[1](embed))
        
        # Up 2 (Concatenate with h3)
        h_up2 = self.act(self.up2[0](torch.cat([h_up1, h3], dim=1)) + self.up2[1](embed))
        
        # Up 3 (Concatenate with h2)
        h_up3 = self.act(self.up3[0](torch.cat([h_up2, h2], dim=1)) + self.up3[1](embed))
        
        # Final Output (Concatenate with h1)
        out = self.conv_out(torch.cat([h_up3, h1], dim=1))
        
        # Normalize output by sigma (Technique from NCSN++ / Diffusion to keep scale consistent)
        return out / sigma[:, None, None, None]