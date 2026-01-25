import torch
import torch.nn as nn
from .unet import UNet

class NCSN(nn.Module):
    def __init__(self, sigmas):
        super().__init__()
        self.sigmas = sigmas
        self.unet = UNet()

    def forward(self, x, sigma_idx):
        sigma = self.sigmas[sigma_idx].view(-1, 1, 1, 1)
        x = x / sigma
        score = self.unet(x)
        return score / sigma
