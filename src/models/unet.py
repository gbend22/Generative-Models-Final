import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(h + self.skip(x))

class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        self.down1 = ResidualBlock(in_channels, base_channels)
        self.down2 = ResidualBlock(base_channels, base_channels * 2)
        self.down3 = ResidualBlock(base_channels * 2, base_channels * 4)

        self.up2 = ResidualBlock(base_channels * 4, base_channels * 2)
        self.up1 = ResidualBlock(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, in_channels, 1)

        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))

        u2 = F.interpolate(d3, scale_factor=2)
        u2 = self.up2(u2 + d2)

        u1 = F.interpolate(u2, scale_factor=2)
        u1 = self.up1(u1 + d1)

        return self.out(u1)
