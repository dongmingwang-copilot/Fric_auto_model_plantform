from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base: int = 32, dropout: float = 0.0):
        super().__init__()
        self.down1 = ConvBlock(1, base)
        self.down2 = ConvBlock(base, base * 2)
        self.down3 = ConvBlock(base * 2, base * 4)
        self.down4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.mid = ConvBlock(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.conv4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        m = self.drop(self.mid(self.pool(d4)))
        x = self.conv4(torch.cat([self.up4(m), d4], dim=1))
        x = self.conv3(torch.cat([self.up3(x), d3], dim=1))
        x = self.conv2(torch.cat([self.up2(x), d2], dim=1))
        x = self.conv1(torch.cat([self.up1(x), d1], dim=1))
        return self.out(x)

