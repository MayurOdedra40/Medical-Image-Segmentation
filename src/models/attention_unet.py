import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv, DownBlock, AttentionGate


class AttentionUpBlock(nn.Module):
    """Upsample + attention-gated skip connection + DoubleConv."""

    def __init__(self, in_ch, skip_ch, out_ch, F_int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attn = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=F_int)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attn(g=x, x=skip)
        if x.shape[2:] != skip.shape[2:]:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net (Oktay et al., 2018).
    Adds soft spatial attention gates on every skip connection so the decoder
    can focus on the cardiac structures rather than background clutter.

    Input:  (B, in_channels, 256, 256)
    Output: (B, num_classes, 256, 256) — raw logits
    """

    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()

        # Encoder
        self.enc1 = DownBlock(in_channels, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder with attention gates
        # F_int = skip_ch // 2 is a common heuristic
        self.dec4 = AttentionUpBlock(in_ch=1024, skip_ch=512, out_ch=512, F_int=256)
        self.dec3 = AttentionUpBlock(in_ch=512,  skip_ch=256, out_ch=256, F_int=128)
        self.dec2 = AttentionUpBlock(in_ch=256,  skip_ch=128, out_ch=128, F_int=64)
        self.dec1 = AttentionUpBlock(in_ch=128,  skip_ch=64,  out_ch=64,  F_int=32)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.head(x)
