import torch.nn as nn
from .blocks import DoubleConv, DownBlock, UpBlock


class UNet(nn.Module):
    """
    Classic U-Net (Ronneberger et al., 2015).
    Input:  (B, in_channels, 256, 256)
    Output: (B, num_classes, 256, 256) — raw logits
    """

    def __init__(self, in_channels=1, num_classes=4, bilinear=True):
        super().__init__()

        # Encoder
        self.enc1 = DownBlock(in_channels, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder — each UpBlock receives (upsampled + skip) so in_ch = 2 × skip_ch
        self.dec4 = UpBlock(1024 + 512, 512, bilinear)
        self.dec3 = UpBlock(512 + 256, 256, bilinear)
        self.dec2 = UpBlock(256 + 128, 128, bilinear)
        self.dec1 = UpBlock(128 + 64, 64, bilinear)

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
