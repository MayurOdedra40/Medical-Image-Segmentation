import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv, DownBlock, TransformerBottleneck


class TransUNet(nn.Module):
    """
    TransUNet (Chen et al., 2021) — CNN encoder + Transformer bottleneck + CNN decoder.

    The first three encoder stages extract local CNN features with spatial skip connections.
    The feature map at the deepest CNN stage (256 channels, 32×32) is tokenised and passed
    through a Transformer encoder to capture long-range dependencies. The decoder mirrors
    the CNN encoder using skip connections.

    Input:  (B, in_channels, 256, 256)
    Output: (B, num_classes, 256, 256) — raw logits

    Args:
        d_model:  Transformer token dimension (projected from 256 channels).
        n_heads:  Number of attention heads.
        n_layers: Number of Transformer encoder layers.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=4,
        d_model=512,
        n_heads=8,
        n_layers=6,
    ):
        super().__init__()

        # CNN encoder — 3 stages (spatial: 256 → 128 → 64 → 32)
        self.enc1 = DownBlock(in_channels, 64)   # skip: 64ch @ 256×256
        self.enc2 = DownBlock(64, 128)            # skip: 128ch @ 128×128
        self.enc3 = DownBlock(128, 256)           # skip: 256ch @ 64×64

        # Extra down to reach 32×32 before the transformer
        self.enc4 = DownBlock(256, 256)           # skip: 256ch @ 32×32

        # Transformer bottleneck on the 32×32 feature map
        # spatial_size=32 is fixed because input is always 256×256
        self.transformer = TransformerBottleneck(
            in_ch=256, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, spatial_size=32,
        )

        # CNN decoder — mirrors encoder, uses skip connections
        self.dec4 = _UpBlock(256 + 256, 256)   # concat with enc4 skip
        self.dec3 = _UpBlock(256 + 256, 128)   # concat with enc3 skip
        self.dec2 = _UpBlock(128 + 128, 64)    # concat with enc2 skip
        self.dec1 = _UpBlock(64  + 64,  64)    # concat with enc1 skip

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x, s1 = self.enc1(x)   # s1: 64ch  @ 256×256
        x, s2 = self.enc2(x)   # s2: 128ch @ 128×128
        x, s3 = self.enc3(x)   # s3: 256ch @ 64×64
        x, s4 = self.enc4(x)   # s4: 256ch @ 32×32

        x = self.transformer(x)  # still 256ch @ 32×32

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.head(x)


class _UpBlock(nn.Module):
    """Bilinear upsample + skip concat + DoubleConv (private to TransUNet)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
