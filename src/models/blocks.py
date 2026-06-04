import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if spatial dims differ (can happen with odd input sizes)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """Soft attention gate applied to encoder skip connection, gated by decoder signal."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        # g: decoder gate signal (coarser), x: encoder skip (finer)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # upsample g to match x spatial size
        g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        alpha = self.psi(F.relu(g1 + x1, inplace=True))
        return x * alpha


class TransformerBottleneck(nn.Module):
    """
    Flattens a (B, C, H, W) feature map into (B, H*W, d_model) tokens,
    runs a Transformer encoder, then reshapes back to (B, C, H, W).
    """

    def __init__(self, in_ch, d_model, n_heads, n_layers, spatial_size):
        super().__init__()
        self.H, self.W = spatial_size, spatial_size
        self.proj_in = nn.Linear(in_ch, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj_out = nn.Linear(d_model, in_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)   # (B, H*W, C)
        tokens = self.proj_in(tokens)            # (B, H*W, d_model)
        tokens = self.transformer(tokens)        # (B, H*W, d_model)
        tokens = self.proj_out(tokens)           # (B, H*W, C)
        out = tokens.transpose(1, 2).reshape(B, C, H, W)
        return out
