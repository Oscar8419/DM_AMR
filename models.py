# models.py

import torch
import torch.nn as nn
import math
from config import CONFIG, DEVICE


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=DEVICE) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.to_qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(
            b, self.num_heads, c // self.num_heads, l), qkv)
        dots = torch.einsum("b h d i, b h d j -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h d i", attn, v)
        out = out.reshape(b, c, l)
        return self.to_out(out)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, class_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.class_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(class_emb_dim, out_channels))
        self.block1 = nn.Sequential(nn.Conv1d(
            in_channels, out_channels, 3, padding=1), nn.GroupNorm(8, out_channels), nn.SiLU())
        self.block2 = nn.Sequential(nn.Conv1d(
            out_channels, out_channels, 3, padding=1), nn.GroupNorm(8, out_channels), nn.SiLU())
        self.res_conv = nn.Conv1d(
            in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attn = AttentionBlock(out_channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, class_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h += self.time_mlp(time_emb)[..., None] + \
            self.class_mlp(class_emb)[..., None]
        h = self.block2(h)
        h = h + self.res_conv(x)
        return self.attn(h) + h


class ConditionalUNet1D(nn.Module):
    def __init__(self, signal_length: int = CONFIG["signal_length"], num_classes: int = CONFIG["num_classes"]):
        super().__init__()
        dim = 64  # TODO: ? hardcode
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(
            dim), nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))
        self.class_emb = nn.Embedding(num_classes, dim * 4)
        self.init_conv = nn.Conv1d(2, dim, 3, padding=1)
        self.down1 = ResBlock(dim, 128, dim * 4, dim * 4)
        self.down_conv1 = nn.Conv1d(128, 128, 3, stride=2, padding=1)
        self.down2 = ResBlock(128, 256, dim * 4, dim * 4)
        self.down_conv2 = nn.Conv1d(256, 256, 3, stride=2, padding=1)
        self.mid1 = ResBlock(256, 256, dim * 4, dim * 4)
        self.mid2 = ResBlock(256, 256, dim * 4, dim * 4)
        self.up_conv1 = nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1)
        # Fix: input channels = 128 (from up_conv1) + 256 (from h2 skip connection) = 384
        self.up1 = ResBlock(384, 128, dim * 4, dim * 4)
        self.up_conv2 = nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1)
        # Fix: input channels = 64 (from up_conv2) + 128 (from h1 skip connection) = 192
        self.up2 = ResBlock(192, 64, dim * 4, dim * 4)
        self.final_conv = nn.Conv1d(dim, 2, 1)

    def forward(self, x: torch.Tensor, time: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 1024)
        t_emb = self.time_mlp(time)       # (B, dim*4) -> (B, 256)
        c_emb = self.class_emb(classes)   # (B, dim*4) -> (B, 256)

        x = self.init_conv(x)             # (B, 64, 1024)

        # Downsample
        h1 = self.down1(x, t_emb, c_emb)  # (B, 128, 1024)
        h1_down = self.down_conv1(h1)     # (B, 128, 512)

        h2 = self.down2(h1_down, t_emb, c_emb)  # (B, 256, 512)
        h2_down = self.down_conv2(h2)     # (B, 256, 256)

        # Middle
        mid = self.mid1(h2_down, t_emb, c_emb)  # (B, 256, 256)
        mid = self.mid2(mid, t_emb, c_emb)     # (B, 256, 256)

        # Upsample
        up1 = self.up_conv1(mid)          # (B, 128, 512)
        up1 = torch.cat([up1, h2], dim=1)  # (B, 128+256, 512) -> (B, 384, 512)
        up1 = self.up1(up1, t_emb, c_emb)  # (B, 128, 512)

        up2 = self.up_conv2(up1)          # (B, 64, 1024)
        # (B, 64+128, 1024) -> (B, 192, 1024)
        up2 = torch.cat([up2, h1], dim=1)
        up2 = self.up2(up2, t_emb, c_emb)  # (B, 64, 1024)

        return self.final_conv(up2)       # (B, 2, 1024)


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = CONFIG["num_classes"]):
        super().__init__()
        sl = CONFIG["signal_length"]
        self.conv_stack = nn.Sequential(
            nn.Conv1d(2, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.fc_stack = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * (sl // 8), 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_stack(self.conv_stack(x))
