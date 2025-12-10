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


class GRU(nn.Module):
    def __init__(self, input_dim=16, hidden_size=128, num_classes=24, bidire=False):
        super().__init__()
        self.input_dim = input_dim

        # 双向GRU层配置
        self.gru1 = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,    # 输入格式为(batch, seq_len, input_size)
            bidirectional=bidire
        )

        self.gru2 = nn.GRU(
            input_size=hidden_size * 2 if bidire else hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidire
        )

        # 全连接层
        if bidire:
            self.fc = nn.Linear(2*hidden_size, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        # shape=[N, 2, 1024]
        # 输入形状: [batch_size, seq_len, input_dim]
        # x = x.reshape(x.size(0),-1, self.input_dim)
        batch_size = x.size(0)
        x = x.reshape(batch_size, 2, -1, self.input_dim//2).transpose(-2, -
                                                                      3).contiguous().reshape(batch_size, -1, self.input_dim)

        # 第一层GRU（返回所有时间步输出）
        gru1_out, _ = self.gru1(x)  # out shape:

        # 第二层GRU（仅返回最后时间步）
        gru2_out, _ = self.gru2(gru1_out)  # out shape:
        last_output = gru2_out[:, -1, :]    # 取最后时间步 [batch, 128]

        # 全连接层
        out = self.fc(last_output)
        return out


# --- DiT Components ---

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed1D(nn.Module):
    """ 1D Image to Patch Embedding """

    def __init__(self, signal_length=1024, patch_size=16, in_chans=2, embed_dim=768):
        super().__init__()
        num_patches = signal_length // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.num_patches = num_patches

    def forward(self, x):
        # x: (B, C, L)
        x = self.proj(x)  # (B, D, N)
        x = x.transpose(1, 2)  # (B, N, D)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=1)

        # Attention
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    embed_dim: output dimension for each position
    length: number of positions to be encoded
    """
    import numpy as np
    if embed_dim % 2 != 0:
        raise ValueError("Embed dim must be divisible by 2")

    grid = np.arange(length, dtype=np.float32)
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    out = np.einsum('m,d->md', grid, omega)  # (L, D/2)

    emb_sin = np.sin(out)  # (L, D/2)
    emb_cos = np.cos(out)  # (L, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (L, D)
    return emb


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        signal_length=CONFIG["signal_length"],
        patch_size=16,
        in_channels=2,
        hidden_size=384,
        depth=12//2,
        num_heads=6,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=CONFIG["num_classes"],
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed1D(
            signal_length, patch_size, in_channels, hidden_size)
        self.t_embedder = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.y_embedder = nn.Embedding(num_classes, hidden_size)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.x_embedder.num_patches)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size * out_channels)
        imgs: (N, out_channels, L)
        """
        c = self.out_channels
        p = self.patch_size
        h = x.shape[1]  # num_patches

        x = x.reshape(shape=(x.shape[0], h, p, c))
        x = torch.einsum('nhpc->ncph', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        x: (N, C, L)
        t: (N,)
        y: (N,)
        """
        x = self.x_embedder(
            x) + self.pos_embed  # (N, T, D), where T = L/patch_size
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y)                   # (N, D)
        c = t + y                                # (N, D)

        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)

        # (N, T, patch_size * out_channels)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)                   # (N, out_channels, L)
        return x


def get_diffusion_model(model_type: str, **kwargs):
    if model_type == "unet1d":
        return ConditionalUNet1D(**kwargs)
    elif model_type == "dit":
        return DiffusionTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown diffusion model type: {model_type}")


def get_classifier_model(model_type: str, **kwargs):
    if model_type == "cnn":
        return CNNClassifier(**kwargs)
    elif model_type == "gru":
        return GRU(**kwargs)
    else:
        raise ValueError(f"Unknown classifier model type: {model_type}")
