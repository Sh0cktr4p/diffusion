from typing import List, Tuple

import torch as th
import torch.nn as nn

from einops.layers.torch import Rearrange

from block_attention import BlockAttention
from sinusoidal_position_embedding import SinusoidalPositionEmbedding


class ConvNextBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        multiplier: int = 2,
        time_emb_dim: int | None = None,
        norm: bool = True,
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, in_channels),
            )
            if time_emb_dim is not None
            else None
        )

        self.in_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels,
        )

        self.block = nn.Sequential(
            nn.GroupNorm(1, in_channels) if norm else nn.Identity(),
            nn.Conv2d(
                in_channels,
                out_channels * multiplier,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.GroupNorm(
                1,
                out_channels * multiplier
            ) if norm else nn.Identity(),
            nn.Conv2d(
                out_channels * multiplier,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        self.residual_conv = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: th.Tensor, t: th.Tensor | None = None) -> th.Tensor:
        h = self.in_conv(x)
        if self.mlp is not None and t is not None:
            h += self.mlp(t).reshape(t.shape[0], h.shape[1], 1, 1)
        h = self.block(h)
        return h + self.residual_conv(x)


class Downsample(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (p1 p2 c) h w", p1=2, p2=2),
            nn.Conv2d(
                dim * 4,
                dim_out or dim,
                kernel_size=1,
            ),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


class Upsample(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                dim,
                dim_out or dim,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


class DownPath(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()

        self.bu = ConvNextBlock(
            in_channels,
            out_channels,
            time_emb_dim=time_emb_dim
        )
        self.down = Downsample(out_channels)
        self.bd = ConvNextBlock(out_channels, out_channels)

    def forward(
        self,
        x: th.Tensor,
        t: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        x = self.bu(x, t)
        h = x
        x = self.down(h)
        x = self.bd(x, t)
        return x, h


class UpPath(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        res_channels: int | None = None,
    ):
        super().__init__()

        self.bd = ConvNextBlock(
            in_channels,
            in_channels,
            time_emb_dim=time_emb_dim
        )
        self.up = Upsample(in_channels)
        self.att = BlockAttention(
            in_channels,
            res_channels or in_channels
        )
        self.bu = ConvNextBlock(
            in_channels,
            out_channels,
        )

    def forward(self, x: th.Tensor, h: th.Tensor, t: th.Tensor) -> th.Tensor:
        x = self.bd(x, t)
        x = self.up(x)
        x = self.att(x, h)
        x = self.bu(x, t)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        output_channels: int = 3,
        n_lat_blocks: int = 2,
        time_emb_dim: int = 32,
        down_channels: List[int] = (32, 32),
        up_channels: List[int] | None = None,
        lat_channels: int = 64,
    ):
        super().__init__()

        up_channels = up_channels or list(reversed(down_channels))

        assert len(down_channels) == len(up_channels)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.c0 = nn.Conv2d(
            image_channels,
            down_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.down = nn.ModuleList(
            [
                DownPath(
                    in_channels=down_channels[i],
                    out_channels=(list(down_channels) + [lat_channels])[i + 1],
                    time_emb_dim=time_emb_dim
                )
                for i in range(len(down_channels))
            ]
        )

        self.lat = nn.ModuleList(
            [
                ConvNextBlock(
                    in_channels=lat_channels,
                    out_channels=lat_channels,
                    time_emb_dim=time_emb_dim
                )
                for _ in range(n_lat_blocks)
            ]
        )

        self.up = nn.ModuleList(
            [
                UpPath(
                    in_channels=([lat_channels] + list(up_channels))[i],
                    out_channels=([lat_channels] + list(up_channels))[i + 1],
                    time_emb_dim=time_emb_dim,
                    res_channels=(
                        list(down_channels) + [lat_channels]
                    )[-i - 1],
                )
                for i in range(len(up_channels))
            ]
        )

        self.cn = nn.Conv2d(
            up_channels[-1],
            output_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: th.Tensor, t: th.Tensor) -> th.Tensor:
        t_emb = self.time_mlp(t)
        x = self.c0(x)
        h = []
        for d in self.down:
            x, hi = d(x, t_emb)
            h.append(hi)
        for lb in self.lat:
            x = lb(x, t_emb)
        for u in self.up:
            x = u(x, h.pop(), t_emb)
        x = self.cn(x)
        return x


if __name__ == '__main__':
    print("num params:", sum(p.numel() for p in UNet().parameters()))

    print(UNet()(th.randn(128, 3, 64, 64), th.randint(0, 1000, (128, 1))).shape)
