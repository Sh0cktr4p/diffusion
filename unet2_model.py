import torch as th
import torch.nn as nn

from einops.layers.torch import Rearrange

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
            h += self.mlp(t).reshape(*t.shape, 1, 1)
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
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=2, p2=2),
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


class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int | None = None,
        out_dim: int | None = None,
        dim_mults=(1, 2, 4, 8),
        channels: int = 3,
        sinusoidal_pos_emb_theta: int = 10000,
        convnext_block_groups: int = 8,
    ):
        super().__init__()
        self.channels = channels
        input_channels = channels
        self.init_dim = init_dim or dim
        self.init_conv = nn.Conv2d(input_channels, self.init_dim, 7, padding=3)

        dims = [self.init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        sinu_pos_emb = SinusoidalPositionEmbedding(
            dim,
            theta=sinusoidal_pos_emb_theta
        )

        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            in_channels=dim_in,
                            out_channels=dim_in,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        ConvNextBlock(
                            in_channels=dim_in,
                            out_channels=dim_in,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(
            mid_dim,
            mid_dim,
            time_embedding_dim=time_dim
        )
        self.mid_block2 = ConvNextBlock(
            mid_dim,
            mid_dim,
            time_embedding_dim=time_dim,
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            in_channels=dim_out + dim_in,
                            out_channels=dim_out,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        ConvNextBlock(
                            in_channels=dim_out + dim_in,
                            out_channels=dim_out,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                    ]
                )
            )

        default_out_dim = channels
        self.out_dim = out_dim or default_out_dim

        self.final_res_block = ConvNextBlock(dim * 2, dim, time_embedding_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        b, _, h, w = x.shape
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        unet_stack = []
        for down1, down2, downsample in self.downs:
            x = down1(x, t)
            unet_stack.append(x)
            x = down2(x, t)
            unet_stack.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for up1, up2, upsample in self.ups:
            x = torch.cat((x, unet_stack.pop()), dim=1)
            x = up1(x, t)
            x = torch.cat((x, unet_stack.pop()), dim=1)
            x = up2(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)

        return self.final_conv(x)