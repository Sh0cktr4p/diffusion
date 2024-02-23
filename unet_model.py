import math

import torch as th
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: th.Tensor):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = t.reshape(-1, 1) * emb.reshape(1, -1)
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        up: bool,
    ):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if up:
            self.conv1 = nn.Conv2d(
                2*in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )
            self.transform = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )
            self.transform = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.pool = nn.MaxPool2d(3, stride=2)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: th.Tensor, t: th.Tensor) -> th.Tensor:
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.reshape(*time_emb.shape, 1, 1)
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 512)
        up_channels = (512, 512, 256, 128, 64)
        out_dim = 1
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # Initial projection
        self.conv0 = nn.Conv2d(
            image_channels,
            down_channels[0],
            kernel_size=3,
            padding=1
        )

        self.ds = nn.ModuleList([
            UNetBlock(
                down_channels[i],
                down_channels[i + 1],
                time_emb_dim,
                up=False
            )
            for i in range(len(down_channels) - 1)
        ])

        self.us = nn.ModuleList([
            UNetBlock(
                up_channels[i],
                up_channels[i + 1],
                time_emb_dim,
                up=True
            )
            for i in range(len(up_channels) - 1)
        ])

        self.output = nn.Conv2d(
            up_channels[-1],
            3,
            out_dim,
        )

    def forward(self, x, t):
        # Embed time
        t_emb = self.time_mlp(t)

        x = self.conv0(x)
        residual_inputs = []
        for down in self.ds:
            x = down(x, t_emb)
            residual_inputs.append(x)
        for up in self.us:
            residual_x = residual_inputs.pop()
            x = th.cat((x, residual_x), dim=1)
            x = up(x, t_emb)

        return self.output(x)


if __name__ == '__main__':
    model = SimpleUNet()
    print(
        "Num params: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    x = th.randn(32, 3, 64, 64).cuda()
    t = th.randn(32, 1).cuda()
    out = model(x, t)
    print(out.shape)
