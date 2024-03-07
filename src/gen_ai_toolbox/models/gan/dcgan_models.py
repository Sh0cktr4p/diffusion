import torch as th
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int = 100,
        output_channels: int = 3,
        feature_map_size: int = 64,
        squash_output: bool = False,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                z_dim,
                feature_map_size * 8,
                kernel_size=4,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                feature_map_size * 8,
                feature_map_size * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                feature_map_size * 4,
                feature_map_size * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                feature_map_size * 2,
                feature_map_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                feature_map_size,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh() if squash_output else nn.Identity(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        feature_map_size: int = 64,
        use_bn: bool = False,
        squash_output: bool = False,
    ):
        super().__init__()

        norm_layer = nn.BatchNorm2d if use_bn else nn.InstanceNorm2d

        self.net = nn.Sequential(
            nn.Conv2d(
                input_channels,
                feature_map_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                feature_map_size,
                feature_map_size * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            norm_layer(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                feature_map_size * 2,
                feature_map_size * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            norm_layer(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                feature_map_size * 4,
                feature_map_size * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            norm_layer(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                feature_map_size * 8,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid() if squash_output else nn.Identity(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


if __name__ == '__main__':
    gen = Generator()
    print(
        "Number of generator parameters:",
        sum(p.numel() for p in gen.parameters() if p.requires_grad)
    )

    print(gen(th.randn(1, 100, 1, 1)).shape)

    disc = Discriminator()
    print(
        "Number of discriminator parameters:",
        sum(p.numel() for p in disc.parameters() if p.requires_grad)
    )

    print(disc(th.randn(1, 3, 64, 64)).shape)
