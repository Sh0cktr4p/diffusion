import torch as th
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim, 4*4*512),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=4,
                stride=2,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=5,
                stride=2,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=7,
                stride=3,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                64,
                32,
                kernel_size=7,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                32,
                3,
                kernel_size=4,
            ),
        )

    def forward(
        self,
        x: th.Tensor,
    ) -> th.Tensor:
        return self.net(x)


if __name__ == "__main__":
    z_dim = 100
    gen = Generator(z_dim)
    print(gen)
    print(
        "Num params: ",
        sum(p.numel() for p in gen.parameters() if p.requires_grad)
    )
    x = th.randn(1, z_dim)
    print(gen(x).shape)
    print("Generator test passed")
