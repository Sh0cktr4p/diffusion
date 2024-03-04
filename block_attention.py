import torch as th
import torch.nn as nn


class BlockAttention(nn.Module):
    def __init__(
        self,
        gate_in_channels: int,
        residual_in_channels: int,
    ):
        super().__init__()
        self.gate_conv = nn.Conv2d(
            gate_in_channels,
            gate_in_channels,
            kernel_size=1,
            stride=1,
        )
        self.residual_conv = nn.Conv2d(
            residual_in_channels,
            gate_in_channels,
            kernel_size=1,
            stride=1,
        )
        self.in_conv = nn.Conv2d(
            gate_in_channels,
            1,
            kernel_size=1,
            stride=1,
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: th.Tensor, res: th.Tensor) -> th.Tensor:
        in_attention = self.relu(self.gate_conv(x) + self.residual_conv(res))
        in_attention = self.in_conv(in_attention)
        in_attention = self.sigmoid(in_attention)
        return in_attention * x
