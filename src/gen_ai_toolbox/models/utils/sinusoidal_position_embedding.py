import torch as th
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        half_dim = dim // 2
        emb = th.log(th.tensor(theta)) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim) * -emb).reshape(1, -1)

        self.register_buffer("emb", emb)

    def forward(self, t: th.Tensor):
        emb = t.reshape(-1, 1) * self.emb
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
