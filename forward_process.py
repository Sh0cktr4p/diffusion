import torch as th
import torch.nn as nn


class ForwardProcess(nn.Module):
    def __init__(
        self,
        T: int,
        start: float = 0.0001,
        end: float = 0.02,
    ):
        super().__init__()
        self.T = T
        betas, alphas, cum_alphas = self._get_schedule_variables(
            T,
            start,
            end,
        )
        prev_cum_alphas = th.cat((th.ones(1, 1, 1, 1), cum_alphas[:-1]))
        # print(cum_alphas)
        self.register_buffer(
            "betas",
            betas,
            persistent=False,
        )
        self.register_buffer(
            "sqrt_betas",
            th.sqrt(betas),
            persistent=False,
        )
        self.register_buffer(
            "sqrt_alphas",
            th.sqrt(alphas),
            persistent=False,
        )
        self.register_buffer(
            "cum_alphas",
            cum_alphas,
            persistent=False,
        )
        self.register_buffer(
            "sqrt_cum_alphas",
            th.sqrt(cum_alphas),
            persistent=False,
        )
        self.register_buffer(
            "sqrt_one_minus_cum_alphas",
            th.sqrt(1 - cum_alphas),
            persistent=False,
        )
        self.register_buffer(
            "prev_cum_alphas",
            prev_cum_alphas,
            persistent=False,
        )
        self.register_buffer(
            "sqrt_post_variance",
            th.sqrt(betas * (1 - prev_cum_alphas) / (1 - cum_alphas)),
            persistent=False,
        )

    def forward(self, x_0: th.Tensor, t: th.Tensor):
        noise = th.randn_like(x_0)

        return (
            self.sqrt_cum_alphas[t] * x_0 +
            self.sqrt_one_minus_cum_alphas[t] * noise,
            noise
        )

    def _get_schedule_variables(self, T: int, start: float, end: float):
        raise NotImplementedError("Subclasses must implement this method")


class LinScForwardProcess(ForwardProcess):
    def _get_schedule_variables(self, T: int, start: float, end: float):
        betas = th.linspace(start, end, T).reshape(-1, 1, 1, 1)
        alphas = 1 - betas
        cum_alphas = th.cumprod(alphas, dim=0)
        return betas, alphas, cum_alphas


class CosScForwardProcess(ForwardProcess):
    def _get_schedule_variables(self, T: int, start: float, end: float):
        f = th.cos(
            (th.linspace(0, 1, T + 1) + start) / (1 + start) * th.pi / 2
        ) ** 2
        cum_alphas = (f / f[0]).reshape(-1, 1, 1, 1)
        betas = th.min(
            1 - (cum_alphas[1:] / cum_alphas[:-1]),
            th.tensor(0.999)
        )
        return betas, 1 - betas, cum_alphas[:-1]


if __name__ == '__main__':
    ls = LinScForwardProcess(1000, 0.0001)
    cs = CosScForwardProcess(1000, 0.0001)

    import matplotlib.pyplot as plt

    plt.plot((cs.sqrt_cum_alphas ** 2).squeeze())
    plt.plot((ls.sqrt_cum_alphas ** 2).squeeze())
    plt.show()
