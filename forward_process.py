import torch as th
import torch.nn as nn
import torch.nn.functional as F


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
            "sqrt_cum_alphas",
            th.sqrt(cum_alphas),
            persistent=False,
        )
        self.register_buffer(
            "sqrt_one_minus_cum_alphas",
            th.sqrt(1 - cum_alphas),
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

    def get_loss(self, model, x_0, t):
        x_noisy, noise = self(x_0, t)
        noise_pred = model(x_noisy, t)
        return F.mse_loss(noise_pred, noise)

    @th.no_grad()
    def sample_timestep(self, model: nn.Module, x: th.Tensor, t: th.Tensor):
        sigma = self.sqrt_betas[t] if t > 0 else 0
        x_prev = (
            x - model(x, t) * self.betas[t] / self.sqrt_one_minus_cum_alphas
        ) / self.sqrt_alphas[t]
        noise = th.randn_like(x) * sigma
        return x_prev + noise

    @th.no_grad()
    def sample(self, model: nn.Module, img_size: int):
        device = self.betas.device
        img = th.randn(1, 3, img_size, img_size, device=device)
        T = self.betas.shape[0]
        imgs = [img]
        for i in range(T - 1, -1, -1):
            t = th.full((1,), i, dtype=th.long, device=device)
            img = self.sample_timestep(model, img, t)
            imgs.append(img)

        return imgs


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
        betas = th.min(1 - (cum_alphas[1:] / cum_alphas[:-1]), th.tensor(0.999))
        return betas, 1 - betas, cum_alphas[:-1]


if __name__ == '__main__':
    ls = LinScForwardProcess(1000, 0.0001)
    cs = CosScForwardProcess(1000, 0.0001)

    import matplotlib.pyplot as plt

    plt.plot((cs.sqrt_cum_alphas ** 2).squeeze())
    plt.plot((ls.sqrt_cum_alphas ** 2).squeeze())
    plt.show()
