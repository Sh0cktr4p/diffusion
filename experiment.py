from matplotlib import pyplot as plt

import torch as th
from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn
import torchvision

from PIL import Image
import numpy as np

from dataset_manager import (
    ImageFolderDatasetManager,
    StanfordCarsDatasetManager,
    MNISTImageDatasetManager,
) 
from unet_model import SimpleUNet


def linear_beta_schedule(timesteps: int, start: float = 0.0001, end=0.02):
    return th.linspace(start, end, timesteps)


T = 200
betas = linear_beta_schedule(T)
alphas = 1 - betas
alphas_cumprod = th.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = th.sqrt(1 / alphas)
sqrt_alphas_cumprod = th.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = th.sqrt(1 - alphas_cumprod)
posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)


def get_index_from_list(vals: th.Tensor, t: th.Tensor, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = th.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(
        sqrt_alphas_cumprod,
        t,
        x_0.shape
    )
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod,
        t,
        x_0.shape
    )

    return (
        sqrt_alphas_cumprod_t.to(device) * x_0.to(device) +
        sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device),
        noise.to(device)
    )



def get_loss(model: nn.Module, x_0: th.Tensor, t: th.Tensor):
    x_noisy, noise = forward_diffusion_sample(x_0, t)
    noise_pred = model(x_noisy)
    return F.mse_loss(noise_pred, noise)

@th.no_grad()
def sample_timestep(model: nn.Module, x: th.Tensor, t: th.Tensor):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod,
        t,
        x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(
        sqrt_recip_alphas,
        t,
        x.shape
    )
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(
        posterior_variance,
        t,
        x.shape
    )

    if t == 0:
        return model_mean
    else:
        noise = th.randn_like(x)
        return model_mean + th.sqrt(posterior_variance_t) * noise

@th.no_grad()
def sample_plot_image(model: nn.Module):
    img_size = 64

    img = th.randn(1, 3, img_size, img_size)
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    num_images = 10
    step_size = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = th.full((1,), i, dtype=th.long)
        img = sample_timestep(model, img, t)
        if i % step_size == 0:
            plt.subplot(1, num_images, i / step_size + 1)
            ImageFolderDatasetManager._to_pil_image(img[0].cpu())


def train(dl: ImageFolderDatasetManager, model: nn.Module, n_epochs: int):
    optimizer = th.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        for step, (x, _) in enumerate(dl.loader):
            optimizer.zero_grad()
            t = th.randint(0, T, (x.shape[0],), dtype=th.long)
            loss = get_loss(model, x, t)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss {loss.item()}")
            sample_plot_image(model)


if __name__ == "__main__":
    dl = ImageFolderDatasetManager(
        root="data/pokemon",
        image_size=64,
        batch_size=32
    )

    img_batch = next(iter(dl.loader))[0]
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    num_images = 10
    step_size = int(T / num_images)

    for i in range(0, T, step_size):
        print(i)
        t = th.tensor([i], dtype=th.int64)
        plt.subplot(1, num_images + 1, (i // step_size) + 1)
        noisy_batch, noise = forward_diffusion_sample(img_batch, t)
        plt.imshow(dl._to_pil_image(noisy_batch[0]))

    plt.show()
