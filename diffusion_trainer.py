from typing import List
import os

import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import imageio

from PIL import Image

from omegaconf import OmegaConf

from dataset_manager import DatasetManager


class DiffusionTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: th.optim.Optimizer,
        forward_process: nn.Module,
        dataset_manager: DatasetManager,
        device: th.device,
        lr_scheduler: th.optim.lr_scheduler._LRScheduler | None = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.forward_process = forward_process.to(device)
        self.dataset_manager = dataset_manager
        self.lr_scheduler = lr_scheduler
        self.device = device

    def get_simple_loss(self, x_0, t):
        x_noisy, noise = self.forward_process(x_0, t)
        return F.mse_loss(noise, self.model(x_noisy, t))

    def get_vlb_loss(self, x_0, t):
        pass

    def train(
        self,
        n_epochs: int,
        plot_checkpoints: bool = False,
        model_save_path: str | None = None,
        checkpoint_video_path: str | None = None,
    ):
        if model_save_path is not None:
            assert os.path.exists(os.path.dirname(model_save_path)),  \
                "Model save path must exist"
            assert model_save_path.endswith(".pt"), \
                "Model save path must end with .pth"

        if checkpoint_video_path is not None:
            assert os.path.exists(checkpoint_video_path), \
                "Checkpoint video path must exist"

        for epoch in range(n_epochs):
            for x, _ in self.dataset_manager.loader:
                self.optimizer.zero_grad()
                t = th.randint(
                    0,
                    self.forward_process.T,
                    (x.shape[0],),
                    dtype=th.long,
                )
                loss = self.get_simple_loss(
                    x.to(self.device),
                    t.to(self.device)
                )
                loss.backward()
                self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if epoch % 1 == 0:
                print(f"Epoch {epoch}: Loss {loss.item()}")
                if plot_checkpoints:
                    self.sample_to_plot(10)

                if checkpoint_video_path is not None:
                    self.sample_to_video(
                        os.path.join(checkpoint_video_path, f"{epoch}.mp4")
                    )

                if model_save_path is not None:
                    th.save(self.model.state_dict(), model_save_path)

    @th.inference_mode()
    def sample_timestep(self, x: th.Tensor, t: th.Tensor):
        train_mode_active = self.model.training
        self.model.eval()
        mu = self.forward_process.div_sqrt_alphas[t] * (
            x - (
                self.model(x, t) * self.forward_process.betas[t] /
                self.forward_process.sqrt_one_minus_cum_alphas[t]
            )
        )

        # sigma = self.forward_process.sqrt_betas[t] if t > 0 else 0
        sigma = self.forward_process.sqrt_post_variance[t] if t[0] > 0 else 0
        noise = th.randn_like(x) * sigma
        if train_mode_active:
            self.model.train()
        return mu + noise

    @th.inference_mode()
    def sample(self) -> List[Image.Image]:
        train_mode_active = self.model.training
        self.model.eval()
        img_size = self.dataset_manager.img_size
        img = th.randn(1, 3, img_size, img_size).to(self.device)
        T = self.forward_process.T
        imgs = [self.dataset_manager._to_pil_image(img[0])]
        for i in range(T - 1, -1, -1):
            print(i)
            t = th.full((1,), i, dtype=th.long, device=self.device)
            img = self.sample_timestep(img, t)
            imgs.append(self.dataset_manager._to_pil_image(img[0]))

        if train_mode_active:
            self.model.train()

        return imgs

    @th.inference_mode()
    def render_batch(self, batch_size: int) -> List[Image.Image]:
        train_mode_active = self.model.training
        self.model.eval()
        img_size = self.dataset_manager.img_size
        imgs = th.randn(batch_size, 3, img_size, img_size).to(self.device)
        T = self.forward_process.T
        for i in range(T - 1, -1, -1):
            print(i)
            t = th.full((batch_size,), i, dtype=th.long, device=self.device)
            imgs = self.sample_timestep(imgs, t)

        imgs = [self.dataset_manager._to_pil_image(img) for img in imgs]
        if train_mode_active:
            self.model.train()

        return imgs

    def sample_to_plot(self, num_images: int = 10):
        imgs = self.sample()
        _, axes = plt.subplots(1, num_images, figsize=(20, 2))
        step_size = int(len(imgs) / num_images)
        for i, img in enumerate(imgs[::step_size][:num_images]):
            axes[i].imshow(img)
            axes[i].axis("off")
        plt.show()

    def sample_to_video(self, path: str):
        imgs = self.sample()
        writer = imageio.get_writer(path, fps=60)
        for img in imgs:
            writer.append_data(np.array(img))
        writer.close()

    @staticmethod
    def from_config(config: OmegaConf) -> "DiffusionTrainer":
        pass


if __name__ == '__main__':
    from dataset_manager import (  # noqa: F401
        ImageFolderDatasetManager,
        MNISTImageDatasetManager,
        ImageNetDatasetManager,
        StanfordCarsDatasetManager,
        CIFAR10DatasetManager,
        CelebADatasetManager,
    )
    from forward_process import (  # noqa: F401
        LinScForwardProcess,
        CosScForwardProcess,
    )
    from unet_model import SimpleUNet
    from my_unet import UNet

    th.random.manual_seed(0)
    device = "cuda"
    model = SimpleUNet().to(device)
    model.load_state_dict(th.load("models/celeba_18_su.pt"))
    # model = UNet().to(device)
    # dm = ImageFolderDatasetManager(
    #     root="data/pokemon",
    #     image_size=64,
    #     batch_size=128
    # )
    # dm = CIFAR10DatasetManager(32, 128)
    dm = CelebADatasetManager(64, 128)
    # dm = MNISTImageDatasetManager(batch_size=32, image_size=64)
    fp = LinScForwardProcess(1000).to(device)
    # fp = CosScForwardProcess(1000, 0.008).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=2e-4)
    # scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    trainer = DiffusionTrainer(model, optimizer, fp, dm, device)

    # trainer.sample_to_video(
    #     "gifs/pokemon/id.mp4"
    # )
    trainer.train(
        5001,
        plot_checkpoints=False,
        checkpoint_video_path="gifs/celeba/comp/",
        model_save_path="models/celeba_su.pt"
    )
    # imgs = trainer.sample()
    # plt.figure(figsize=(15, 15))
    # plt.axis("off")
    # for i, img in enumerate(imgs):
    #     plt.subplot(1, len(imgs), i + 1)
    #     plt.imshow(img)
    # plt.show()
