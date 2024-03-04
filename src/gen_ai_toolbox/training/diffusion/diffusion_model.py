from typing import Iterator, List

import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import imageio

from PIL import Image

from .noise_schedule import NoiseSchedule

from gen_ai_toolbox import training
from gen_ai_toolbox.datasets import DatasetManager
from gen_ai_toolbox.utils.training_callback import TrainingCallback
from gen_ai_toolbox.utils.torch_context_managers import eval, train


class DiffusionModel(training.GenerativeModel):
    def __init__(
        self,
        model: nn.Module,
        noise_schedule: NoiseSchedule,
        image_size: int,
        image_channels: int,
    ):
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule
        self.image_size = image_size
        self.image_channels = image_channels

    @property
    def device(self) -> th.device:
        return next(self.model.parameters()).device

    def train_model_batch(
        self,
        x: th.Tensor,
        optimizer: th.optim.Optimizer,
        t: th.Tensor | None = None,
    ) -> float:
        if t is None:
            t = self.get_random_t_vector(x.shape[0])

        with train(self.model):
            optimizer.zero_grad()
            loss = self.get_simple_loss(x, t)
            loss.backward()
            optimizer.step()
        return loss.item()

    def train_model_epoch(
        self,
        epoch: int,
        optimizer: th.optim.Optimizer,
        dataset_manager: DatasetManager,
        lr_scheduler: th.optim.lr_scheduler._LRScheduler | None = None,
        callback: TrainingCallback | None = None,
    ):
        if callback is None:
            callback = TrainingCallback()

        callback.on_epoch_begin(epoch=epoch, model=self.model)

        with train(self.model):
            for i, (x, _) in enumerate(dataset_manager.train_loader):
                self.train_model_batch(x, optimizer)

        with eval(self.model):
            t = self.get_random_t_vector(x.shape[0])
            train_batch = next(iter(dataset_manager.train_loader))
            val_batch = next(iter(dataset_manager.val_loader))
            train_loss = self.get_simple_loss(train_batch, t)
            val_loss = self.get_simple_loss(val_batch, t)

        callback.on_epoch_end(
            epoch=epoch,
            model=self.model,
            training_loss=train_loss,
            validation_loss=val_loss,
        )

        if lr_scheduler is not None:
            lr_scheduler.step()

    def train_model(
        self,
        n_epochs: int,
        optimizer: th.optim.Optimizer,
        dataset_manager: DatasetManager,
        lr_schedule: th.optim.lr_scheduler._LRScheduler | None = None,
        callback: TrainingCallback | None = None,
        start_epoch: int = 0
    ) -> nn.Module:
        if callback is None:
            callback = TrainingCallback()

        callback.on_training_begin(self.model)

        with train(self.model):
            for epoch in range(n_epochs):
                self.train_model_epoch(
                    epoch=epoch + start_epoch,
                    optimizer=optimizer,
                    dataset_manager=dataset_manager,
                    lr_scheduler=lr_schedule,
                    callback=callback,
                )

        callback.on_training_end(self.model)

    def get_simple_loss(self, x_0: th.Tensor, t: th.Tensor) -> th.Tensor:
        device = self.device
        x_0 = x_0.to(device)
        t = t.to(device)
        x_t, noise = self.noise_schedule.diffuse(x_0, t)
        return F.mse_loss(noise, self.model(x_t, t))

    def get_random_image_batch(self, batch_size: int) -> th.Tensor:
        return th.randn(
            batch_size,
            self.image_channels,
            self.image_size,
            self.image_size
        ).to(self.device)

    def get_random_t_vector(self, batch_size: int) -> th.Tensor:
        return th.randint(
            0,
            self.noise_schedule.T,
            (batch_size,),
            dtype=th.long,
        )

    def sample_trajectory_tensor_iterator(
        self,
        x: th.Tensor,
    ) -> Iterator[th.Tensor]:
        yield x

        for i in range(self.T - 1, -1, -1):
            x = self.noise_schedule.sample_timestep(
                self.model,
                x,
                th.tensor(i, dtype=th.long, device=x.device),
            )
            yield x

    def trajectory_sample_image_iterator(
        self,
        x: th.Tensor
    ) -> Iterator[Image.Image]:
        return map(
            lambda tensor: DatasetManager._to_pil_image(tensor[0]),
            self.sample_trajectory_tensor_iterator(x)
        )

    @th.inference_mode()
    def plot_trajectory(self, n_images: int = 5) -> List[Image.Image]:
        img_dim = 2

        with eval(self.model):
            imgs = list(self.trajectory_sample_image_iterator(
                self.get_random_image_batch(1)
            ))

        _, axes = plt.subplots(
            1,
            n_images,
            figsize=(n_images * img_dim, img_dim)
        )

        for plot_index, trajectory_index in enumerate(
            np.linspace(0, len(imgs) - 1, n_images, dtype=int)
        ):
            axes[plot_index].imshow(imgs[trajectory_index])
            axes[plot_index].axis("off")

        plt.show()

    @th.inference_mode()
    def sample_trajectory_video(self, path: str):
        writer = imageio.get_writer(path, fps=60)

        with eval(self.model):
            for img in self.trajectory_sample_image_iterator(
                self.get_random_image_batch(1),
            ):
                writer.append_data(img.cpu().numpy())

        writer.close()

    @th.inference_mode()
    def render_batch(
        self,
        n_rows: int = 4,
        n_columns: int = 5,
        path: str | None = None,
    ) -> List[Image.Image]:
        img_dim = 2

        batch = self.get_random_image_batch(n_rows * n_columns)

        with eval(self.model):
            for batch in self.sample_trajectory_tensor_iterator(batch):
                pass

        images = [
            DatasetManager._to_pil_image(row)
            for row in batch
        ]

        _, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(n_columns*img_dim, n_rows*img_dim)
        )

        for i, img in enumerate(images):
            axes[i // n_columns, i % n_columns].imshow(img)
            axes[i // n_columns, i % n_columns].axis("off")

        if path is None:
            plt.show()
        else:
            plt.savefig(path)

    def save_model_state_dict(
        self,
        path: str,
    ):
        th.save(self.model.state_dict(), path)

    def load_model_state_dict(
        self,
        path: str,
    ):
        self.model.load_state_dict(th.load(path))


if __name__ == '__main__':
    # from dataset_manager import (  # noqa: F401
    #     ImageFolderDatasetManager,
    #     MNISTImageDatasetManager,
    #     ImageNetDatasetManager,
    #     StanfordCarsDatasetManager,
    #     CIFAR10DatasetManager,
    #     CelebADatasetManager,
    # )
    # from forward_process import (  # noqa: F401
    #     LinScForwardProcess,
    #     CosScForwardProcess,
    # )
    # from unet_model import SimpleUNet
    # from my_unet import UNet

    # th.random.manual_seed(0)
    # device = "cuda"
    # model = SimpleUNet().to(device)
    # model.load_state_dict(th.load("models/celeba_18_su.pt"))
    # dm = CelebADatasetManager(64, 128)
    # fp = LinScForwardProcess(1000).to(device)
    # optimizer = th.optim.Adam(model.parameters(), lr=2e-4)
    # trainer = DiffusionTrainer(model, optimizer, fp, dm, device)

    # trainer.train(
    #     5001,
    #     plot_checkpoints=False,
    #     checkpoint_video_path="gifs/celeba/comp/",
    #     model_save_path="models/celeba_su.pt"
    # )
    pass
