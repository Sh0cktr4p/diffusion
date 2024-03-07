from matplotlib import pyplot as plt

import numpy as np
import torch as th
from torch.utils import data
import torchvision

from PIL import Image


def rgba_to_rgb_img_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGBA")
        img_arr = np.array(img)
        return Image.fromarray(
            (img_arr[:, :, :3] * (img_arr[:, :, 3:] / 255)).astype(np.uint8)
        )


class DatasetManager:
    def __init__(
        self,
        dataset: data.Dataset,
        batch_size: int,
        img_size: int,
        val_split: float = 0.1,
    ):
        self.train_set, self.val_set = data.random_split(
            dataset,
            [
                int(len(dataset) * (1 - val_split)),
                int(len(dataset) * val_split),
            ]
        )
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_loader = self._get_dataloader(
            dataset=self.train_set,
            batch_size=batch_size
        )
        self.val_loader = self._get_dataloader(
            dataset=self.val_set,
            batch_size=batch_size
        )

    @staticmethod
    def _get_dataloader(dataset: data.Dataset, batch_size: int):
        return data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,  # drop last batch if it's not full
        )

    @staticmethod
    def _get_transform(image_size: int) -> torchvision.transforms.Compose:
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),  # Scales to [0, 1]
            torchvision.transforms.Normalize(   # Normalizes to [-1, 1]
                0.5,
                0.5
            ),
        ])

    @staticmethod
    def _to_pil_image(tensor: th.Tensor) -> Image.Image:
        reverse_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x.clamp(-1, 1)),
            torchvision.transforms.Normalize(
                -1,
                2
            ),
            torchvision.transforms.ToPILImage(),
        ])

        return reverse_transform(tensor)

    @staticmethod
    def render_batch(
        tensor: th.Tensor,
        n_rows: int,
        n_columns: int,
        img_dim: float = 2.0,
        path: str | None = None,
    ):
        assert tensor.shape[0] == n_rows * n_columns

        images = [DatasetManager._to_pil_image(row) for row in tensor]

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

    def show_images(self, num_samples=20, cols=4, from_train_set: bool = True):
        plt.figure(figsize=(10, 10))
        for i, img in list(enumerate(
            self.train_set if from_train_set else self.val_set
        ))[:num_samples]:
            plt.subplot(num_samples // cols + 1, cols, i + 1)
            plt.imshow(img[0].permute(1, 2, 0))
        plt.show()


class ImageFolderDatasetManager(DatasetManager):
    def __init__(
        self,
        root: str,
        batch_size: int,
        img_size: int,
        val_split: float = 0.1
    ):
        dataset = ImageFolderDatasetManager._get_dataset(root, img_size)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            img_size=img_size,
            val_split=val_split,
        )

    @staticmethod
    def _get_dataset(root: str, image_size: int):
        return torchvision.datasets.ImageFolder(
            root=root,
            transform=ImageFolderDatasetManager._get_transform(image_size),
            loader=rgba_to_rgb_img_loader,
        )


class MNISTDatasetManager(DatasetManager):
    def __init__(
        self,
        batch_size: int,
        img_size: int,
        val_split: float = 0.1,
    ):
        dataset = MNISTDatasetManager._get_dataset(img_size)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            img_size=img_size,
            val_split=val_split,
        )

    @staticmethod
    def _get_dataset(image_size: int):
        return torchvision.datasets.MNIST(
            root="./data",
            transform=MNISTDatasetManager._get_transform(image_size),
            download=True,
        )


class StanfordCarsDatasetManager(DatasetManager):
    def __init__(
        self,
        batch_size: int,
        img_size: int,
        val_split: float = 0.1,
    ):
        dataset = StanfordCarsDatasetManager._get_dataset(img_size)
        super().__init__(dataset, batch_size, img_size)

    @staticmethod
    def _get_dataset(image_size: int):
        return torchvision.datasets.StanfordCars(
            root="data/",
            transform=StanfordCarsDatasetManager._get_transform(image_size),
            download=True,
        )


class CelebADatasetManager(DatasetManager):
    def __init__(
        self,
        batch_size: int,
        img_size: int,
        val_split: float = 0.1,
    ):
        dataset = CelebADatasetManager._get_dataset(img_size)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            img_size=img_size,
            val_split=val_split,
        )

    @staticmethod
    def _get_dataset(image_size: int):
        return torchvision.datasets.CelebA(
            root="data/",
            transform=CelebADatasetManager._get_transform(image_size),
            download=True,
        )


class ImageNetDatasetManager(DatasetManager):
    def __init__(
        self,
        batch_size: int,
        img_size: int,
        val_split: float = 0.1,
    ):
        dataset = ImageNetDatasetManager._get_dataset(img_size)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            img_size=img_size,
            val_split=val_split,
        )

    @staticmethod
    def _get_dataset(image_size: int):
        return torchvision.datasets.ImageNet(
            root="data/",
            transform=ImageNetDatasetManager._get_transform(image_size),
            download=True,
        )


class CIFAR10DatasetManager(DatasetManager):
    def __init__(
        self,
        batch_size: int,
        img_size: int,
        val_split: float = 0.1,
    ):
        dataset = CIFAR10DatasetManager._get_dataset(img_size)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            img_size=img_size,
            val_split=val_split,
        )

    @staticmethod
    def _get_dataset(image_size: int):
        return torchvision.datasets.CIFAR10(
            root="data/",
            transform=CIFAR10DatasetManager._get_transform(image_size),
            download=True,
        )


if __name__ == '__main__':
    from gen_ai_toolbox.training.diffusion.noise_schedule import (
        LinearNoiseSchedule
    )

    dl = ImageFolderDatasetManager(
        root="data/pokemon",
        img_size=64,
        batch_size=32
    )

    T = 1000

    img_batch = next(iter(dl.train_loader))[0]
    plt.figure(figsize=(15, 5))
    plt.axis("off")
    num_images = 10
    step_size = int(T / num_images)

    noise_schedule = LinearNoiseSchedule(T)

    for i in range(0, T, step_size):
        print(i)
        t = th.tensor([i], dtype=th.int64)
        plt.subplot(1, num_images + 1, (i // step_size) + 1)
        noisy_batch, noise = noise_schedule.diffuse(img_batch, t)
        plt.imshow(dl._to_pil_image(noisy_batch[0]))

    plt.show()
