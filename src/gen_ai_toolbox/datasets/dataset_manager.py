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
    def __init__(self, dataset: data.Dataset, batch_size: int, img_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.img_size = img_size
        self.loader = self._get_dataloader(
            dataset=dataset,
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

    def show_images(self, num_samples=20, cols=4):
        plt.figure(figsize=(10, 10))
        for i, img in list(enumerate(self.dataset))[:num_samples]:
            plt.subplot(num_samples // cols + 1, cols, i + 1)
            plt.imshow(img[0].permute(1, 2, 0))
        plt.show()


class ImageFolderDatasetManager(DatasetManager):
    def __init__(self, root: str, image_size: int, batch_size: int):
        dataset = ImageFolderDatasetManager._get_dataset(root, image_size)
        super().__init__(dataset, batch_size, image_size)

    @staticmethod
    def _get_dataset(root: str, image_size: int):
        return torchvision.datasets.ImageFolder(
            root=root,
            transform=ImageFolderDatasetManager._get_transform(image_size),
            loader=rgba_to_rgb_img_loader,
        )


class MNISTDatasetManager(DatasetManager):
    def __init__(self, image_size: int, batch_size: int):
        dataset = MNISTDatasetManager._get_dataset(image_size)
        super().__init__(dataset, batch_size, image_size)

    @staticmethod
    def _get_dataset(image_size: int):
        return torchvision.datasets.MNIST(
            root="./data",
            transform=MNISTDatasetManager._get_transform(image_size),
            download=True,
        )


class StanfordCarsDatasetManager(DatasetManager):
    def __init__(self, image_size: int, batch_size: int):
        dataset = StanfordCarsDatasetManager._get_dataset(image_size)
        super().__init__(dataset, batch_size, image_size)

    @staticmethod
    def _get_dataset(image_size: int):
        return torchvision.datasets.StanfordCars(
            root="data/",
            transform=StanfordCarsDatasetManager._get_transform(image_size),
            download=True,
        )


class CelebADatasetManager(DatasetManager):
    def __init__(self, image_size: int, batch_size: int):
        dataset = CelebADatasetManager._get_dataset(image_size)
        super().__init__(dataset, batch_size, image_size)

    @staticmethod
    def _get_dataset(image_size: int):
        return torchvision.datasets.CelebA(
            root="data/",
            transform=CelebADatasetManager._get_transform(image_size),
            download=True,
        )


class ImageNetDatasetManager(DatasetManager):
    def __init__(self, image_size: int, batch_size: int):
        dataset = ImageNetDatasetManager._get_dataset(image_size)
        super().__init__(dataset, batch_size, image_size)

    @staticmethod
    def _get_dataset(image_size: int):
        return torchvision.datasets.ImageNet(
            root="data/",
            transform=ImageNetDatasetManager._get_transform(image_size),
            download=True,
        )


class CIFAR10DatasetManager(DatasetManager):
    def __init__(self, image_size: int, batch_size: int):
        dataset = CIFAR10DatasetManager._get_dataset(image_size)
        super().__init__(dataset, batch_size, image_size)

    @staticmethod
    def _get_dataset(image_size: int):
        return torchvision.datasets.CIFAR10(
            root="data/",
            transform=CIFAR10DatasetManager._get_transform(image_size),
            download=True,
        )


if __name__ == '__main__':
    from forward_process import CosScForwardProcess

    dl = ImageFolderDatasetManager(
        root="data/pokemon",
        image_size=64,
        batch_size=32
    )

    T = 1000

    img_batch = next(iter(dl.loader))[0]
    plt.figure(figsize=(15, 5))
    plt.axis("off")
    num_images = 10
    step_size = int(T / num_images)

    for i in range(0, T, step_size):
        print(i)
        t = th.tensor([i], dtype=th.int64)
        plt.subplot(1, num_images + 1, (i // step_size) + 1)
        noisy_batch, noise = CosScForwardProcess(T)(img_batch, t)
        plt.imshow(dl._to_pil_image(noisy_batch[0]))

    plt.show()
