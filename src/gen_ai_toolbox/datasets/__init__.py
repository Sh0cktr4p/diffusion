from functools import partial

from omegaconf import OmegaConf

from dataset_manager import (
    DatasetManager,
    ImageFolderDatasetManager,
    MNISTDatasetManager,
    StanfordCarsDatasetManager,
    CelebADatasetManager,
    ImageNetDatasetManager,
    CIFAR10DatasetManager,
)


DATASETS = {
    "MNIST": MNISTDatasetManager,
    "StanfordCars": StanfordCarsDatasetManager,
    "CelebA": CelebADatasetManager,
    "ImageNet": ImageNetDatasetManager,
    "CIFAR10": CIFAR10DatasetManager,
    "Pokemon": partial(ImageFolderDatasetManager, root="data/pokemon"),
    "ImageFolder": ImageFolderDatasetManager,
}


def from_config(dataset_config: OmegaConf) -> DatasetManager:
    if dataset_config.id not in DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset_config.id}."
            f"Available datasets: {list(DATASETS.keys())}"
        )
    return DATASETS[dataset_config.id](**dataset_config.params)
