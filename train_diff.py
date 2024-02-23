import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dataset_manager import (
    DatasetManager,
    ImageFolderDatasetManager,
)
from forward_process import (
    ForwardProcess,
    LinScForwardProcess,
    CosScForwardProcess,
)
from unet_model import SimpleUNet
import matplotlib.pyplot as plt


def train(
    dm: DatasetManager,
    fp: ForwardProcess,
    model: nn.Module,
    n_epochs: int,
    device: str = "cuda",
):
    optimizer = th.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        for step, (x, _) in enumerate(dm.loader):
            optimizer.zero_grad()
            t = th.randint(0, fp.T, (x.shape[0],), dtype=th.long)
            loss = fp.get_loss(model, x.to(device), t.to(device))
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss {loss.item()}")
            imgs = fp.sample(model, img_size=64)
            plt.figure(figsize=(15, 15))
            plt.axis("off")
            num_images = 10
            step_size = int(fp.T / num_images)
            for i, img in enumerate(imgs[::step_size]):
                plt.subplot(1, num_images, i / step_size + 1)
                ImageFolderDatasetManager._to_pil_image(img[0].cpu())
            plt.show()


if __name__ == '__main__':
    device = "cuda"
    model = SimpleUNet().to(device)
    dm = ImageFolderDatasetManager(
        root="data/pokemon",
        image_size=64,
        batch_size=16
    )
    fp = CosScForwardProcess(200, 0.0001).to(device)
    train(dm, fp, model, 100, device)
