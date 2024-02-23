import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dataset_manager import DatasetManager


class DiffusionTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: th.optim.Optimizer,
        forward_process: nn.Module,
        dataset_manager: DatasetManager,
        device: th.device,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.forward_process = forward_process.to(device)
        self.dataset_manager = dataset_manager
        self.device = device

    def train(self):
        pass

    @th.no_grad()
    def sample_timestep(x: th.Tensor, t: th.Tensor):
        pass

    @th.no_grad()
    def sample(self):
        img_size = self.dataset_manager.img_size