import torch as th
import torch.nn as nn

from gen_ai_toolbox.datasets import DatasetManager
from gen_ai_toolbox.utils.training_callback import TrainingCallback


class GenerativeModel(nn.Module):
    def __init__(self):
        super().__init__()

    def train_model(
        self,
        n_epochs: int,
        optimizer: th.optim.Optimizer,
        dataset_manager: DatasetManager,
        lr_schedule: th.optim.lr_scheduler._LRScheduler | None = None,
        callback: TrainingCallback | None = None,
        start_epoch: int = 0,
    ):
        raise NotImplementedError("train_model method not implemented")

    @th.inference_mode()
    def render_batch(
        self,
        n_rows: int,
        n_columns: int,
        path: str | None = None,
    ):
        raise NotImplementedError("render_batch method not implemented")

    def save_model_state_dict(
        self,
        path: str,
    ):
        raise NotImplementedError("save_model method not implemented")

    def load_model_state_dict(
        self,
        path: str,
    ):
        raise NotImplementedError("load_model method not implemented")
