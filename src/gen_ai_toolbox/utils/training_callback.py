from typing import Callable, List
import os
import shutil
from datetime import datetime

import torch.nn as nn


class TrainingCallback():
    def on_training_begin(self, model: nn.Module) -> None:
        pass

    def on_training_end(self, model: nn.Module) -> None:
        pass

    def on_epoch_begin(self, epoch: int, model: nn.Module) -> None:
        pass

    def on_epoch_end(self, epoch: int, model: nn.Module) -> None:
        pass


class FnTrainingCallback(TrainingCallback):
    def __init__(
        self,
        on_training_begin: Callable[..., None] | None = None,
        on_training_end: Callable[..., None] | None = None,
        on_epoch_begin: Callable[..., None] | None = None,
        on_epoch_end: Callable[..., None] | None = None,
    ):
        self._on_training_begin = on_training_begin or (lambda **_: None)
        self._on_training_end = on_training_end or (lambda **_: None)
        self._on_epoch_begin = on_epoch_begin or (lambda **_: None)
        self._on_epoch_end = on_epoch_end or (lambda **_: None)

    def on_training_begin(self, **kwargs) -> None:
        self._on_training_begin(**kwargs)

    def on_training_end(self, **kwargs) -> None:
        self._on_training_end(**kwargs)

    def on_epoch_begin(self, **kwargs) -> None:
        self._on_epoch_begin(**kwargs)

    def on_epoch_end(self, **kwargs) -> None:
        self._on_epoch_end(**kwargs)


class TrainingCallbackList(TrainingCallback):
    def __init__(self, callbacks: List[TrainingCallback]):
        self.callbacks = callbacks

    def on_training_begin(self, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_training_begin(**kwargs)

    def on_training_end(self, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_training_end(**kwargs)

    def on_epoch_begin(self, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(**kwargs)

    def on_epoch_end(self, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(**kwargs)


class InfoCallback(TrainingCallback):
    def __init__(self):
        self.training_start_time = None
        self.start_epoch = None

    def on_training_begin(
        self,
        model: nn.Module,
        epoch: int,
        **kwargs
    ) -> None:
        self.training_start_time = datetime.now()
        self.start_epoch = epoch
        print("=====================================")
        print("Starting training...")
        print("Model:")
        print(model)
        print("Model has {} parameters.".format(
            sum(p.numel() for p in model.parameters())
        ))
        print("=====================================")

    def on_training_end(self, **kwargs) -> None:
        print("=====================================")
        print("Training finished.")
        print(f"Time elapsed: {datetime.now() - self.training_start_time}")
        print("=====================================")

    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        print(f"Epoch {epoch}:")

    def on_epoch_end(
        self,
        epoch: int,
        training_loss: float | None,
        validation_loss: float | None,
        **kwargs,
    ) -> None:
        time_delta = datetime.now() - self.training_start_time
        print("=====================================")
        print(f"Epoch {epoch} finished.")
        print(f"Time elapsed: {time_delta}")
        print(f"Time per epoch: {(time_delta) / (epoch - self.start_epoch)}")
        print(f"Training loss: {training_loss}")
        print(f"Validation loss: {validation_loss}")
        print("=====================================")


class SaveArtifactCallback(TrainingCallback):
    def __init__(
        self,
        artifacts_base_path: str,
        callbacks: List[Callable[[str], None]],
        save_freq: int = 1,
    ):
        assert os.path.exists(
            os.path.dirname(artifacts_base_path)
        ), "Model save path must exist"

        if os.path.exists(artifacts_base_path):
            print(f"Model save path {artifacts_base_path} already exists.")
            print("Override? (y/[n])")
            if input() not in ["y", "Y"]:
                raise ValueError("Model save path already exists")
            else:
                shutil.rmtree(artifacts_base_path)

        os.mkdir(artifacts_base_path)

        self.artifacts_base_path = artifacts_base_path
        self.save_freq = save_freq
        self.callbacks = callbacks

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        if epoch % self.save_freq == 0:
            path = self.get_epoch_path(self.artifacts_base_path, epoch)
            os.mkdir(path)
            for callback in self.callbacks:
                callback(path)

    @staticmethod
    def get_epoch_path(artifacts_base_path: str, epoch: int) -> str:
        return os.path.join(artifacts_base_path, f"EP_{epoch}")
