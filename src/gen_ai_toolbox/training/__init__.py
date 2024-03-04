import os

import torch as th

from omegaconf import OmegaConf

from diffusion import from_config as diffusion_from_config
from gen_ai_toolbox.training.generative_model import GenerativeModel
from gen_ai_toolbox.datasets import from_config as dataset_from_config
from gen_ai_toolbox.utils.training_callback import (
    TrainingCallback,
    TrainingCallbackList,
    InfoCallback,
    SaveArtifactCallback,
)


OPTIMIZERS = {
    "Adam": th.optim.Adam,
}

LR_SCHEDULES = {
    "ExponentialLR": th.optim.lr_scheduler.ExponentialLR,
}


def generative_model_from_config(config: OmegaConf) -> GenerativeModel:
    assert hasattr(config, "type") and config.type is not None
    if config.type == "Diffusion":
        model = diffusion_from_config(config)
        if config.training.start_epoch > 0:
            model.load_model_state_dict(
                os.path.join(
                    config.training.artifact_dir,
                    SaveArtifactCallback.get_epoch_path(
                        config.artifact_dir,
                        config.start_epoch,
                    ),
                    "model.pt",
                ),
            )
    else:
        raise ValueError(f"Unknown trainer type {config.type}.")


def optimizer_from_config(
    optimizer_config: OmegaConf,
    model: th.nn.Module,
) -> th.optim.Optimizer:
    if optimizer_config.id not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer {optimizer_config.id}."
            f"Available optimizers: {list(OPTIMIZERS.keys())}"
        )
    return OPTIMIZERS[optimizer_config.id](
        model.parameters(),
        **optimizer_config.params
    )


def lr_schedule_from_config(
    lr_schedule_config: OmegaConf,
    optimizer: th.optim.Optimizer,
):
    if lr_schedule_config.id not in LR_SCHEDULES:
        raise ValueError(
            f"Unknown lr schedule {lr_schedule_config.id}."
            f"Available lr schedules: {list(LR_SCHEDULES.keys())}"
        )
    return LR_SCHEDULES[lr_schedule_config.id](
        optimizer,
        **lr_schedule_config.params
    )


def callback_from_config(
    config: OmegaConf,
    generative_model: GenerativeModel,
) -> TrainingCallback:
    callbacks = []
    if config.training.verbose:
        callbacks.append(InfoCallback())

    if not config.training.debug:
        callbacks.append(SaveArtifactCallback(
            artifacts_base_dir=config.artifact_dir,
            callbacks=[
                # Save config file
                lambda path: config.to_yaml(os.path.join(path, "config.yaml")),
                # Save model file
                lambda path: generative_model.save_model_state_dict(
                    os.path.join(path, "model.pt"),
                ),
                # Save rendered images
                lambda path: generative_model.render_batch(
                    n_rows=4,
                    n_colums=4,
                    path=os.path.join(path, "renders.png"),
                ),
            ],
            save_freq=config.training.save_freq,
        ))

    return TrainingCallbackList(callbacks)


def train_from_config(
    config: OmegaConf,
    generative_model: GenerativeModel | None = None,
):
    assert hasattr(config, "training") and config.training is not None
    assert hasattr(config, "dataset") and config.dataset is not None
    if generative_model is None:
        generative_model = generative_model_from_config(config)

    callback = None
    if config.verbose:
        pass

    if config.type == "Diffusion":
        optimizer = optimizer_from_config(
            config.training.optimizer,
            generative_model.model
        )
        lr_schedule = None
        if (
            hasattr(config.training, "lr_schedule") and
            config.training.lr_schedule is not None
        ):
            lr_schedule = lr_schedule_from_config(
                config.training.lr_schedule,
                optimizer
            )
        dataset_manager = dataset_from_config(config.dataset)

        generative_model.train_model(
            n_epochs=config.training.n_epochs,
            optimizer=optimizer,
            dataset_manager=dataset_manager,
            lr_schedule=lr_schedule,
            callback=callback,
            start_epoch=config.initial_epoch,
        )
    else:
        raise ValueError(f"Unknown trainer type {config.type}.")
