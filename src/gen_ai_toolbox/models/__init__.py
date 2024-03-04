import torch.nn as nn

from omegaconf import OmegaConf

from . import diffusion


MODELS = {
    "SimpleUNet": diffusion.SimpleUNet,
    "AttentionUNet": diffusion.AttentionUNet,
}


def from_config(model_config: OmegaConf) -> nn.Module:
    if model_config.name not in MODELS:
        raise ValueError(
            f"Unknown model {model_config.name}."
            f"Available models: {list(MODELS.keys())}"
        )
    return MODELS[model_config.name](**model_config.params)
