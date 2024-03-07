import torch.nn as nn

from omegaconf import OmegaConf

from . import diffusion
from . import gan


MODELS = {
    "SimpleUNet": diffusion.SimpleUNet,
    "AttentionUNet": diffusion.AttentionUNet,
    "DCGenerator": gan.Generator,
    "DCDiscriminator": gan.Discriminator,
}


def from_config(model_config: OmegaConf) -> nn.Module:
    if model_config.id not in MODELS:
        raise ValueError(
            f"Unknown model {model_config.id}."
            f"Available models: {list(MODELS.keys())}"
        )
    return MODELS[model_config.id](**model_config.params)
