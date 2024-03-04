from omegaconf import OmegaConf

from . import noise_schedule

from gen_ai_toolbox.models import from_config as model_from_config
from gen_ai_toolbox.datasets import from_config as dataset_from_config

from .diffusion_model import DiffusionModel


NOISE_SCHEDULES = {
    "Linear": noise_schedule.LinearNoiseSchedule,
    "Cosine": noise_schedule.CosineNoiseSchedule,
}


def noise_schedule_from_config(noise_schedule_config: OmegaConf):
    if noise_schedule_config.id not in NOISE_SCHEDULES:
        raise ValueError(
            f"Unknown noise schedule {noise_schedule_config.id}."
            f"Available noise schedules: {list(NOISE_SCHEDULES.keys())}"
        )
    return NOISE_SCHEDULES[noise_schedule_config.id](
        **noise_schedule_config.params
    )


def from_config(config: OmegaConf):
    assert config.type == "Diffusion"
    assert hasattr(config, "noise_schedule") and \
        config.noise_schedule is not None
    assert hasattr(config, "model") and config.model is not None

    img_size = None
    if hasattr(config, "dataset") and config.dataset is not None:
        img_size = dataset_from_config(config.dataset).img_size
    else:
        # Allow specifying an image size directly when not supplying a dataset
        assert hasattr(config, "image_size") and config.image_size is not None
        img_size = config.image_size

    noise_schedule = noise_schedule_from_config(config.noise_schedule)
    model = model_from_config(config.model)

    return DiffusionModel(
        model=model,
        noise_schedule=noise_schedule,
        image_size=img_size,
        image_channels=config.model.image_channels,
    ).to(config.device)
