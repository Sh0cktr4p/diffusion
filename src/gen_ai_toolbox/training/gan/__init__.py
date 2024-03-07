from omegaconf import OmegaConf

from gen_ai_toolbox.models import from_config as model_from_config

from .wgan_gp_model import WGANGPModel


def from_config(config: OmegaConf):
    assert config.type == "GAN"
    assert hasattr(config, "generator_model") and \
        config.generator_model is not None
    assert hasattr(config, "critic_model") and \
        config.critic_model is not None

    generator_model = model_from_config(config.generator_model)
    critic_model = model_from_config(config.critic_model)

    return WGANGPModel(
        generator_model=generator_model,
        critic_model=critic_model,
        z_dim=config.generator_model.params.z_dim,
    ).to(config.device)
