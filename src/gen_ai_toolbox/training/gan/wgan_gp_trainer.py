import os

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gen_ai_toolbox.datasets import DatasetManager


class WGANGPTrainer:
    def __init__(
        self,
        generator_model: nn.Module,
        critic_model: nn.Module,
        generator_optimizer: th.optim.Optimizer,
        critic_optimizer: th.optim.Optimizer,
        n_critic_updates: int,
        gp_lambda: float,
        z_dim: int,
        dataset_manager: DatasetManager,
        device: th.device,
    ):
        self.generator_model = generator_model.to(device)
        self.critic_model = critic_model.to(device)
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer
        self.n_critic_updates = n_critic_updates
        self.gp_lambda = gp_lambda
        self.z_dim = z_dim
        self.dataset_manager = dataset_manager
        self.device = device

    def train(
        self,
        n_epochs: int,
        generator_model_save_path: str | None = None,
        critic_model_save_path: str | None = None,
    ):
        if generator_model_save_path is not None:
            assert os.path.exists(
                os.path.dirname(generator_model_save_path)
            ), "Model save path must exist"
            assert generator_model_save_path.endswith(".pt"), \
                "Model save path must end with .pt"

        if critic_model_save_path is not None:
            assert os.path.exists(
                os.path.dirname(critic_model_save_path)
            ), "Model save path must exist"
            assert critic_model_save_path.endswith(".pt"), \
                "Model save path must end with .pt"

        for epoch in range(n_epochs):
            for x, _ in self.dataset_manager.loader:
                for _ in range(self.n_critic_updates):
                    critic_loss = self._get_critic_loss(x.to(self.device))
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                generator_loss = self._get_generator_loss(x.shape[0])
                self.generator_optimizer.zero_grad()
                generator_loss.backward()
                self.generator_optimizer.step()

            if epoch % 1 == 0:
                print(f"Epoch {epoch}:")
                print(f"Generator loss: {generator_loss.item()}")
                print(f"Critic loss: {critic_loss.item()}")
                print()

            if generator_model_save_path is not None:
                th.save(
                    self.generator_model.state_dict(),
                    generator_model_save_path
                )

            if critic_model_save_path is not None:
                th.save(
                    self.critic_model.state_dict(),
                    critic_model_save_path
                )

    def _gradient_penalty(
        self,
        x_true: th.Tensor,
        x_fake: th.Tensor
    ) -> th.Tensor:
        b, c, h, w = x_true.shape
        epsilon = th.rand(
            (b, 1, 1, 1),
            device=self.device
        ).repeat(1, c, h, w)
        x_mix = epsilon * x_true + (1 - epsilon) * x_fake

        mixed_scores = self.critic_model(x_mix)
        gradients = th.autograd.grad(
            inputs=x_mix,
            outputs=mixed_scores,
            grad_outputs=th.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0].view(b, -1)
        grad_norm = gradients.norm(2, dim=1)
        grad_pen = F.mse_loss(grad_norm, 1)
        return grad_pen

    def _generate_fake_batch(self, batch_size: int) -> th.Tensor:
        z = th.randn((batch_size, self.z_dim), device=self.device)
        x_fake = self.generator_model(z)
        return x_fake

    def _get_generator_loss(self, batch_size: int) -> th.Tensor:
        x_fake = self._generate_fake_batch(batch_size)
        generator_loss = -th.mean(self.critic_model(x_fake))
        return generator_loss

    def _get_critic_loss(self, x_true: th.Tensor) -> th.Tensor:
        x_fake = self._generate_fake_batch(x_true.shape[0])
        y_true = th.mean(self.critic_model(x_true))
        y_fake = th.mean(self.critic_model(x_fake))
        grad_pen = self._gradient_penalty(x_true, x_fake)
        critic_loss = y_fake - y_true + grad_pen * self.gp_lambda
        return critic_loss

    @staticmethod
    def initialize_weights(model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
