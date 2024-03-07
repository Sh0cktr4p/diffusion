import torch as th
import torch.nn as nn

from gen_ai_toolbox import training
from gen_ai_toolbox.datasets import DatasetManager
from gen_ai_toolbox.utils.training_callback import TrainingCallback
from gen_ai_toolbox.utils.torch_context_managers import eval, train


class WGANGPModel(training.GenerativeModel):
    def __init__(
        self,
        generator_model: nn.Module,
        critic_model: nn.Module,
        z_dim: int,
    ):
        super().__init__()
        self.generator_model = generator_model
        self.critic_model = critic_model
        self.z_dim = z_dim

    @property
    def device(self) -> th.device:
        return next(self.generator_model.parameters()).device

    def train_critic_batch(
        self,
        x: th.Tensor,
        optimizer: th.optim.Optimizer,
        n_critic_updates: int,
        gp_lambda: float,
    ):
        loss = 0
        with train(self.critic_model):
            for _ in range(n_critic_updates):
                optimizer.zero_grad()  # TODO check order
                critic_loss = self.get_critic_loss(x, gp_lambda)
                critic_loss.backward()
                optimizer.step()
                loss += critic_loss.item()

        return loss / n_critic_updates

    def train_generator_batch(
        self,
        batch_size: int,
        optimizer: th.optim.Optimizer,
    ):
        with train(self.generator_model):
            optimizer.zero_grad()  # TODO check order
            generator_loss = self.get_generator_loss(batch_size)
            generator_loss.backward()
            optimizer.step()

        return generator_loss.item()

    def train_epoch(
        self,
        epoch: int,
        n_critic_updates: int,
        gp_lambda: float,
        generator_optimizer: th.optim.Optimizer,
        critic_optimizer: th.optim.Optimizer,
        dataset_manager: DatasetManager,
        gen_lr_scheduler: th.optim.lr_scheduler._LRScheduler | None = None,
        crit_lr_scheduler: th.optim.lr_scheduler._LRScheduler | None = None,
        callback: TrainingCallback | None = None,
    ):
        if callback is None:
            callback = TrainingCallback()

        callback.on_epoch_begin(
            epoch=epoch,
            model=self.generator_model,
            critic_model=self.critic_model,
        )

        with train(self.generator_model, self.critic_model):
            for i, (x, _) in enumerate(dataset_manager.train_loader):
                self.train_critic_batch(
                    x=x,
                    optimizer=critic_optimizer,
                    n_critic_updates=n_critic_updates,
                    gp_lambda=gp_lambda,
                )

                self.train_generator_batch(
                    batch_size=x.shape[0],
                    optimizer=generator_optimizer,
                )

        with eval(
            self.generator_model,
            self.critic_model
        ), th.inference_mode():
            x_train = next(iter(dataset_manager.train_loader))[0]
            x_val = next(iter(dataset_manager.val_loader))[0]
            train_loss = self.get_critic_loss(x_train, gp_lambda=0).item()
            val_loss = self.get_critic_loss(x_val, gp_lambda=0).item()

        callback.on_epoch_end(
            epoch=epoch,
            model=self.generator_model,
            critic_model=self.critic_model,
            training_loss=train_loss,
            validation_loss=val_loss,
        )

        if gen_lr_scheduler is not None:
            gen_lr_scheduler.step()

        if crit_lr_scheduler is not None:
            crit_lr_scheduler.step()

    def train_model(
        self,
        n_epochs: int,
        n_critic_updates: int,
        gp_lambda: float,
        generator_optimizer: th.optim.Optimizer,
        critic_optimizer: th.optim.Optimizer,
        dataset_manager: DatasetManager,
        gen_lr_scheduler: th.optim.lr_scheduler._LRScheduler | None = None,
        crit_lr_scheduler: th.optim.lr_scheduler._LRScheduler | None = None,
        callback: TrainingCallback | None = None,
        start_epoch: int = 0
    ):
        if callback is None:
            callback = TrainingCallback()

        callback.on_training_begin(
            model=self.generator_model,
            critic_model=self.critic_model,
            epoch=start_epoch
        )

        with train(self.generator_model, self.critic_model):
            for epoch in range(n_epochs):
                self.train_epoch(
                    epoch=epoch + start_epoch,
                    n_critic_updates=n_critic_updates,
                    gp_lambda=gp_lambda,
                    generator_optimizer=generator_optimizer,
                    critic_optimizer=critic_optimizer,
                    dataset_manager=dataset_manager,
                    gen_lr_scheduler=gen_lr_scheduler,
                    crit_lr_scheduler=crit_lr_scheduler,
                    callback=callback,
                )

        callback.on_training_end(
            model=self.generator_model,
            critic_model=self.critic_model
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
        )
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
        grad_pen = th.mean((grad_norm - 1) ** 2)
        return grad_pen

    def _generate_fake_batch(
        self,
        batch_size: int,
        z: th.Tensor | None = None
    ) -> th.Tensor:
        if z is None:
            z = th.randn((batch_size, self.z_dim, 1, 1), device=self.device)
        x_fake = self.generator_model(z)
        return x_fake

    def get_generator_loss(self, batch_size: int) -> th.Tensor:
        x_fake = self._generate_fake_batch(batch_size)
        generator_loss = -th.mean(self.critic_model(x_fake))
        return generator_loss

    def get_critic_loss(
        self,
        x_true: th.Tensor,
        gp_lambda: float
    ) -> th.Tensor:
        x_true = x_true.to(self.device)
        x_fake = self._generate_fake_batch(x_true.shape[0])
        y_true = th.mean(self.critic_model(x_true))
        y_fake = th.mean(self.critic_model(x_fake))
        if gp_lambda > 0:
            grad_pen = self._gradient_penalty(x_true, x_fake)
        else:
            grad_pen = 0  # save computation
        critic_loss = y_fake - y_true + grad_pen * gp_lambda
        return critic_loss

    def initialize_weights(self):
        self.initialize_model_weights(self.generator_model)
        self.initialize_model_weights(self.critic_model)

    @staticmethod
    def initialize_model_weights(model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    @th.inference_mode()
    def render_batch(
        self,
        n_rows: int,
        n_columns: int,
        path: str | None = None,
    ):
        with eval(self.generator_model):
            x_fake = self._generate_fake_batch(n_rows * n_columns)

        DatasetManager.render_batch(
            tensor=x_fake,
            n_rows=n_rows,
            n_columns=n_columns,
            path=path,
        )

    def save_model_state_dict(
        self,
        path: str,
    ):
        th.save(
            {
                "generator": self.generator_model.state_dict(),
                "critic": self.critic_model.state_dict(),
            },
            path
        )

    def load_model_state_dict(
        self,
        path: str,
    ):
        state_dict = th.load(path)
        self.generator_model.load_state_dict(state_dict["generator"])
        self.critic_model.load_state_dict(state_dict["critic"])
