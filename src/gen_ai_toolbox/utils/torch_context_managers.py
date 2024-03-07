from contextlib import contextmanager

import torch.nn as nn


@contextmanager
def eval(*models: nn.Module):
    training_modes = [model.training for model in models]
    try:
        for model in models:
            model.eval()
        yield
    finally:
        for training_mode, model in zip(training_modes, models):
            if training_mode:
                model.train()


@contextmanager
def train(*models: nn.Module):
    training_modes = [model.training for model in models]
    try:
        for model in models:
            model.train()
        yield
    finally:
        for training_mode, model in zip(training_modes, models):
            if not training_mode:
                model.eval()
