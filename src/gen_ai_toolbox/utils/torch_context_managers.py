from contextlib import contextmanager

import torch.nn as nn


@contextmanager
def eval(model: nn.Module):
    training_mode = model.training
    try:
        model.eval()
        yield
    finally:
        if training_mode:
            model.train()


@contextmanager
def train(model: nn.Module):
    training_mode = model.training
    try:
        model.train()
        yield
    finally:
        if not training_mode:
            model.eval()
