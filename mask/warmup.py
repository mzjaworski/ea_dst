from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from itertools import cycle
import torch

class Warmup(ABC):

    @abstractmethod
    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        pass

class NullWarmup(Warmup):

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

class GradientWarmup(Warmup): # used eg for gradient regrowth

    def __init__(self, dataset: DataLoader, minibatch_budget: int = 256):
        self.dataset = dataset
        self.minibatch_budget = minibatch_budget

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        model.train()

        # zero all gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        # accumulate gradients
        current_device = next(model.parameters()).device
        for batch_index, (inputs, targets) in enumerate(cycle(self.dataset)):
            if batch_index >= self.minibatch_budget:
                break

            outputs = model(inputs.to(current_device))
            loss = torch.nn.functional.cross_entropy(outputs, targets.to(current_device))
            loss.backward()

        model.eval()
        return model
