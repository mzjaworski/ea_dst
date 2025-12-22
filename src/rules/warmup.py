from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from itertools import cycle
import torch

class Warmup(ABC):

    @abstractmethod
    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        pass

    def amortized_flops(self, dense_flop_estimate: float, sparse_flop_estimate: float, minibatch_budget: int) -> float:
        return 3.0 * sparse_flop_estimate # follows Rigging the Lottery: Making All Tickets Winners logic

class NullWarmup(Warmup):

    def __init__(self, **_ignored):
        pass

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

    def amortized_flops(self, dense_flop_estimate: float, sparse_flop_estimate: float, minibatch_budget: int) -> float:
        warmup_samples = self.minibatch_budget * self.dataset.batch_size
        train_samples = minibatch_budget * self.dataset.batch_size
        return ((3.0 * sparse_flop_estimate * train_samples) + (2.0 * sparse_flop_estimate + dense_flop_estimate) * warmup_samples) / (train_samples + warmup_samples) # follows (adjusted) Rigging the Lottery: Making All Tickets Winners logic
