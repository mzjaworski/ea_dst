from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from itertools import cycle
import torch

class Warmup(ABC):

    @abstractmethod
    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        pass

    @abstractmethod
    def amortized_flops(self, dense_flop_estimate: float, sparse_flop_estimate: float, minibatch_budget: int) -> float:
        pass

    @abstractmethod
    def constant_cost(self) -> float:
        pass

    @abstractmethod
    def update_factor(self, factor: float) -> None:
        pass

class NullWarmup(Warmup):

    def __init__(self, constant_cost: int = 0, **_ignored):
        self.dense_flop_estimate = None
        self._constant_cost = constant_cost

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def amortized_flops(self, dense_flop_estimate: float, sparse_flop_estimate: float, minibatch_budget: int) -> float:
        self.dense_flop_estimate = dense_flop_estimate
        return 3.0 * sparse_flop_estimate # follows Rigging the Lottery: Making All Tickets Winners logic

    def constant_cost(self) -> float:
        return self._constant_cost * self.dense_flop_estimate

    def update_factor(self, factor: float) -> None:
        pass

class GradientWarmup(Warmup): # used eg for gradient regrowth

    def __init__(self, dataset: DataLoader, minibatch_budget: int = 256, constant_cost: int = 0):
        self.dense_flop_estimate = None
        self.dataset = dataset
        self.minibatch_budget = minibatch_budget
        self._constant_cost = constant_cost
        self.factor = None

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
        self.dense_flop_estimate = dense_flop_estimate
        warmup_samples = self.minibatch_budget * self.dataset.batch_size
        train_samples = minibatch_budget * self.dataset.batch_size
        # follows (adjusted) Rigging the Lottery: Making All Tickets Winners logic
        return ((3.0 * sparse_flop_estimate * train_samples) + (2.0 * sparse_flop_estimate + dense_flop_estimate) * warmup_samples * self.factor) / train_samples # we divide by 'train_samples' and not 'train_samples + warmup_samples' as externally we only track 'train_samples' hence we need to slightly inflate cost to account for it

    def constant_cost(self) -> float:
        return self._constant_cost * self.dense_flop_estimate

    def update_factor(self, factor: float) -> None:
        self.factor = factor