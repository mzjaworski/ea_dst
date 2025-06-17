from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from itertools import cycle
import network
import torch

def soft_top_k(data: torch.Tensor, select: int = 1, decay_rate: float = 1.5) -> torch.BoolTensor:
    data_count = data.numel()

    indices = torch.arange(data_count, device=data.device, dtype=torch.float32)
    weights = torch.exp(-decay_rate * indices)
    probs = weights / weights.sum()

    selected = torch.multinomial(probs, num_samples=min(select, data_count), replacement=False)

    mask = torch.zeros(data_count, dtype=torch.bool, device=data.device)
    mask[selected] = True

    return mask



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




class Pruning(ABC):

    @abstractmethod
    def apply(self, layer: network.MaskedLayer) -> (int, network.MaskedLayer): #(pruned_count, pruned_layer)
        pass

class NullPruning(Pruning):

    def apply(self, layer: network.MaskedLayer) -> (int, network.MaskedLayer):
        return 0, layer

class MagnitudePruning(Pruning):
    def __init__(self, pruned_fraction: float = 0.1, decay_rate: float = 1.5):
        if pruned_fraction < 0.0 or pruned_fraction > 1.0:
            raise ValueError("pruned_fraction must be in [0.0, 1.0]")

        self.pruned_fraction = pruned_fraction
        self.decay_rate = decay_rate

    def apply(self, layer: network.MaskedLayer) -> (int, network.MaskedLayer):
        weight = layer.layer.weight.data
        mask = layer.mask.data

        flat_weight = weight.view(-1)
        flat_mask = mask.view(-1)

        # identify active weights
        active_neurons = (flat_mask == 1).nonzero(as_tuple=True)[0]
        total_active = active_neurons.numel()
        if total_active == 0:
            return 0, layer

        # sort descending by magnitude
        magnitudes, order = torch.sort(flat_weight[active_neurons].abs(), descending=True)
        sorted_idx = active_neurons[order]

        # soft-top-k selection on sorted magnitudes
        sel_mask = soft_top_k(magnitudes, select=int(self.pruned_fraction * total_active), decay_rate=self.decay_rate)

        # map mask back to absolute indices
        to_prune = sorted_idx[sel_mask]
        num_pruned = to_prune.numel()
        if num_pruned > 0:
            flat_mask[to_prune] = 0
            layer.mask.data = flat_mask.view_as(mask)

        return num_pruned, layer




class Regrowth(ABC):

    @abstractmethod
    def apply(self, pruned_count: int, layer: network.MaskedLayer) -> network.MaskedLayer: # regrown_layer
        pass

class NullRegrowth(Regrowth):

    def apply(self, pruned_count: int, layer: network.MaskedLayer) -> network.MaskedLayer:
        return layer

class UniformRegrowth(Regrowth):
    def apply(self, pruned_count: int, layer: network.MaskedLayer) -> network.MaskedLayer:
        weight = layer.layer.weight.data
        mask = layer.mask.data

        flat_mask = mask.view(-1)

        inactive_idx = (flat_mask == 0).nonzero(as_tuple=True)[0]
        if inactive_idx.numel() == 0:
            return layer

        # regrow connections with uniform probability
        perm = torch.randperm(inactive_idx.size(0), device=inactive_idx.device)
        grow = inactive_idx[perm[:pruned_count]]
        flat_mask[grow] = True

        flat_weight = weight.view(-1)
        new_w = flat_weight[flat_mask.to(torch.bool)]
        norm = new_w.norm() if new_w.numel() else 1.0
        std = norm / (new_w.numel() ** 0.5)
        torch.nn.init.normal_(flat_weight[grow], std=std)

        layer.mask.data = flat_mask.view_as(mask)
        return layer

def update_mask(model: torch.nn.Module, warmup_strategy: Warmup, prunning_strategy: Pruning, regrowth_strategy: Regrowth) -> torch.nn.Module:
    model = warmup_strategy.apply(model)

    for layer in model.model:
        if not isinstance(layer, network.MaskedLayer):
            continue

        pruned_count, pruned_layer = prunning_strategy.apply(layer)
        regrowth_strategy.apply(pruned_count, pruned_layer)
        # not assigning back should be safe as we deal with references

    return model
