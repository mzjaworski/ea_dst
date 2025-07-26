from abc import ABC, abstractmethod
from typing import Tuple
import torch

import mask.utilities
import mask.network

class Pruning(ABC):

    @abstractmethod
    def apply(self, layer: mask.network.MaskedLayer) -> Tuple[int, mask.network.MaskedLayer]: # (pruned_count, pruned_layer)
        pass

class NullPruning(Pruning):

    def apply(self, layer: mask.network.MaskedLayer) -> Tuple[int, mask.network.MaskedLayer]:
        return 0, layer

class MagnitudePruning(Pruning):

    def __init__(self, pruned_fraction: float, decay_rate: float):
        if pruned_fraction < 0.0 or pruned_fraction > 1.0:
            raise ValueError("pruned_fraction must be in [0.0, 1.0]")

        self.pruned_fraction = pruned_fraction
        self.decay_rate = decay_rate

    def apply(self, layer: mask.network.MaskedLayer) -> Tuple[int, mask.network.MaskedLayer]:
        weight = layer.layer.weight.data
        original_mask = layer.mask.data

        flat_weight = weight.view(-1)
        flat_mask = original_mask.view(-1)

        # identify active weights
        active_neurons = (flat_mask == 1).nonzero(as_tuple=True)[0]
        total_active = active_neurons.numel()
        if total_active == 0:
            return 0, layer

        pruned_count = int(self.pruned_fraction * total_active)
        mags = flat_weight[active_neurons].abs()

        # soft-top-k selection on sorted magnitudes
        magnitudes, order = torch.sort(mags, descending=False)
        sorted_idx = active_neurons[order]
        sel_mask = mask.utilities.soft_top_k(magnitudes, select=pruned_count, decay_rate=self.decay_rate)

        # map mask back to absolute indices
        to_prune = sorted_idx[sel_mask]
        num_pruned = to_prune.numel()
        if num_pruned > 0:
            flat_mask[to_prune] = 0
            layer.mask.data = flat_mask.view_as(original_mask)

        return num_pruned, layer

class UniformPruning(Pruning):

    def __init__(self, pruned_fraction: float):
        if pruned_fraction < 0.0 or pruned_fraction > 1.0:
            raise ValueError("pruned_fraction must be in [0.0, 1.0]")

        self.pruned_fraction = pruned_fraction

    def apply(self, layer: mask.network.MaskedLayer) -> Tuple[int, mask.network.MaskedLayer]:
        weight = layer.layer.weight.data
        original_mask = layer.mask.data

        flat_mask = original_mask.view(-1)

        # identify active weights
        active_indices = (flat_mask == 1).nonzero(as_tuple=True)[0]
        total_active = active_indices.numel()

        if total_active == 0:
            return 0, layer

        # determine how many weights to prune
        num_to_prune = int(self.pruned_fraction * total_active)

        if num_to_prune == 0:
            return 0, layer

        # randomly select indices to prune
        prune_indices = active_indices[torch.randperm(total_active)[:num_to_prune]]

        # prune the weights
        flat_mask[prune_indices] = 0
        layer.mask.data = flat_mask.view_as(original_mask)

        return num_to_prune, layer

class RSensitivityPruning(Pruning):

    def __init__(self, pruned_fraction: float, decay_rate: float):
        if pruned_fraction < 0.0 or pruned_fraction > 1.0:
            raise ValueError("pruned_fraction must be in [0.0, 1.0]")

        self.pruned_fraction = pruned_fraction
        self.decay_rate = decay_rate

    def apply(self, layer: mask.network.MaskedLayer) -> Tuple[int, mask.network.MaskedLayer]:
        weight = layer.layer.weight
        original_mask = layer.mask.data

        if weight.grad is None:
            raise RuntimeError("gradient of the weights is required for RSensitivity pruning! enable GradientWarmup")

        grad = weight.grad
        flat_weight = weight.data.view(-1)
        flat_grad = grad.view(-1)
        flat_mask = original_mask.view(-1)

        # identify active weights
        active_indices = (flat_mask == 1).nonzero(as_tuple=True)[0]
        total_active = active_indices.numel()

        if total_active == 0:
            return 0, layer

        # compute sensitivity scores: |w| / |grad|
        abs_weight = flat_weight[active_indices].abs()
        abs_grad = flat_grad[active_indices].abs()
        sensitivity = abs_weight / (abs_grad + 1e-8)

        pruned_count = int(self.pruned_fraction * total_active)

        # apply soft_top_k selection mask
        sorted_sensitivity, order = torch.sort(sensitivity, descending=True)
        sorted_indices = active_indices[order]
        sel_mask = mask.utilities.soft_top_k(sorted_sensitivity, select=pruned_count, decay_rate=self.decay_rate)

        # map mask back to indices to prune
        prune_indices = sorted_indices[sel_mask]
        num_pruned = prune_indices.numel()

        if num_pruned > 0:
            flat_mask[prune_indices] = 0
            layer.mask.data = flat_mask.view_as(original_mask)

        return num_pruned, layer
