from abc import ABC, abstractmethod
from typing import Tuple
import torch

import src.utilities
import src.network.detail
import src.rules.decay

class Pruning(ABC):

    @abstractmethod
    def apply(self, layer: src.network.detail.MaskedLayer) -> Tuple[int, src.network.detail.MaskedLayer]: # (pruned_count, pruned_layer)
        pass

    @abstractmethod
    def pruned_fraction(self) -> float:
        pass

    @abstractmethod
    def decay_pruned_fraction(self, strategy: src.rules.decay.DecayStrategy = src.rules.decay.NullDecay()) -> None:
        pass

class NullPruning(Pruning):

    def apply(self, layer: src.network.detail.MaskedLayer) -> Tuple[int, src.network.detail.MaskedLayer]:
        return 0, layer

    def pruned_fraction(self) -> float:
        return 0.0

    def decay_pruned_fraction(self, strategy: src.rules.decay.DecayStrategy = src.rules.decay.NullDecay()) -> None:
        pass

class DirectedFlowPruning(Pruning):

    def __init__(self, pruned_fraction: float, decay_rate: float):
        if pruned_fraction < 0.0 or pruned_fraction > 1.0:
            raise ValueError("pruned_fraction must be in [0.0, 1.0]")

        self._pruned_fraction = pruned_fraction
        self.decay_rate = decay_rate

    def apply(self, layer):

        forward_flow = layer.forward_flow.data
        backward_flow = layer.backward_flow.data

        original_mask = layer.mask.data
        flat_mask = original_mask.view(-1)

        # identify active weights
        active_indices = (flat_mask == 1).nonzero(as_tuple=True)[0]
        total_active = active_indices.numel()
        if total_active == 0:
            return 0, layer

        # identify number of weights to remove
        pruned_count = max(int(self._pruned_fraction * total_active), 1)

        # compute normalized flow
        flat_forward_flow = forward_flow.view(-1)[active_indices]
        flat_backward_flow = backward_flow.view(-1)[active_indices]

        flat_f_norm = flat_forward_flow / (flat_forward_flow.max() + 1e-12)
        flat_b_norm = flat_backward_flow / (flat_backward_flow.max() + 1e-12)

        # select weights to remove
        scores = flat_f_norm * flat_b_norm
        magnitudes, order = torch.sort(scores, descending=False)
        sel_mask = src.utilities.soft_top_k(magnitudes, select=pruned_count, decay_rate=self.decay_rate)

        # retrieve their indices
        sorted_idx = active_indices[order]
        to_prune = sorted_idx[sel_mask]
        num_pruned = to_prune.numel()

        # remove selected weights
        flat_mask[to_prune] = 0
        layer.mask.data = flat_mask.view_as(original_mask)

        return num_pruned, layer

    def pruned_fraction(self) -> float:
        return self._pruned_fraction

    def decay_pruned_fraction(self, strategy: src.rules.decay.DecayStrategy = src.rules.decay.NullDecay()) -> None:
        self._pruned_fraction = strategy.decay(self._pruned_fraction)

class MagnitudePruning(Pruning):

    def __init__(self, pruned_fraction: float, decay_rate: float):
        if pruned_fraction < 0.0 or pruned_fraction > 1.0:
            raise ValueError("pruned_fraction must be in [0.0, 1.0]")

        self._pruned_fraction = pruned_fraction
        self.decay_rate = decay_rate

    def apply(self, layer: src.network.detail.MaskedLayer) -> Tuple[int, src.network.detail.MaskedLayer]:
        weight = layer.layer.weight.data
        original_mask = layer.mask.data

        flat_weight = weight.view(-1)
        flat_mask = original_mask.view(-1)

        # identify active weights
        active_neurons = (flat_mask == 1).nonzero(as_tuple=True)[0]
        total_active = active_neurons.numel()
        if total_active == 0:
            return 0, layer

        # identify number of weights to remove
        pruned_count = max(int(self._pruned_fraction * total_active), 1)

        # select weights to remove
        mags = flat_weight[active_neurons].abs()
        magnitudes, order = torch.sort(mags, descending=False)
        sel_mask = src.utilities.soft_top_k(magnitudes, select=pruned_count, decay_rate=self.decay_rate)

        # retrieve their indices
        sorted_idx = active_neurons[order]
        to_prune = sorted_idx[sel_mask]
        num_pruned = to_prune.numel()

        # remove selected weights
        flat_mask[to_prune] = 0
        layer.mask.data = flat_mask.view_as(original_mask)

        return num_pruned, layer

    def pruned_fraction(self) -> float:
        return self._pruned_fraction

    def decay_pruned_fraction(self, strategy: src.rules.decay.DecayStrategy = src.rules.decay.NullDecay()) -> None:
        self._pruned_fraction = strategy.decay(self._pruned_fraction)

class UniformPruning(Pruning):

    def __init__(self, pruned_fraction: float):
        if pruned_fraction < 0.0 or pruned_fraction > 1.0:
            raise ValueError("pruned_fraction must be in [0.0, 1.0]")

        self._pruned_fraction = pruned_fraction

    def apply(self, layer: src.network.detail.MaskedLayer) -> Tuple[int, src.network.detail.MaskedLayer]:
        original_mask = layer.mask.data
        flat_mask = original_mask.view(-1)

        # identify active weights
        active_indices = (flat_mask == 1).nonzero(as_tuple=True)[0]
        total_active = active_indices.numel()
        if total_active == 0:
            return 0, layer

        # identify number of weights to remove
        num_pruned = max(int(self._pruned_fraction * total_active), 1)

        if num_pruned == 0:
            return 0, layer

        # randomly select indices to prune
        prune_indices = active_indices[torch.randperm(total_active)[:num_pruned]]

        # prune the weights
        flat_mask[prune_indices] = 0
        layer.mask.data = flat_mask.view_as(original_mask)

        return num_pruned, layer

    def pruned_fraction(self) -> float:
        return self._pruned_fraction

    def decay_pruned_fraction(self, strategy: src.rules.decay.DecayStrategy = src.rules.decay.NullDecay()) -> None:
        self._pruned_fraction = strategy.decay(self._pruned_fraction)

class RSensitivityPruning(Pruning):

    def __init__(self, pruned_fraction: float, decay_rate: float):
        if pruned_fraction < 0.0 or pruned_fraction > 1.0:
            raise ValueError("pruned_fraction must be in [0.0, 1.0]")

        self._pruned_fraction = pruned_fraction
        self.decay_rate = decay_rate

    def apply(self, layer: src.network.detail.MaskedLayer) -> Tuple[int, src.network.detail.MaskedLayer]:
        weight = layer.layer.weight
        original_mask = layer.mask.data

        if weight.grad is None:
            raise RuntimeError("gradient of the weights is required for RSensitivity pruning! enable GradientWarmup")

        grad = weight.grad

        flat_weight = weight.data.view(-1)
        flat_grad = grad.view(-1)
        flat_mask = original_mask.view(-1)

        # identify active weights
        active_neurons = (flat_mask == 1).nonzero(as_tuple=True)[0]
        total_active = active_neurons.numel()
        if total_active == 0:
            return 0, layer

        # identify number of weights to remove
        pruned_count = max(int(self._pruned_fraction * total_active), 1)

        # compute sensitivity scores: |w| / |grad|
        abs_weight = flat_weight[active_neurons].abs()
        abs_grad = flat_grad[active_neurons].abs()
        sensitivity = abs_weight / (abs_grad + 1e-8)

        # select weights to remove
        sorted_sensitivity, order = torch.sort(sensitivity, descending=False)
        sel_mask = src.utilities.soft_top_k(sorted_sensitivity, select=pruned_count, decay_rate=self.decay_rate)

        # retrieve their indices
        sorted_indices = active_neurons[order]
        prune_indices = sorted_indices[sel_mask]
        num_pruned = prune_indices.numel()

        # remove selected weights
        flat_mask[prune_indices] = 0
        layer.mask.data = flat_mask.view_as(original_mask)

        return num_pruned, layer

    def pruned_fraction(self) -> float:
        return self._pruned_fraction

    def decay_pruned_fraction(self, strategy: src.rules.decay.DecayStrategy = src.rules.decay.NullDecay()) -> None:
        self._pruned_fraction = strategy.decay(self._pruned_fraction)
