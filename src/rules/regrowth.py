from abc import ABC, abstractmethod
import torch

import src.utilities
import src.network.detail
import src.rules.weight

class Regrowth(ABC):

    @abstractmethod
    def apply(self, pruned_count: int, layer: src.network.detail.MaskedLayer) -> src.network.detail.MaskedLayer: # regrown_layer
        pass

class NullRegrowth(Regrowth):

    def apply(self, pruned_count: int, layer: src.network.detail.MaskedLayer) -> src.network.detail.MaskedLayer:
        return layer

class UniformRegrowth(Regrowth):

    def __init__(self, weight_reinitialization: str = "xavier_uniform"):
        self.reinitialization = src.rules.weight.Reinitialization.lookup(weight_reinitialization)

    def apply(self, pruned_count: int, layer: src.network.detail.MaskedLayer) -> src.network.detail.MaskedLayer:
        mask = layer.mask.data

        flat_mask = mask.view(-1)

        # get indices of inactive elements
        inactive_idx = (flat_mask == 0).nonzero(as_tuple=True)[0]
        if inactive_idx.numel() == 0:
            return layer

        # pick their random subset
        perm = torch.randperm(inactive_idx.size(0), device=inactive_idx.device)
        grow = inactive_idx[perm[:pruned_count]]

        # update mask
        flat_mask[grow] = 1

        layer.mask.data = flat_mask.view_as(mask)
        layer.layer.weight.data = self.reinitialization(layer.layer.weight.data, grow)

        return layer

class GradientRegrowth(Regrowth):

    def __init__(self, decay_rate: float, weight_reinitialization: str = "zero"):
        self.decay_rate = decay_rate
        self.reinitialization = src.rules.weight.Reinitialization.lookup(weight_reinitialization)

    def apply(self, pruned_count: int, layer: src.network.detail.MaskedLayer) -> src.network.detail.MaskedLayer:
        weight = layer.layer.weight
        original_mask = layer.mask.data

        if weight.grad is None:
            raise RuntimeError("Gradient of the weights is required for GradientRegrowth, but it is None.")

        grad = weight.grad.data

        flat_grad = grad.view(-1)
        flat_mask = original_mask.view(-1)

        # get indices of inactive elements
        inactive_idx = (flat_mask == 0).nonzero(as_tuple=True)[0]
        if inactive_idx.numel() == 0:
            return layer

        # score by gradient magnitude
        scores = flat_grad[inactive_idx].abs()

        # sort descending to satisfy soft_top_k assumption
        sorted_scores, order = torch.sort(scores, descending=True)
        sorted_indices = inactive_idx[order]

        # apply soft_top_k to sorted scores
        sel_mask = src.utilities.soft_top_k(sorted_scores, select=pruned_count, decay_rate=self.decay_rate)
        grow = sorted_indices[sel_mask]

        # update mask
        flat_mask[grow] = 1

        layer.mask.data = flat_mask.view_as(original_mask)
        layer.layer.weight.data = self.reinitialization(weight.data, grow)

        return layer

class GuidedUniformRegrowth(Regrowth):

    def __init__(self, weight_reinitialization: str = "xavier_uniform", guidance_boost: float = 100.0, mode: str = "functional"):
        assert guidance_boost >= 1.0
        self.reinitialization = src.rules.weight.Reinitialization.lookup(weight_reinitialization)
        self.guidance_boost = guidance_boost
        self.accessor = {
            "forward": lambda layer: layer.forward_reachability,
            "backward": lambda layer: layer.backward_reachability,
            "functional": lambda layer: layer.functional_reachability,
            "nonfunctional": lambda layer: (layer.forward_reachability | layer.backward_reachability) & (~layer.functional_reachability),
            "dead": lambda layer: ~(layer.forward_reachability | layer.backward_reachability),
            "combined": lambda layer: (layer.forward_reachability | layer.backward_reachability)
        }[mode]

    def apply(self, pruned_count: int, layer: src.network.detail.MaskedLayer) -> src.network.detail.MaskedLayer:

        mask = layer.mask.data
        guidance = self.accessor(layer)

        flat_mask = mask.view(-1)
        flat_guidance = guidance.view(-1)

        inactive_idx = (flat_mask == 0).nonzero(as_tuple=True)[0]
        if inactive_idx.numel() == 0:
            return layer

        # weights: base = 1, boosted where guidance == True
        weights = torch.ones(inactive_idx.numel(), device=inactive_idx.device, dtype=torch.float)

        guided = flat_guidance[inactive_idx].bool()
        weights[guided] *= self.guidance_boost

        # handle pruned_count > available
        k = min(pruned_count, inactive_idx.numel())

        # weighted uniform sampling without replacement
        sampled = torch.multinomial(weights, k, replacement=False)
        grow = inactive_idx[sampled]

        # update mask
        flat_mask[grow] = 1
        layer.mask.data = flat_mask.view_as(mask)

        # reinitialize grown weights
        layer.layer.weight.data = self.reinitialization(layer.layer.weight.data, grow)

        return layer

class GuidedGradientRegrowth(Regrowth):

    def __init__(self, decay_rate: float, weight_reinitialization: str = "zero", mode: str = "functional"):
        self.decay_rate = decay_rate
        self.reinitialization = src.rules.weight.Reinitialization.lookup(weight_reinitialization)
        self.accessor = {
            "forward": lambda layer: layer.forward_reachability,
            "backward": lambda layer: layer.backward_reachability,
            "functional": lambda layer: layer.functional_reachability,
            "nonfunctional": lambda layer: (layer.forward_reachability | layer.backward_reachability) & (~layer.functional_reachability),
            "dead": lambda layer: ~(layer.forward_reachability | layer.backward_reachability),
            "combined": lambda layer: (layer.forward_reachability | layer.backward_reachability)
        }[mode]

    def apply(self, pruned_count: int, layer: src.network.detail.MaskedLayer) -> src.network.detail.MaskedLayer:
        weight = layer.layer.weight
        original_mask = layer.mask.data

        if weight.grad is None:
            raise RuntimeError("Gradient of the weights is required for GradientRegrowth, but it is None.")

        grad = weight.grad.data
        guidance = self.accessor(layer)

        flat_grad = grad.view(-1)
        flat_mask = original_mask.view(-1)
        flat_guidance = guidance.view(-1)

        # get indices of inactive elements
        inactive_idx = (flat_mask == 0).nonzero(as_tuple=True)[0]
        if inactive_idx.numel() == 0:
            return layer

        # score by gradient magnitude
        scores = flat_grad[inactive_idx].abs() * flat_guidance[inactive_idx]

        # sort descending to satisfy soft_top_k assumption
        sorted_scores, order = torch.sort(scores, descending=True)
        sorted_indices = inactive_idx[order]

        # apply soft_top_k to sorted scores
        sel_mask = src.utilities.soft_top_k(sorted_scores, select=pruned_count, decay_rate=self.decay_rate)
        grow = sorted_indices[sel_mask]

        # update mask
        flat_mask[grow] = 1

        layer.mask.data = flat_mask.view_as(original_mask)
        layer.layer.weight.data = self.reinitialization(weight.data, grow)

        return layer
