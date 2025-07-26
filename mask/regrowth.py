from abc import ABC, abstractmethod
import torch

import mask.utilities
import mask.network

class Regrowth(ABC):

    @abstractmethod
    def apply(self, pruned_count: int, layer: mask.network.MaskedLayer) -> mask.network.MaskedLayer: # regrown_layer
        pass

class NullRegrowth(Regrowth):

    def apply(self, pruned_count: int, layer: mask.network.MaskedLayer) -> mask.network.MaskedLayer:
        return layer

class UniformRegrowth(Regrowth):

    def apply(self, pruned_count: int, layer: mask.network.MaskedLayer) -> mask.network.MaskedLayer:
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


class GradientRegrowth(Regrowth):

    def __init__(self, decay_rate: float):
        self.decay_rate = decay_rate

    def apply(self, pruned_count: int, layer: mask.network.MaskedLayer) -> mask.network.MaskedLayer:
        weight = layer.layer.weight
        original_mask = layer.mask.data

        if weight.grad is None:
            raise RuntimeError("Gradient of the weights is required for GradientRegrowth, but it is None.")

        grad = weight.grad.data
        flat_grad = grad.view(-1)
        flat_mask = original_mask.view(-1)

        # identify inactive (pruned) weights
        inactive_idx = (flat_mask == 0).nonzero(as_tuple=True)[0]
        if inactive_idx.numel() == 0:
            return layer

        # score by gradient magnitude
        scores = flat_grad[inactive_idx].abs()

        # sort descending to satisfy soft_top_k assumption
        sorted_scores, order = torch.sort(scores, descending=True)
        sorted_indices = inactive_idx[order]

        # apply soft_top_k to sorted scores
        sel_mask = mask.utilities.soft_top_k(sorted_scores, select=pruned_count, decay_rate=self.decay_rate)
        grow = sorted_indices[sel_mask]

        # regrow these weights
        flat_mask[grow] = 1
        weight.data.view(-1)[grow] = 0.0
        layer.mask.data = flat_mask.view_as(original_mask)

        return layer
