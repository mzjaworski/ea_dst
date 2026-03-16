from typing import Callable
import torch

import src.network.detail

class Reachability:

    @staticmethod
    def forward(layer: src.network.detail.MaskedLayer):
        return layer.forward_reachability

    @staticmethod
    def backward(layer: src.network.detail.MaskedLayer):
        return layer.backward_reachability

    @staticmethod
    def functional(layer: src.network.detail.MaskedLayer):
        functional = layer.forward_reachability & layer.backward_reachability
        if torch.sum(functional) == 0:
            functional = layer.forward_reachability | layer.backward_reachability
        return functional

    @staticmethod
    def nonfunctional(layer: src.network.detail.MaskedLayer):
        return (layer.forward_reachability | layer.backward_reachability) & Reachability.functional(layer)

    @staticmethod
    def dead(layer: src.network.detail.MaskedLayer):
        return ~(layer.forward_reachability | layer.backward_reachability)

    @staticmethod
    def combined(layer: src.network.detail.MaskedLayer):
        return layer.forward_reachability | layer.backward_reachability

    registry: dict[str, Callable[[src.network.detail.MaskedLayer], torch.Tensor]] = {
        "forward": forward.__func__,
        "backward": backward.__func__,
        "functional": functional.__func__,
        "nonfunctional": nonfunctional.__func__,
        "dead": dead.__func__,
        "combined": combined.__func__,
    }

    @classmethod
    def lookup(cls, name: str) -> Callable[[src.network.detail.MaskedLayer], torch.Tensor]:
        if name not in cls.registry:
            raise ValueError(f"Reachability '{name}' not found. Available: {list(cls.registry.keys())}")
        return cls.registry[name]
