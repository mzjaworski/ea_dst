from typing import Callable
import torch

class Reinitialization:

    @staticmethod
    def none(weights: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return weights

    @staticmethod
    def zero(weights: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        weights.view(-1)[positions] = 0.0
        return weights

    @staticmethod
    def xavier_uniform(weights: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        weights.view(-1)[positions] = torch.nn.init.xavier_uniform_(torch.zeros_like(weights)).view(-1)[positions]
        return weights

    registry: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "none": none.__func__,
        "zero": zero.__func__,
        "xavier_uniform": xavier_uniform.__func__,
    }

    @classmethod
    def lookup(cls, name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if name not in cls.registry:
            raise ValueError(f"Reinitialization '{name}' not found. Available: {list(cls.registry.keys())}")
        return cls.registry[name]
