from abc import ABC, abstractmethod
from typing import Tuple
import torch

class ActivationFunctionWrapper(torch.nn.Module):

    def __init__(self, activation: torch.nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)

    def dormancy_score(self, x: torch.Tensor, tau: float = 0.1) -> float: # The Dormant Neuron Phenomenon in Deep Reinforcement Learning
        if len(x.shape) == 2:
            expectation = x.mean(dim=0)
            expectation = expectation / ((1.0 / expectation.shape[0]) * expectation.sum())
            score = (expectation <= tau).int().sum() / expectation.shape[0]
            return score.item()
        else:
            raise RuntimeError("unsupported shape for the input!")

class MaskedWeight(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return weight * mask

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return gradient, None

class MaskedLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, sparsity: float):
        super().__init__()
        assert isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)), "MaskedLayer supports Linear or Conv2d only"
        assert 0.0 <= sparsity < 1.0, "sparsity must be in [0, 1.0)"

        self.layer = layer
        self.sparsity = sparsity

        # Create mask with given sparsity
        mask = torch.zeros_like(layer.weight)
        k = int((1 - sparsity) * mask.numel())
        mask.view(-1)[torch.randperm(mask.numel())[:k]] = 1
        self.mask = torch.nn.Parameter(mask, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = MaskedWeight.apply(self.layer.weight, self.mask) # otherwise backward pass does not propagate gradients for zeroed weights
        if isinstance(self.layer, torch.nn.Linear): # there are much better ways to do this but it works for now
            return torch.nn.functional.linear(x, masked_weight, self.layer.bias)
        elif isinstance(self.layer, torch.nn.Conv2d):
            return torch.nn.functional.conv2d(x, masked_weight, self.layer.bias, stride=self.layer.stride, padding=self.layer.padding, dilation=self.layer.dilation, groups=self.layer.groups)
        raise RuntimeError(f"unsupported layer type: {type(self.layer)} in MaskedLayer! ")

    def expected_sparsity(self) -> float:
        return self.sparsity

    def realized_sparsity(self) -> float:
        return 1.0 - self.mask.sum().item() / self.mask.numel()

    def active_weights(self) -> int:
        return int(self.mask.sum().item())

    def total_weights(self) -> int:
        return self.mask.numel()

    def estimate_flops(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        def flop_estimate(non_zero: int):
            # now estimate effective flop count based on type of layer
            if isinstance(self.layer, torch.nn.Linear):
                weight_flops = 2.0 * non_zero
                bias_flops = self.layer.out_features if self.layer.bias is not None else 0
                return weight_flops + bias_flops
            elif isinstance(self.layer, torch.nn.Conv2d):
                _, _, H_in, W_in = input_tensor.shape

                Kh, Kw = self.layer.kernel_size
                Sh, Sw = self.layer.stride
                Ph, Pw = self.layer.padding
                Dh, Dw = self.layer.dilation

                H_out = (H_in + 2 * Ph - Dh * (Kh - 1) - 1) // Sh + 1
                W_out = (W_in + 2 * Pw - Dw * (Kw - 1) - 1) // Sw + 1

                weight_flops = 2 * (non_zero / self.layer.out_channels) * H_out * W_out
                bias_flops = self.layer.out_channels * H_out * W_out if self.layer.bias is not None else 0

                return weight_flops + bias_flops
            raise RuntimeError(f"unsupported layer type: {type(self.layer)} in MaskedLayer! ")

        return flop_estimate(self.total_weights()), flop_estimate(self.active_weights())

class NetworkInterface(torch.nn.Module, ABC):

    def expected_sparsity(self) -> float:
        data = [layer.expected_sparsity() for layer in self.modules() if isinstance(layer, MaskedLayer)]
        return sum(data) / len(data)

    def realized_sparsity(self) -> float:
        data = [layer.realized_sparsity() for layer in self.modules() if isinstance(layer, MaskedLayer)]
        return sum(data) / len(data)

    def active_weights(self) -> int:
        return sum(
            [layer.active_weights() for layer in self.modules() if isinstance(layer, MaskedLayer)])

    def total_weights(self) -> int:
        return sum([layer.total_weights() for layer in self.modules() if isinstance(layer, MaskedLayer)])
