from abc import ABC, abstractmethod
from typing import Tuple, List
import torch
import src.network.mask_initialization

def initialize_masks(network: torch.nn.Module, method: str = 'uniform', sparsity: float = 1.0) -> None:
    assert 0.0 <= sparsity < 1.0, "sparsity must be in [0, 1.0)"

    layers = []

    for module in network.modules():
        if isinstance(module, MaskedLayer):
            layers.append(module)

    if method == 'uniform':
        for layer in layers:
            layer.mask.copy_(src.network.mask_initialization.uniform(layer.mask, sparsity))
    elif method == 'erk':
        is_dense, epsilon = src.network.mask_initialization.compute_erk_initialization([layer.nout_nin_kh_kw() for layer in layers], sparsity)
        for dense, layer in zip(is_dense, layers):
            if not dense:
                layer.mask.copy_(src.network.mask_initialization.erk(layer.mask, layer.nout_nin_kh_kw(), epsilon))
    else:
        raise RuntimeError(f"unsupported mask initialization: {method}")

class ActivationFunctionWrapper(torch.nn.Module):

    def __init__(self, activation: torch.nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)

    def dormancy_score(self, x: torch.Tensor, tau: float = 0.1) -> Tuple[torch.Tensor, float]: # The Dormant Neuron Phenomenon in Deep Reinforcement Learning
        if len(x.shape) == 2:
            expectation = x.abs().mean(dim=0) # abs is my idea
        elif len(x.shape) == 4:
            expectation = x.abs().mean(dim=(0, 2, 3)) # abs is my idea
        else:
            raise RuntimeError(f"unsupported shape for the input: {x.shape}")

        # normalize
        expectation = expectation / ((1.0 / expectation.shape[0]) * expectation.sum())
        score = (expectation <= tau).int().sum() / expectation.shape[0]
        return expectation, score.item()

class MaskedWeight(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return weight * mask

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return gradient, None

class MaskedLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        assert isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)), "MaskedLayer supports Linear or Conv2d only"

        self.layer = layer
        self.mask = torch.nn.Parameter(torch.ones_like(layer.weight), requires_grad=False)

        # used to compute flow going through node
        self.register_buffer('forward_flow', torch.zeros_like(layer.weight))

        self.register_buffer('backward_flow', torch.zeros_like(layer.weight))
        self.register_buffer('backward_reachability', torch.zeros_like(layer.weight))
        self.register_buffer('functional_reachability', torch.zeros_like(layer.weight))

        self._activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._activation = x # for flow computation

        masked_weight = MaskedWeight.apply(self.layer.weight, self.mask) # otherwise backward pass does not propagate gradients for zeroed weights
        if isinstance(self.layer, torch.nn.Linear): # there are much better ways to do this but it works for now
            return torch.nn.functional.linear(x, masked_weight, self.layer.bias)
        elif isinstance(self.layer, torch.nn.Conv2d):
            return torch.nn.functional.conv2d(x, masked_weight, self.layer.bias, stride=self.layer.stride, padding=self.layer.padding, dilation=self.layer.dilation, groups=self.layer.groups)
        raise RuntimeError(f"unsupported layer type: {type(self.layer)} in MaskedLayer! ")

    def nout_nin_kh_kw(self) -> Tuple[int, int, int, int]: # amazing method name
        if isinstance(self.layer, torch.nn.Linear):
            n_out, n_in = self.layer.weight.shape
            return n_out, n_in, 0, 0
        elif isinstance(self.layer, torch.nn.Conv2d):
            n_out, n_in, kH, kW = self.layer.weight.shape
            return n_out, n_in, kH, kW
        raise RuntimeError(f"unsupported layer type: {type(self.layer)} in MaskedLayer! ")

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

    def accumulate_flow(self) -> None:
        if self.layer.weight.grad is None or self._activation is None:
            return

        act = self._activation.detach().abs()
        grad = self.layer.weight.grad.detach().abs()
        weight = self.layer.weight.detach().abs()
        mask = self.mask

        if isinstance(self.layer, torch.nn.Linear):
            dim = 0
        elif isinstance(self.layer, torch.nn.Conv2d):
            dim = (0, 2, 3)
        else:
            raise RuntimeError(f"unsupported layer type: {type(self.layer)} in MaskedLayer! ")

        self.forward_flow += (act.mean(dim=dim, keepdim=True) * weight) * mask
        self.backward_flow += (grad * weight) * self.backward_reachability * mask

    def reset_flow(self) -> None:
        self.forward_flow.zero_()
        self.backward_flow.zero_()

class NetworkInterface(torch.nn.Module, ABC):

    def realized_sparsity(self) -> float:
        return 1.0 - self.active_weights() / self.total_weights()

    def active_weights(self) -> int:
        return sum([layer.active_weights() for layer in self.modules() if isinstance(layer, MaskedLayer)])

    def total_weights(self) -> int:
        return sum([layer.total_weights() for layer in self.modules() if isinstance(layer, MaskedLayer)])

    def accumulate_flow(self) -> None:
        for layer in self.modules():
            if isinstance(layer, MaskedLayer):
                layer.accumulate_flow()

    def reset_flow(self) -> None:
        with torch.no_grad():
            for layer in self.modules():
                if isinstance(layer, MaskedLayer):
                    layer.reset_flow()
