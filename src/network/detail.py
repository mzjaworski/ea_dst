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

    network.global_weight_target = network.active_weights()


class MaskedWeight(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return weight * mask

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return gradient, None

class Graph(ABC):

    @abstractmethod
    def forward_pass(self, input_connectivity: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def backward_pass(self, output_connectivity: torch.Tensor) -> torch.Tensor:
        pass

class MaskedLayer(torch.nn.Module, Graph, ABC):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()

        self.layer = layer
        self.mask = torch.nn.Parameter(torch.ones_like(layer.weight), requires_grad=False)

        # used to compute flow going through node
        self.register_buffer('_forward_reachability', torch.zeros_like(layer.weight).to_sparse())
        self.register_buffer('_backward_reachability', torch.zeros_like(layer.weight).to_sparse())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.layer.weight.data = MaskedWeight.apply(self.layer.weight.data, self.mask) # otherwise backward pass does not propagate gradients for zeroed weights
        return self.layer(x)

    def realized_sparsity(self) -> float:
        return 1.0 - self.mask.sum().item() / self.mask.numel()

    def active_weights(self) -> int:
        return int(self.mask.sum().item())

    def total_weights(self) -> int:
        return self.mask.numel()

    @abstractmethod
    def nout_nin_kh_kw(self) -> Tuple[int, int, int, int]: # amazing method name
        pass

    @abstractmethod
    def estimate_flops(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        pass

    @property
    def forward_reachability(self) -> torch.Tensor:
        return self._forward_reachability.clone().to(torch.bool).to_dense()

    @property
    def backward_reachability(self) -> torch.Tensor:
        return self._backward_reachability.clone().to(torch.bool).to_dense()

class MaskedLinear(MaskedLayer):

    def __init__(self, layer: torch.nn.Linear):
        assert isinstance(layer, torch.nn.Linear), "MaskedLinear supports Linear only"
        super().__init__(layer)

    def nout_nin_kh_kw(self) -> Tuple[int, int, int, int]: # amazing method name
        n_out, n_in = self.layer.weight.shape
        return n_out, n_in, 0, 0

    def estimate_flops(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        def flop_estimate(non_zero: int):
            weight_flops = 2.0 * non_zero
            bias_flops = self.layer.out_features if self.layer.bias is not None else 0
            return weight_flops + bias_flops
        return int(flop_estimate(self.total_weights())), int(flop_estimate(self.active_weights()))

    def forward_pass(self, input_connectivity: torch.Tensor) -> torch.Tensor:
        original_shape = input_connectivity.shape
        in_features = self.mask.shape[1]
        out_features = self.mask.shape[0]

        # MLP-Mixer requires some additional logic to make it work properly
        x = input_connectivity.reshape(-1, in_features)
        outputs = []
        for sample in x:
            out_reach = (self.mask.to(torch.bool) & sample.unsqueeze(0)).any(dim=1)
            outputs.append(out_reach)

        self._forward_reachability.copy_((torch.ones_like(outputs[-1]).unsqueeze(1) & x.any(dim=0).unsqueeze(0)).bool().to_sparse()) # we store this purely for logging; everything can be accumulated to one matrix
        output_reachability = torch.stack(outputs, dim=0).reshape(*original_shape[:-1], out_features)

        return output_reachability

    def backward_pass(self, output_connectivity: torch.Tensor) -> torch.Tensor:
        original_shape = output_connectivity.shape
        in_features = self.mask.shape[1]
        out_features = self.mask.shape[0]

        # MLP-Mixer requires some additional logic to make it work properly
        x = output_connectivity.reshape(-1, out_features)
        inputs = []
        for sample in x:
            in_reach = (self.mask.to(torch.bool) & sample.unsqueeze(1)).any(dim=0)
            inputs.append(in_reach)

        self._backward_reachability.copy_((x.any(dim=0).unsqueeze(1) & torch.ones_like(inputs[-1]).unsqueeze(0)).bool().to_sparse()) # we store this purely for logging; everything can be accumulated to one matrix
        input_reachability = torch.stack(inputs, dim=0).reshape(*original_shape[:-1], in_features)

        return input_reachability

class MaskedConv2D(MaskedLayer):

    def __init__(self, layer: torch.nn.Module):
        assert isinstance(layer, torch.nn.Conv2d), "MaskedConv2D supports Conv2d only"
        super().__init__(layer)

    def nout_nin_kh_kw(self) -> Tuple[int, int, int, int]: # amazing method name
        n_out, n_in, kH, kW = self.layer.weight.shape
        return n_out, n_in, kH, kW

    def estimate_flops(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        def flop_estimate(non_zero: int):
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

        return int(flop_estimate(self.total_weights())), int(flop_estimate(self.active_weights()))

    def forward_pass(self, input_connectivity: torch.Tensor) -> torch.Tensor:
        in_channels, H_in, W_in = input_connectivity.shape

        H_out = (H_in + 2 * self.layer.padding[0] - self.layer.dilation[0] * (self.layer.kernel_size[0] - 1) - 1) // self.layer.stride[0] + 1
        W_out = (W_in + 2 * self.layer.padding[1] - self.layer.dilation[1] * (self.layer.kernel_size[1] - 1) - 1) // self.layer.stride[1] + 1

        self._forward_reachability.copy_(self.mask.clone().bool().to_sparse()) # we store this purely for logging; everything can be accumulated to one matrix
        output_reachability = torch.ones((self.layer.out_channels, H_out, W_out), dtype=torch.bool, device=input_connectivity.device)

        return output_reachability

    def backward_pass(self, output_connectivity: torch.Tensor) -> torch.Tensor:
        out_channels, H_out, W_out = output_connectivity.shape
        in_channels, kernel_h, kernel_w = self.mask.shape[1], self.mask.shape[2], self.mask.shape[3]

        H_in = (H_out - 1) * self.layer.stride[0] - 2 * self.layer.padding[0] + self.layer.dilation[0] * (kernel_h - 1) + 1
        W_in = (W_out - 1) * self.layer.stride[1] - 2 * self.layer.padding[1] + self.layer.dilation[1] * (kernel_w - 1) + 1

        self._backward_reachability.copy_(self.mask.clone().bool().to_sparse()) # we store this purely for logging; everything can be accumulated to one matrix
        input_reachability = torch.ones((in_channels, H_in, W_in), dtype=torch.bool, device=output_connectivity.device)

        return input_reachability

class NetworkInterface(torch.nn.Module, ABC):

    def realized_sparsity(self) -> float:
        return 1.0 - self.active_weights() / self.total_weights()

    def active_weights(self) -> int:
        return sum([layer.active_weights() for layer in self.modules() if isinstance(layer, MaskedLayer)])

    def total_weights(self) -> int:
        return sum([layer.total_weights() for layer in self.modules() if isinstance(layer, MaskedLayer)])
