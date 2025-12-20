from typing import Tuple

import torch
import mask.utilities
import numpy

class MaskedWeight(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, mask):
        return weight * mask

    @staticmethod
    def backward(ctx, gradient):
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

    def expected_sparsity(self) -> float:
        return self.sparsity

    def realized_sparsity(self) -> float:
        return 1.0 - self.mask.sum().item() / self.mask.numel()

    def active_weights(self) -> int:
        return int(self.mask.sum().item())

    def total_weights(self) -> int:
        return self.mask.numel()

    def forward(self, x):
        masked_weight = MaskedWeight.apply(self.layer.weight, self.mask) # otherwise backward pass does not propagate gradients for zeroed weights
        if isinstance(self.layer, torch.nn.Linear): # there are much better ways to do this but it works for now
            return torch.nn.functional.linear(x, masked_weight, self.layer.bias)
        elif isinstance(self.layer, torch.nn.Conv2d):
            return torch.nn.functional.conv2d(x, masked_weight, self.layer.bias, stride=self.layer.stride, padding=self.layer.padding, dilation=self.layer.dilation, groups=self.layer.groups)
        raise RuntimeError(f"unsupported layer type: {type(self.layer)} in MaskedLayer! ")

class DormantNeuronHelper(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def dormancy_score(self, x, tau: float = 0.1) -> float:
        if len(x.shape) == 2:
            expectation = x.mean(dim=0)
            expectation = expectation / ((1.0 / expectation.shape[0]) * expectation.sum())
            score = (expectation <= tau).int().sum() / expectation.shape[0]
            return score.item()
        else:
            raise RuntimeError("unsupported shape for the input!")

class SparseANN(torch.nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_hidden_layers: int, num_classes: int, sparsity: float, act: str, layernorm: bool):
        super().__init__()

        act_f = torch.nn.Tanh if act == "tanh" else torch.nn.ReLU
        norm = torch.nn.LayerNorm if layernorm else torch.nn.Identity

        layers = []

        # input layer
        layers.append(MaskedLayer(torch.nn.Linear(in_features, hidden_dim, bias=False), sparsity))
        layers.append(norm(hidden_dim))
        layers.append(act_f())
        layers.append(DormantNeuronHelper())

        # hidden layers
        for layer_index in range(num_hidden_layers):
            layers.append(MaskedLayer(torch.nn.Linear(hidden_dim, hidden_dim, bias=False), sparsity))
            layers.append(norm(hidden_dim))
            layers.append(act_f())
            layers.append(DormantNeuronHelper())

        # output layer
        layers.append(MaskedLayer(torch.nn.Linear(hidden_dim, num_classes), sparsity))

        self.model = torch.nn.Sequential(*layers)

    def expected_sparsity(self) -> float:
        data = [layer.expected_sparsity() for layer in self.model if isinstance(layer, MaskedLayer)]
        return sum(data) / len(data)

    def realized_sparsity(self) -> float:
        data = [layer.realized_sparsity() for layer in self.model if isinstance(layer, MaskedLayer)]
        return sum(data) / len(data)

    def active_weights(self) -> int:
        return sum([layer.active_weights() for layer in self.model if isinstance(layer, MaskedLayer)])

    def total_weights(self) -> int:
        return sum([layer.total_weights() for layer in self.model if isinstance(layer, MaskedLayer)])

    def forward(self, x) -> Tuple[torch.Tensor, float]:
        scores = []

        for layer in self.model:
            x = layer(x)
            if isinstance(layer, mask.network.DormantNeuronHelper):
                scores.append(layer.dormancy_score(x, 0.1))

        return x, numpy.mean(scores)

class DenseANNFromSparseANN(torch.nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_hidden_layers: int, num_classes: int, sparsity: float, act: str, layernorm: bool):
        super().__init__()

        network_depth = 3
        weight_target = SparseANN(in_features=in_features, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, num_classes=num_classes, sparsity=sparsity, act=act, layernorm=layernorm).active_weights()
        neuron_count = mask.utilities.estimate_neuron_count(features=in_features, classes=num_classes, network_depth=network_depth, parameter_target=weight_target)

        act_f = torch.nn.Tanh if act == "tanh" else torch.nn.ReLU
        norm = torch.nn.LayerNorm if layernorm else torch.nn.Identity

        layers = []

        # input layer
        layers.append(MaskedLayer(torch.nn.Linear(in_features, neuron_count, bias=False), 0.0))
        layers.append(norm(neuron_count))
        layers.append(act_f())
        layers.append(DormantNeuronHelper())

        # hidden layers
        for _ in range(network_depth - 2):
            layers.append(MaskedLayer(torch.nn.Linear(neuron_count, neuron_count, bias=False), 0.0))
            layers.append(norm(neuron_count))
            layers.append(act_f())
            layers.append(DormantNeuronHelper())

        # output layer
        layers.append(MaskedLayer(torch.nn.Linear(neuron_count, num_classes), 0.0))

        self.model = torch.nn.Sequential(*layers)

    def expected_sparsity(self) -> float:
        data = [layer.expected_sparsity() for layer in self.model if isinstance(layer, MaskedLayer)]
        return sum(data) / len(data)

    def realized_sparsity(self) -> float:
        data = [layer.realized_sparsity() for layer in self.model if isinstance(layer, MaskedLayer)]
        return sum(data) / len(data)

    def active_weights(self) -> int:
        return sum([layer.active_weights() for layer in self.model if isinstance(layer, MaskedLayer)])

    def total_weights(self) -> int:
        return sum([layer.total_weights() for layer in self.model if isinstance(layer, MaskedLayer)])

    def forward(self, x) -> Tuple[torch.Tensor, float]:
        scores = []

        for layer in self.model:
            x = layer(x)
            if isinstance(layer, mask.network.DormantNeuronHelper):
                scores.append(layer.dormancy_score(x, 0.1))

        return x, numpy.mean(scores)
