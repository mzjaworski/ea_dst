from typing import Tuple, List
import src.utilities
import src.network.detail

import torch
import numpy

class SparseANN(src.network.detail.NetworkInterface):
    def __init__(self, layers: List[Tuple[int, int]] = [(3072, 1024), (1024, 512), (512, 512), (512, 10)], sparsity: float = 0.0, act: str = "relu", layernorm: bool = True, method: str = 'uniform'):
        super().__init__()

        act_f = torch.nn.Tanh if act == "tanh" else torch.nn.ReLU
        norm = torch.nn.LayerNorm if layernorm else torch.nn.Identity

        self.in_features = layers[0][0]
        self.num_classes = layers[-1][1]

        network = []
        for layer_index, (in_dim, out_dim) in enumerate(layers):
            network.append(src.network.detail.MaskedLinear(torch.nn.Linear(in_dim, out_dim, bias=False)))
            if layer_index < len(layers) - 1:
                network.append(norm(out_dim))
                network.append(act_f())
        self.model = torch.nn.Sequential(*network)

        src.network.detail.initialize_masks(self, method=method, sparsity=sparsity)

    def forward(self, x) -> torch.Tensor:
        return self.model(torch.flatten(x, start_dim=1))

    def compute_reachabilities(self) -> None:
        modules = [layer for layer in self.modules() if isinstance(layer, src.network.detail.MaskedLayer)]
        device = next(self.parameters()).device

        forward = torch.ones(self.in_features, dtype=torch.bool).to(device)
        backward = torch.ones(self.num_classes, dtype=torch.bool).to(device)

        for module in modules:
            forward = module.forward_pass(forward)

        for module in reversed(modules):
            backward = module.backward_pass(backward)


class DenseANNFromSparseANN(src.network.detail.NetworkInterface):
    def __init__(self, layers: List[Tuple[int, int]] = [(3072, 1024), (1024, 512), (512, 512), (512, 10)], sparsity: float = 0.0, act: str = "relu", layernorm: bool = True, method: str = 'uniform'):
        super().__init__()

        self.in_features = layers[0][0]
        self.num_classes = layers[-1][1]

        network_depth = 3
        weight_target = SparseANN(layers=layers, sparsity=sparsity, act=act, layernorm=layernorm, method=method).active_weights()
        neuron_count = src.utilities.estimate_neuron_count(features=self.in_features, classes=self.num_classes, network_depth=network_depth, parameter_target=weight_target)

        act_f = torch.nn.Tanh if act == "tanh" else torch.nn.ReLU
        norm = torch.nn.LayerNorm if layernorm else torch.nn.Identity

        network = []

        network.append(src.network.detail.MaskedLinear(torch.nn.Linear(self.in_features, neuron_count, bias=False)))
        network.append(norm(neuron_count))
        network.append(act_f())

        for _ in range(network_depth - 2):
            network.append(src.network.detail.MaskedLinear(torch.nn.Linear(neuron_count, neuron_count, bias=False)))
            network.append(norm(neuron_count))
            network.append(act_f())

        network.append(src.network.detail.MaskedLinear(torch.nn.Linear(neuron_count, self.num_classes)))
        network.append(torch.nn.Identity())

        self.model = torch.nn.Sequential(*network)
        self.global_weight_target = self.active_weights()

    def forward(self, x) -> torch.Tensor:
        return self.model(torch.flatten(x, start_dim=1))

    def compute_reachabilities(self) -> None:
        modules = [layer for layer in self.modules() if isinstance(layer, src.network.detail.MaskedLayer)]
        device = next(self.parameters()).device

        forward = torch.ones(self.in_features, dtype=torch.bool).to(device)
        backward = torch.ones(self.num_classes, dtype=torch.bool).to(device)

        for module in modules:
            forward = module.forward_pass(forward)

        for module in reversed(modules):
            backward = module.backward_pass(backward)
