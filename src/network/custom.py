from typing import Tuple
import src.utilities
import src.network.detail

import torch
import numpy

class SparseANN(src.network.detail.NetworkInterface):
    def __init__(self, in_features: int, hidden_dim: int, num_hidden_layers: int, num_classes: int, sparsity: float, act: str, layernorm: bool):
        super().__init__()

        act_f = torch.nn.Tanh if act == "tanh" else torch.nn.ReLU
        norm = torch.nn.LayerNorm if layernorm else torch.nn.Identity

        layers = []

        # input layer
        layers.append(src.network.detail.MaskedLayer(torch.nn.Linear(in_features, hidden_dim, bias=False), sparsity))
        layers.append(norm(hidden_dim))
        layers.append(src.network.detail.ActivationFunctionWrapper(act_f()))

        # hidden layers
        for layer_index in range(num_hidden_layers):
            layers.append(src.network.detail.MaskedLayer(torch.nn.Linear(hidden_dim, hidden_dim, bias=False), sparsity))
            layers.append(norm(hidden_dim))
            layers.append(src.network.detail.ActivationFunctionWrapper(act_f()))

        # output layer
        layers.append(src.network.detail.MaskedLayer(torch.nn.Linear(hidden_dim, num_classes), sparsity))
        layers.append(src.network.detail.ActivationFunctionWrapper(torch.nn.Identity()))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        for layer in self.model:
            x = layer(x)
        return x

class DenseANNFromSparseANN(src.network.detail.NetworkInterface):
    def __init__(self, in_features: int, hidden_dim: int, num_hidden_layers: int, num_classes: int, sparsity: float, act: str, layernorm: bool):
        super().__init__()

        network_depth = 3
        weight_target = SparseANN(in_features=in_features, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, num_classes=num_classes, sparsity=sparsity, act=act, layernorm=layernorm).active_weights()
        neuron_count = src.utilities.estimate_neuron_count(features=in_features, classes=num_classes, network_depth=network_depth, parameter_target=weight_target)

        act_f = torch.nn.Tanh if act == "tanh" else torch.nn.ReLU
        norm = torch.nn.LayerNorm if layernorm else torch.nn.Identity

        layers = []

        # input layer
        layers.append(src.network.detail.MaskedLayer(torch.nn.Linear(in_features, neuron_count, bias=False), 0.0))
        layers.append(norm(neuron_count))
        layers.append(src.network.detail.ActivationFunctionWrapper(act_f()))

        # hidden layers
        for _ in range(network_depth - 2):
            layers.append(src.network.detail.MaskedLayer(torch.nn.Linear(neuron_count, neuron_count, bias=False), 0.0))
            layers.append(norm(neuron_count))
            layers.append(src.network.detail.ActivationFunctionWrapper(act_f()))

        # output layer
        layers.append(src.network.detail.MaskedLayer(torch.nn.Linear(neuron_count, num_classes), 0.0))
        layers.append(src.network.detail.ActivationFunctionWrapper(torch.nn.Identity()))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        for layer in self.model:
            x = layer(x)
        return x
