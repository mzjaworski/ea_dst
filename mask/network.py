import torch
import mask.utilities

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

        # hidden layers
        for _ in range(num_hidden_layers):
            layers.append(MaskedLayer(torch.nn.Linear(hidden_dim, hidden_dim, bias=False), sparsity))
            layers.append(norm(hidden_dim))
            layers.append(act_f())

        # output layer
        layers.append(torch.nn.Linear(hidden_dim, num_classes))

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

    def forward(self, x):
        return self.model(x)

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

        # hidden layers
        for _ in range(network_depth - 2):
            layers.append(MaskedLayer(torch.nn.Linear(neuron_count, neuron_count, bias=False), 0.0))
            layers.append(norm(neuron_count))
            layers.append(act_f())

        # output layer
        layers.append(torch.nn.Linear(neuron_count, num_classes))

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

    def forward(self, x):
        return self.model(x)

class SparseCNN(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_blocks: int, hidden_channels: int, sparsity: float, act: str, layernorm: bool):
        super().__init__()

        act_fn = torch.nn.ReLU if act == "relu" else torch.nn.Tanh
        norm_fn = lambda c: torch.nn.BatchNorm2d(c) if layernorm else torch.nn.Identity()

        layers = []

        # initial convolutional block
        layers.append(MaskedLayer(torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1), sparsity))
        layers.append(norm_fn(hidden_channels))
        layers.append(act_fn())
        layers.append(torch.nn.MaxPool2d(2))

        # hidden convolutional blocks
        for _ in range(num_blocks - 1):
            layers.append(MaskedLayer(torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1), sparsity))
            layers.append(norm_fn(hidden_channels))
            layers.append(act_fn())
            layers.append(torch.nn.MaxPool2d(2))

        # fully connected layer before output
        layers.append(torch.nn.AdaptiveAvgPool2d(1))
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
        layers.append(act_fn())
        layers.append(torch.nn.Linear(hidden_channels, num_classes))

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

    def forward(self, x):
        return self.model(x)
