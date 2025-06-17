import torch

class MaskedLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, sparsity: float = 0.8):
        super().__init__()
        assert isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)), "MaskedLayer supports Linear or Conv2d only"
        self.layer = layer

        # Create mask with given sparsity
        mask = torch.zeros_like(layer.weight)
        k = int((1 - sparsity) * mask.numel())
        mask.view(-1)[torch.randperm(mask.numel())[:k]] = 1
        self.mask = torch.nn.Parameter(mask, requires_grad=False)

    def forward(self, x):
        masked_weight = self.layer.weight * self.mask
        if isinstance(self.layer, torch.nn.Linear): # there are much better ways to do this but it works for now
            return torch.nn.functional.linear(x, masked_weight, self.layer.bias)
        elif isinstance(self.layer, torch.nn.Conv2d):
            return torch.nn.functional.conv2d(x, masked_weight, self.layer.bias, stride=self.layer.stride, padding=self.layer.padding, dilation=self.layer.dilation, groups=self.layer.groups)
        raise RuntimeError(f"unsupported layer type: {type(self.layer)} in MaskedLayer! ")

class SparseANN(torch.nn.Module):
    def __init__(self, in_features: int = 784, hidden_dim: int = 250, num_hidden_layers: int = 3, sparsity: float = 0.8, act: str = "tanh", layernorm: bool = True):
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
        layers.append(torch.nn.Linear(hidden_dim, 10))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class SparseCNN(torch.nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, num_blocks: int = 5, hidden_channels: int = 128, sparsity: float = 0.8, act: str = "relu", layernorm: bool = True):
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

    def forward(self, x):
        return self.model(x)
