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

    # TODO split into modules with coherent API
    def random_mask_update(self, zeta=0.3):
        for layer in self.model:
            if not isinstance(layer, MaskedLayer):
                continue

            weight = layer.layer.weight.data
            mask = layer.mask.data

            flat_weight = weight.view(-1)
            flat_mask = mask.view(-1)

            active_indices = (flat_mask == 1).nonzero(as_tuple=True)[0]
            active_weights = flat_weight[active_indices]

            if active_weights.numel() == 0:
                continue

            # get magnitudes and sort from strongest to weakest
            magnitudes = active_weights.abs()
            sorted_magnitudes, sorted_indices = torch.sort(magnitudes, descending=True)

            # linearly assign mutation probabilities: 0 (strongest) to zeta (weakest)
            probs = torch.linspace(0, zeta, steps=len(sorted_indices))

            # get actual indices in flat_weight corresponding to sorted order
            mutation_candidates = active_indices[sorted_indices]

            # draw random values to determine which neurons to mutate
            random_vals = torch.rand_like(probs)
            to_mutate = random_vals < probs

            if to_mutate.sum() == 0:
                continue

            mutate_indices = mutation_candidates[to_mutate]
            flat_mask[mutate_indices] = 0  # prune

            # grow same number of weights randomly from inactive ones
            inactive_indices = (flat_mask == 0).nonzero(as_tuple=True)[0]
            num_to_grow = to_mutate.sum().item()
            if len(inactive_indices) > 0:
                grow_indices = inactive_indices[torch.randperm(len(inactive_indices))[:num_to_grow]]
                flat_mask[grow_indices] = 1

            # reinitialize the active weights with Gaussian noise scaled by their norm
            updated_active_weights = flat_weight[flat_mask.bool()]
            norm = updated_active_weights.norm() if updated_active_weights.numel() > 0 else 1.0
            torch.nn.init.normal_(flat_weight[flat_mask.bool()], std=norm / updated_active_weights.numel() ** 0.5)

            # update the mask
            layer.mask.data = flat_mask.view_as(layer.mask)

    # TODO split into modules with coherent API
    def set_mask_update(self, zeta=0.3):
        for layer in self.model:
            if not isinstance(layer, MaskedLayer):
                continue

            weight = layer.layer.weight.data
            mask = layer.mask.data

            flat_weight = weight.view(-1)
            flat_mask = mask.view(-1)

            active_weights = flat_weight[flat_mask.bool()]
            k = int(zeta * active_weights.numel())
            if k == 0: continue

            threshold = active_weights.abs().topk(k, largest=False).values.max()
            prune_mask = (flat_weight.abs() <= threshold) & (flat_mask == 1)
            flat_mask[prune_mask] = 0

            inactive_indices = (flat_mask == 0).nonzero(as_tuple=True)[0]
            if len(inactive_indices) > 0:
                grow_indices = inactive_indices[torch.randperm(len(inactive_indices))[:k]]
                flat_mask[grow_indices] = 1

            # calculate the norm of the active weights
            norm_active_weights = active_weights.norm() if active_weights.numel() > 0 else 1.0

            # reinitialize the active weights with random values drawn from a Gaussian distribution
            # and scaled by the norm of the active weights
            torch.nn.init.normal_(flat_weight[flat_mask.bool()], std=norm_active_weights / active_weights.numel() ** 0.5)

            # update the mask data
            layer.mask.data = flat_mask.view_as(layer.mask)

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

    # TODO split into modules with coherent API
    def set_mask_update(self, zeta=0.3):
        for layer in self.model:
            if not isinstance(layer, MaskedLayer):
                continue

            weight = layer.layer.weight.data
            mask = layer.mask.data

            flat_weight = weight.view(-1)
            flat_mask = mask.view(-1)

            active_weights = flat_weight[flat_mask.bool()]
            k = int(zeta * active_weights.numel())
            if k == 0: continue

            threshold = active_weights.abs().topk(k, largest=False).values.max()
            prune_mask = (flat_weight.abs() <= threshold) & (flat_mask == 1)
            flat_mask[prune_mask] = 0

            inactive_indices = (flat_mask == 0).nonzero(as_tuple=True)[0]
            if len(inactive_indices) > 0:
                grow_indices = inactive_indices[torch.randperm(len(inactive_indices))[:k]]
                flat_mask[grow_indices] = 1

            # calculate the norm of the active weights
            norm_active_weights = active_weights.norm() if active_weights.numel() > 0 else 1.0

            # reinitialize the active weights with random values drawn from a Gaussian distribution
            # and scaled by the norm of the active weights
            torch.nn.init.normal_(flat_weight[flat_mask.bool()], std=norm_active_weights / active_weights.numel() ** 0.5)

            # update the mask data
            layer.mask.data = flat_mask.view_as(layer.mask)

    def forward(self, x):
        return self.model(x)
