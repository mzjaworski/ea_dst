import torch
import numpy

def uniform(weights: torch.Tensor, sparsity: float) -> torch.Tensor:
    mask = torch.zeros_like(weights)
    k = int((1 - sparsity) * mask.numel())
    mask.view(-1)[torch.randperm(mask.numel())[:k]] = 1
    return mask

def erk(weights: torch.Tensor, epsilon: float) -> torch.Tensor:
    shape = weights.shape

    if weights.ndim == 2:
        n_out, n_in = shape
    elif weights.ndim == 4:
        n_out, n_in, kH, kW = shape
        n_in = n_in * kH * kW
    else:
        raise ValueError(f"Unsupported weight shape: {shape}")

    p = epsilon * (n_in + n_out) / (n_in * n_out)
    p = min(p, 1.0)

    return (torch.rand_like(weights) < p).float()

def compute_erk_epsilon(layers, target_sparsity, erk_power: float = 1.0):
    raw_probs = [(numpy.sum(shape) / numpy.prod(shape)) ** erk_power for shape in layers]
    dense_layers = set()

    while True:
        rhs = 0.0
        divisor = 0.0
        for i, shape in enumerate(layers):
            n_param = numpy.prod(shape)
            if i in dense_layers:
                continue
            n_ones = (1 - target_sparsity) * n_param
            rhs += n_ones
            divisor += raw_probs[i] * n_param

        epsilon = rhs / divisor

        # Check for layers exceeding probability 1
        max_prob = max(raw_probs[i] for i in range(len(layers)) if i not in dense_layers)
        if epsilon * max_prob > 1.0:
            for i in range(len(layers)):
                if raw_probs[i] == max_prob:
                    dense_layers.add(i)
        else:
            break

    return epsilon
