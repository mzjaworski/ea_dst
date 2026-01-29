from typing import List, Tuple
import torch
import numpy

def uniform(weights: torch.Tensor, sparsity: float) -> torch.Tensor:
    mask = torch.zeros_like(weights)
    k = int((1 - sparsity) * mask.numel())
    mask.view(-1)[torch.randperm(mask.numel())[:k]] = 1
    return mask

def erk(weights: torch.Tensor, fan_out_fan_in: Tuple[int, int], epsilon: float) -> torch.Tensor:
    n_out, n_in = fan_out_fan_in
    probability = min(epsilon * (n_in + n_out) / (n_in * n_out), 1.0)
    return (torch.rand_like(weights) < probability).float()

# based/adapted of/from https://github.com/alooow/fantastic_weights_paper/blob/565c21f51fe977efda7def26276b3b7e554c8f91/sparselearning/sparsity_initializers.py#L37
def compute_erk_initialization(layers: List[Tuple[int, int]], target_sparsity: float, erk_power: float = 1.0) -> Tuple[List[bool], float]:
    target_density = 1.0 - target_sparsity
    is_epsilon_valid = False
    dense_layers = set()

    while not is_epsilon_valid:

        raw_probabilities = {}
        divisor = 0.0
        rhs = 0.0

        for idx, (fan_out, fan_in) in enumerate(layers):
            n_param = fan_out * fan_in
            n_zeros = n_param * (1.0 - target_density)
            n_ones = n_param * target_density

            if idx in dense_layers:
                rhs -= n_zeros
            else:
                rhs += n_ones
                raw_probabilities[idx] = ((fan_in + fan_out) / n_param) ** erk_power
                divisor += raw_probabilities[idx] * n_param

        epsilon = rhs / divisor

        # check if any layer exceeds density=1 if yes mark it as dense
        max_prob = max(raw_probabilities.values(), default=0)
        if max_prob * epsilon > 1.0:
            is_epsilon_valid = False
            for idx, prob in raw_probabilities.items():
                if prob == max_prob:
                    dense_layers.add(idx)
        else:
            is_epsilon_valid = True

    is_dense = [idx in dense_layers for idx in range(len(layers))]

    return is_dense, epsilon
