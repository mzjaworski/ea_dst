from typing import Tuple, List

import src.network.detail
import src.rules.warmup
import torch
import numpy


def soft_top_k(data: torch.Tensor, select: int = 1, decay_rate: float = 1.5) -> torch.BoolTensor:
    data_count = data.numel()

    if decay_rate is None or data_count <= 1:
        mask = torch.zeros(data_count, dtype=torch.bool, device=data.device)
        mask[:min(select, data_count)] = True
        return mask

    indices = torch.arange(data_count, device=data.device, dtype=torch.float32)
    percentiles = indices / (data_count - 1)

    weights = torch.exp(-decay_rate * percentiles)
    probs = weights / weights.sum()

    selected = torch.multinomial(probs, num_samples=min(select, data_count), replacement=False)

    mask = torch.zeros(data_count, dtype=torch.bool, device=data.device)
    mask[selected] = True

    return mask

def estimate_neuron_count(features: int, classes: int, network_depth: int, parameter_target: int) -> int:

    if network_depth < 2:
        raise RuntimeError("network_depth must be set to at least 2")

    def network_size(candidate: int) -> int:
        total = features * candidate + candidate
        total += (network_depth - 2) * (candidate * candidate + candidate)
        total += candidate * classes + classes
        return total

    neurons = 1
    while network_size(neurons) < parameter_target:
        neurons += 1

    return neurons

def estimate_amortized_flops(model: torch.nn.Module, warmup_strategy: src.rules.warmup.Warmup, data_loader: torch.utils.data.DataLoader, minibatch_budget: int) -> float:
    device = next(model.parameters()).device
    input_tensor = next(iter(data_loader))[0].to(device)

    dense_flop_estimate = 0
    sparse_flop_estimate = 0

    # here we esitmate FLOPs via registered hooks and (unfortunately) one forward pass to trigger them - which is NOT counted into the total
    def estimate_flops(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        nonlocal dense_flop_estimate, sparse_flop_estimate
        _1, _2 = module.estimate_flops(input[0])
        dense_flop_estimate += _1
        sparse_flop_estimate += _2

    # register hooks
    handles = [m.register_forward_hook(estimate_flops) for m in model.modules() if isinstance(m, src.network.detail.MaskedLayer)]

    # trigger hooks
    with torch.no_grad():
        _ = model(input_tensor)

    # remove hooks
    for h in handles:
        h.remove()

    return warmup_strategy.amortized_flops(dense_flop_estimate, sparse_flop_estimate, minibatch_budget)

def extract_gradients(model: torch.nn.Module) -> numpy.ndarray:
    gradients = []
    for current_layer in model.modules():
        if not isinstance(current_layer, src.network.detail.MaskedLayer):
            continue

        if current_layer.layer.weight.grad is not None:
            gradients.append(current_layer.layer.weight.grad[current_layer.mask == 1].flatten())
    return torch.cat(gradients).cpu().numpy()



# TODO function creating log entry
