import src.network.detail
import torch

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

def estimate_amortized_flops(model: torch.nn.Module, warmup_strategy: src.rules.warmup.Warmup, input_shape: torch.Size, minibatch_budget: int) -> float:
    dense_flop_estimate = 0
    sparse_flop_estimate = 0

    for current_layer in model.model:
        # we are interested only in custom MaskedLayers
        if not isinstance(current_layer, src.network.detail.MaskedLayer):
            continue

        # update estimates for fully sparse and dense networks
        _1, _2 = current_layer.estimate_flops(input_shape)
        dense_flop_estimate += _1
        sparse_flop_estimate += _2

    return warmup_strategy.amortized_flops(dense_flop_estimate, sparse_flop_estimate, minibatch_budget)
