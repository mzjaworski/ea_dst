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
