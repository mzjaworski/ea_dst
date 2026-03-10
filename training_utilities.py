from typing import List, Dict, Tuple

import src.rules.warmup
import src.rules.pruning
import src.rules.regrowth
import src.utilities

import torch
import numpy
import math

def top_k_accuracy(model: torch.nn.Module, testloader: torch.utils.data.DataLoader, device: torch.device, top_k: list[int]) -> dict[str, float]:
    correct = {k: 0 for k in top_k}
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            num_classes = outputs.size(1)

            max_k = min(max(top_k), num_classes)
            _, preds = outputs.topk(max_k, dim=1)
            preds = preds.t()

            matches = preds.eq(targets.view(1, -1))

            for k in top_k:
                if k > num_classes:
                    correct[k] += 0
                else:
                    correct[k] += matches[:k].any(dim=0).sum().item()

            total += targets.size(0)

    return {f"acc-top_{k}": 100.0 * correct[k] / total for k in top_k}

def update_mask(model: torch.nn.Module, pruning_strategies: List[src.rules.pruning.Pruning], regrowth_strategies: List[src.rules.regrowth.Regrowth]) -> torch.nn.Module:

    with torch.no_grad():
        model.compute_reachabilities()

    # iterate over all layers
    if pruning_strategies is not None and regrowth_strategies is not None:
        modules = list([module for module in model.modules() if isinstance(module, src.network.detail.MaskedLayer)])
        reversed_order = bool(numpy.random.randint(0, 2))

        for layer in (reversed(modules) if reversed_order else modules):

            # apply removal/regrowth only for layers that actually require it
            if layer.realized_sparsity() < 1.0:

                # draw random strategy from the supplied list
                pruning_index = numpy.random.randint(0, len(pruning_strategies))
                regrowth_index = numpy.random.randint(0, len(regrowth_strategies))

                # remove weights
                pruned_count, pruned_layer = pruning_strategies[pruning_index].apply(layer)

                # zero out momentum for inactive weights - this way regrown weights do not inherit parents momentum state
                param = layer.layer.weight
                if param in model.optimizer.state:
                    state = model.optimizer.state[param]

                    if state is None or 'momentum_buffer' not in state:
                        continue

                    buffer = state['momentum_buffer'].view(-1)
                    buffer[(~layer.mask.bool()).view(-1).nonzero(as_tuple=True)[0]] = 0
                    state['momentum_buffer'] = buffer.view_as(param)

                # regrow weights
                regrowth_strategies[regrowth_index].apply(pruned_layer, pruned_count)
                # not assigning back should be safe as we deal with references

    with torch.no_grad():
        model.compute_reachabilities()

    return model

# redistributes sparsity across layers, also recomputes reachabilities after adjusting every layer
def update_mask_redistribute(model: torch.nn.Module, warmup_strategy: src.rules.warmup.Warmup, pruning_strategies: List[src.rules.pruning.Pruning], regrowth_strategies: List[src.rules.regrowth.Regrowth], noise: float = 0.05) -> torch.nn.Module:
    model = warmup_strategy.apply(model)

    with torch.no_grad():
        model.compute_reachabilities()

    # iterate over all layers
    if pruning_strategies is not None and regrowth_strategies is not None:
        modules = [module for module in model.modules() if isinstance(module, src.network.detail.MaskedLayer)]
        reversed_order = bool(numpy.random.randint(0, 2))
        modules = list(reversed(modules)) if reversed_order else modules

        layer_param_counts = numpy.array([layer.total_weights() for layer in modules], dtype=int)
        current_active_counts = numpy.array([int(layer.active_weights()) for layer in modules], dtype=int)

        layer_factors = numpy.random.uniform(1.0 - noise, 1.0 + noise, size=len(modules))
        perturbed_float = current_active_counts.astype(float) * layer_factors

        total_target = model.global_weight_target
        scale = total_target / perturbed_float.sum()
        perturbed_float *= scale

        new_active_counts = numpy.floor(perturbed_float).astype(int)
        remainder = perturbed_float - new_active_counts
        diff = total_target - new_active_counts.sum()

        if diff > 0:
            order = numpy.argsort(-remainder)
            for i in order[:diff]:
                new_active_counts[i] += 1
        elif diff < 0:
            order = numpy.argsort(remainder)
            for i in order[:abs(diff)]:
                new_active_counts[i] -= 1

        perturbed_densities = new_active_counts / layer_param_counts
        perturbed_densities = numpy.clip(perturbed_densities, 0.0, 1.0)

        for layer_index, layer in enumerate(modules):

            if layer.realized_sparsity() < 1.0:

                pruning_index = numpy.random.randint(0, len(pruning_strategies))
                regrowth_index = numpy.random.randint(0, len(regrowth_strategies))

                pruning_strategy = pruning_strategies[pruning_index]
                regrow_strategy = regrowth_strategies[regrowth_index]

                current_target = perturbed_densities[layer_index]
                current_density = 1.0 - layer.realized_sparsity()

                pruned_count, pruned_layer = pruning_strategy.apply(layer)
                total_weights = layer.total_weights()

                delta_density = current_target - current_density
                adjusted_prune = int(total_weights * delta_density)

                pruned_count += adjusted_prune

                # sometimes we should have pruned more than we did, lets fix it
                if pruned_count < 0:
                    _, pruned_layer = pruning_strategies[pruning_index].apply(layer, prune_count=abs(pruned_count))
                    pruned_count = 0

                # zero out momentum for inactive weights - this way regrown weights do not inherit parents momentum state
                param = layer.layer.weight
                if param in model.optimizer.state:
                    state = model.optimizer.state[param]

                    if state is None or 'momentum_buffer' not in state:
                        continue

                    buffer = state['momentum_buffer'].view(-1)
                    buffer[(~layer.mask.bool()).view(-1).nonzero(as_tuple=True)[0]] = 0
                    state['momentum_buffer'] = buffer.view_as(param)

                if pruned_count > 0:
                    regrow_strategy.apply(pruned_layer, pruned_count)
                    # not assigning back should be safe as we deal with references

    with torch.no_grad():
        model.compute_reachabilities()

    return model

# some statistics have to be tracked globally
def initialize_statistics() -> Dict:
    return {
        "amortized_flops": 0.0,
        "current_epoch": 0,
        "epoch_start": 0,
        "epoch_end": 0,
        "total_time": 0,
        "total_batches": 0,
        "prediction_diversity": 0,
        "unique_batches": 0
    }

def reset_tracked_fields(model: torch.nn.Module) -> torch.nn.Module:
    model.composite_loss = []
    model.median_loss = []
    model.mean_loss = []
    model.std_loss = []
    model.neuron_dormancy = []
    return model

def edge_reachability(model: torch.nn.Module) -> List[Tuple[float, float, float]]:
    modules = [module for module in model.modules() if isinstance(module, src.network.detail.MaskedLayer)]
    edge_reachabilities = []

    for module in modules:
        forward_reachability = torch.sum(module.forward_reachability * module.mask != 0).item() / module.active_weights() if module.active_weights() > 0 else 0
        backward_reachability = torch.sum(module.backward_reachability * module.mask != 0).item() / module.active_weights() if module.active_weights() > 0 else 0
        functional_reachability = torch.sum((module.forward_reachability.to(torch.bool) & module.backward_reachability.to(torch.bool)) * module.mask != 0).item() / module.active_weights() if module.active_weights() > 0 else 0
        edge_reachabilities.append((forward_reachability, backward_reachability, functional_reachability))

    return edge_reachabilities

# instantiates network with all statistics to track
def spawn_network(model_factory, optimizer_factory, lr_decay_factory, parent: torch.nn.Module, warmup_strategy: src.rules.warmup.Warmup, pruning_strategies: List[src.rules.pruning.Pruning], regrowth_strategies: List[src.rules.regrowth.Regrowth], train_loader: torch.utils.data.Dataset, minibatch_budget: int, device: torch.device, cycles: int, redistribute: bool = False, noise: float = 0.05) -> torch.nn.Module:
    model = model_factory().to(device)
    model.optimizer = optimizer_factory(model)
    model.lr_decay = lr_decay_factory(model.optimizer)

    if parent is not None:
        model.global_weight_target = parent.global_weight_target
        model.load_state_dict(parent.state_dict(), strict=False)
        model.optimizer.load_state_dict(parent.optimizer.state_dict())
        model.lr_decay.load_state_dict(parent.lr_decay.state_dict())

        # copy over parent gradients if any -
        for p, q in zip(model.parameters(), parent.parameters()):
            if q.grad is not None:
                p.grad = q.grad.clone()
    else:
        # if no parent given fallback to zero gradients
        for p in model.parameters():
            p.grad = torch.zeros_like(p)

    model = reset_tracked_fields(model)

    for cycle in range(cycles):
        if redistribute:
            model = update_mask_redistribute(model, warmup_strategy, pruning_strategies, regrowth_strategies, noise)
        else:
            model = update_mask(model, pruning_strategies, regrowth_strategies)

    model.pruned_fraction = sum([pruning_strategy.pruned_fraction() for pruning_strategy in pruning_strategies]) if pruning_strategies is not None else 0.0

    # fields required to estimate (amortized) FLOPs of operations (slightly adjusted "Rigging the Lottery: Making All Tickets Winners" logic as original idea wouldn't fit here)
    model.amortized_sample_cost = src.utilities.estimate_amortized_flops(model, warmup_strategy, train_loader, minibatch_budget)
    model.constant_cost = warmup_strategy.constant_cost()
    model.seen_batches = 0

    model.train()

    return model

# helper for running forward/backward pass that computes all scores
def forward_and_backward_with_statistics(models: List[torch.nn.Module], criterion: torch.nn.Module, input: torch.Tensor, targets: torch.Tensor, statistics: Dict) -> Tuple[List[torch.nn.Module], Dict]:
    logits = []
    for model in models:
        statistics["total_batches"] += 1

        outputs = model(input)
        loss = criterion(outputs, targets) # get loss
        task_loss = loss.mean()
        model.optimizer.zero_grad() # zero gradients
        task_loss.backward()

        # store stats
        with torch.no_grad():
            model.median_loss.append(loss.quantile(q=0.5).item())
            model.mean_loss.append(loss.mean().item())
            model.std_loss.append(loss.std().item())
            model.seen_batches += 1

        model.optimizer.step()
        logits.append(outputs.detach())

    with torch.no_grad():
        statistics["prediction_diversity"] += compute_diversity(torch.stack(logits, dim=1), targets)
        statistics["unique_batches"] += 1

    return models, statistics

# Theorem 1 - "Diversity and Generalization in Neural Network Ensembles"; cross‑entropy variant - adjusted to work with mixup
def compute_diversity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probabilities = torch.softmax(logits, dim=-1)

    if targets.ndim == 1: # no mixup
        batch_count, model_count, class_count = logits.shape
        true_class_probabilities = probabilities[
            torch.arange(batch_count, device=logits.device)[:, None],
            torch.arange(model_count, device=logits.device)[None, :],
            targets[:, None]
        ]
    else: # with mixup
        true_class_probabilities = (probabilities * targets[:, None, :]).sum(dim=-1)

    normalization = true_class_probabilities.max(dim=1, keepdim=True).values
    p_tilde = true_class_probabilities / (torch.sqrt(torch.tensor(2.0)) * normalization)

    mean = p_tilde.mean(dim=1)
    var = (p_tilde ** 2).mean(dim=1) - mean ** 2

    return var.mean().cpu().item()

# helper for generating population level statistics
def generate_log_entry(statistics: Dict, children: List[torch.nn.Module], train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, device: torch.device) -> dict:
    loss = numpy.array([c.mean_loss for c in children])
    relative_performance = (loss / (numpy.mean(loss, axis=0) + 1e-12)).mean(axis=1)

    # create log entry
    log_entry = {
        # global info
        "epoch": statistics["current_epoch"],
        "total_time": statistics["total_time"],
        "epoch_time": statistics["epoch_end"] - statistics["epoch_start"],
        "total_batches": statistics["total_batches"],
        "amortized_flops": statistics["amortized_flops"] + numpy.sum([(child.amortized_sample_cost * train_loader.batch_size * child.seen_batches) + child.constant_cost for child in children]),
        "prediction_diversity": statistics["prediction_diversity"] / statistics["unique_batches"],

        # best member statistics
        "current_best": children[0].index,
        "fitness_best": children[0].fitness,
        "median_loss": numpy.mean(children[0].median_loss),
        "mean_loss": numpy.mean(children[0].mean_loss),
        "std_loss": numpy.mean(children[0].std_loss),

        # population statistics
        "p_mean_fitness": numpy.mean([child.fitness for child in children]),
        "p_std_fitness": numpy.std([child.fitness for child in children]),

        "p_m_median_loss": numpy.mean([numpy.mean(child.median_loss) for child in children]),
        "p_m_mean_loss": numpy.mean([numpy.mean(child.mean_loss) for child in children]),
        "p_m_std_loss": numpy.mean([numpy.mean(child.std_loss) for child in children]),

        # relative performance measures
        "minimum_rp": relative_performance.min(), # rp stands for relative performance
        "maximum_rp": relative_performance.max(),
        "spread_rp": relative_performance.max() - relative_performance.min(),

        # miscellaneous
        "realized_sparsity": children[0].realized_sparsity(),
        "active_weights": children[0].active_weights(),
        "total_weights": children[0].total_weights(),
        "learning_rate": children[0].optimizer.param_groups[0]["lr"],
        "pruned_fraction": children[0].pruned_fraction,
    }

    # NOTE: this assumes children are already sorted using their fitness!
    children[0].eval()
    for (level, value) in top_k_accuracy(children[0], test_loader, device, [1, 3, 5]).items():
        log_entry[level] = value
    children[0].train()

    return log_entry
