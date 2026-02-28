from typing import List, Dict, Tuple

import src.rules.warmup
import src.rules.pruning
import src.rules.regrowth
import src.utilities

import torch
import numpy
import math

def top_k_accuracy(model: torch.nn.Module, testloader: torch.utils.data.DataLoader, device: torch.device, top_k: list[int]) -> dict[str, float]:
    max_k = max(top_k)

    correct = {k: 0 for k in top_k}
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, preds = outputs.topk(max_k, dim=1)
            preds = preds.t()

            matches = preds.eq(targets.view(1, -1))

            for k in top_k:
                correct[k] += matches[:k].any(dim=0).sum().item()

            total += targets.size(0)

    return {f"acc-top_{k}": 100.0 * correct[k] / total for k in top_k}

# updates network topology according to predefined set of rules
def update_mask(model: torch.nn.Module, warmup_strategy: src.rules.warmup.Warmup, pruning_strategies: List[src.rules.pruning.Pruning], regrowth_strategies: List[src.rules.regrowth.Regrowth]) -> torch.nn.Module:
    model = warmup_strategy.apply(model)

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

# some statistics have to be tracked globally
def initialize_statistics() -> Dict:
    return {
        "amortized_flops": 0.0,
        "current_epoch": 0,
        "epoch_start": 0,
        "epoch_end": 0,
        "total_time": 0,
        "total_batches": 0
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
        forward_reachability = torch.sum(module.forward_reachability * module.mask != 0).item() / module.active_weights()
        backward_reachability = torch.sum(module.backward_reachability * module.mask != 0).item() / module.active_weights()
        functional_reachability = torch.sum((module.forward_reachability.to(torch.bool) & module.backward_reachability.to(torch.bool)) * module.mask != 0).item() / module.active_weights()
        edge_reachabilities.append((forward_reachability, backward_reachability, functional_reachability))

    return edge_reachabilities

# instantiates network with all statistics to track
def spawn_network(model_factory, optimizer_factory, lr_decay_factory, parent: torch.nn.Module, warmup_strategy: src.rules.warmup.Warmup, pruning_strategies: List[src.rules.pruning.Pruning], regrowth_strategies: List[src.rules.regrowth.Regrowth], train_loader: torch.utils.data.Dataset, minibatch_budget: int, device: torch.device, cycles: int) -> torch.nn.Module:
    model = model_factory().to(device)
    model.optimizer = optimizer_factory(model)
    model.lr_decay = lr_decay_factory(model.optimizer)

    if parent is not None:
        model.load_state_dict(parent.state_dict(), strict=False)
        model.optimizer.load_state_dict(parent.optimizer.state_dict())
        model.lr_decay.load_state_dict(parent.lr_decay.state_dict())

    model = reset_tracked_fields(model)

    for cycle in range(cycles):
        model = update_mask(model, warmup_strategy, pruning_strategies, regrowth_strategies)

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
            model.composite_loss.append(task_loss.item())
            model.median_loss.append(loss.quantile(q=0.5).item())
            model.mean_loss.append(loss.mean().item())
            model.std_loss.append(loss.std().item())
            model.seen_batches += 1

        model.optimizer.step()
        logits.append(outputs)
    return models, statistics

# helper for generating population level statistics
def generate_log_entry(statistics: Dict, children: List[torch.nn.Module], train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, device: torch.device) -> dict:

    # create log entry
    log_entry = {
        # global info
        "epoch": statistics["current_epoch"],
        "total_time": statistics["total_time"],
        "epoch_time": statistics["epoch_end"] - statistics["epoch_start"],
        "total_batches": statistics["total_batches"],
        "amortized_flops": statistics["amortized_flops"] + numpy.sum([(child.amortized_sample_cost * train_loader.batch_size * child.seen_batches) + child.constant_cost for child in children]),

        # best member statistics
        "current_best": children[0].index,
        "median_loss": numpy.mean(children[0].median_loss),
        "mean_loss": numpy.mean(children[0].mean_loss),
        "std_loss": numpy.mean(children[0].std_loss),

        # population statistics
        "p_median_loss": numpy.mean([numpy.mean(child.median_loss) for child in children]),
        "p_mean_loss": numpy.mean([numpy.mean(child.mean_loss) for child in children]),
        "p_std_loss": numpy.mean([numpy.mean(child.std_loss) for child in children]),

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
