from typing import List, Dict, Tuple
import graph_utilities.operations

import src.rules.warmup
import src.rules.pruning
import src.rules.regrowth
import src.utilities

import torch
import numpy

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

    # iterate over all layers
    if pruning_strategies is not None and regrowth_strategies is not None:
        for layer in model.modules():
            # we are interested only in custom MaskedLayers
            if not isinstance(layer, src.network.detail.MaskedLayer):
                continue

            # apply removal/regrowth only for layers that actually require it
            if layer.realized_sparsity() < 1.0:
                for (prunning_strategy, regrowth_strategy) in zip(pruning_strategies, regrowth_strategies):
                    pruned_count, pruned_layer = prunning_strategy.apply(layer)
                    regrowth_strategy.apply(pruned_count, pruned_layer)
                    # not assigning back should be safe as we deal with references

            layer.reset_flow()  # set flow to zero

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
        "diversity": 0,
        "unique_batches": 0,
    }

def reset_tracked_fields(model: torch.nn.Module) -> torch.nn.Module:
    model.composite_loss = []
    model.median_loss = []
    model.mean_loss = []
    model.std_loss = []
    model.neuron_dormancy = []
    return model

def print_reachability(model: torch.nn.Module) -> None:
    modules = [module for module in model.modules() if isinstance(module, src.network.detail.MaskedLayer)]
    edge_reachabilities = []

    for module in modules:
        forward_reachability = torch.sum(module.forward_reachability * module.mask != 0).item() / module.active_weights()
        backward_reachability = torch.sum(module.backward_reachability * module.mask != 0).item() / module.active_weights()
        functional_reachability = torch.sum((module.forward_reachability.to(torch.bool) & module.backward_reachability.to(torch.bool)) * module.mask != 0).item() / module.active_weights()
        edge_reachabilities.append((forward_reachability, backward_reachability, functional_reachability))

    print(edge_reachabilities)

# instantiates network with all statistics to track
def spawn_network(model_factory, optimizer_factory, lr_decay_factory, parent: torch.nn.Module, warmup_strategy: src.rules.warmup.Warmup, pruning_strategies: List[src.rules.pruning.Pruning], regrowth_strategies: List[src.rules.regrowth.Regrowth], train_loader: torch.utils.data.Dataset, minibatch_budget: int, device: torch.device) -> torch.nn.Module:
    model = model_factory().to(device)

    if parent is not None:
        model.load_state_dict(parent.state_dict(), strict=False)

    with torch.no_grad():
        model.compute_reachabilities()
    model = update_mask(model, warmup_strategy, pruning_strategies, regrowth_strategies)
    with torch.no_grad():
        model.compute_reachabilities()
        print_reachability(model)

    model.pruned_fraction = sum([pruning_strategy.pruned_fraction() for pruning_strategy in pruning_strategies]) if pruning_strategies is not None else 0.0
    model = reset_tracked_fields(model)

    # fields required to estimate (amortized) FLOPs of operations (slightly adjusted "Rigging the Lottery: Making All Tickets Winners" logic as original idea wouldn't fit here)
    model.amortized_sample_cost = src.utilities.estimate_amortized_flops(model, warmup_strategy, train_loader, minibatch_budget)
    model.seen_batches = 0

    model.train()

    # fields related to optimization state
    model.optimizer = optimizer_factory(model)

    if parent is not None:
        model.optimizer.load_state_dict(parent.optimizer.state_dict())

    model.lr_decay = lr_decay_factory(model.optimizer)

    return model

def clamp(x, lo, hi):
    return max(lo, min(x, hi))

# helper for running forward/backward pass that computes all scores
def forward_and_backward_with_statistics(models: List[torch.nn.Module], criterion: torch.nn.Module, input: torch.Tensor, targets: torch.Tensor, statistics: Dict, epoch: int, batch: int) -> Tuple[torch.nn.Module, Dict]:

    logits = []
    for model in models:
        statistics["total_batches"] += 1

        outputs = model(input)
        loss = criterion(outputs, targets) # get loss

        # model.optimizer.zero_grad() # zero gradients
        # loss.mean().backward(retain_graph=True) # compute gradients but DO NOT DROP computational graph - we need it for flow estimation
        # for layer in model.modules():
        #     if isinstance(layer, src.network.detail.MaskedLayer):
        #         print(f"pure_loss: {layer.layer.weight.grad * layer.mask}, {torch.min(layer.layer.weight.grad * layer.mask)}, {torch.max(layer.layer.weight.grad * layer.mask)}")
        #         break

        task_loss = loss.mean()

        model.optimizer.zero_grad() # zero gradients
        task_loss.backward()

        # for layer in model.modules():
        #     if isinstance(layer, src.network.detail.MaskedLayer):
        #         print(f"task_loss: {layer.layer.weight.grad * layer.mask}, {torch.min(layer.layer.weight.grad * layer.mask)}, {torch.max(layer.layer.weight.grad * layer.mask)}")
        #         break

        # store stats
        with torch.no_grad():
            model.composite_loss.append(task_loss.item())
            model.median_loss.append(loss.quantile(q=0.5).item())
            model.mean_loss.append(loss.mean().item())
            model.std_loss.append(loss.std().item())
            model.seen_batches += 1
            model.accumulate_flow()

        model.optimizer.step()
        logits.append(outputs)

    with torch.no_grad():
        statistics["diversity"] += compute_diversity(torch.stack(logits, dim=1), targets)
        statistics["unique_batches"] += 1

    return models, statistics

# Theorem 1 - "Diversity and Generalization in Neural Network Ensembles"
def compute_diversity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    batch_count, model_count, class_count = logits.shape
    probabilities = torch.softmax(logits, dim=-1)

    true_class_probabilities = probabilities[
        torch.arange(batch_count, device=logits.device)[:, None],
        torch.arange(model_count, device=logits.device)[None, :],
        targets[:, None]
    ]

    normalization = true_class_probabilities.max(dim=1, keepdim=True).values
    p_tilde = true_class_probabilities / (torch.sqrt(torch.tensor(2.0)) * normalization)

    mean = p_tilde.mean(dim=1)
    var = (p_tilde ** 2).mean(dim=1) - mean ** 2

    return var.mean().cpu().item()

# helper for generating population level statistics
def generate_log_entry(statistics: Dict, children: List[torch.nn.Module], train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, device: torch.device) -> dict:

    # create log entry
    log_entry = {
        # global info
        "epoch": statistics["current_epoch"],
        "total_time": statistics["total_time"],
        "epoch_time": statistics["epoch_end"] - statistics["epoch_start"],
        "total_batches": statistics["total_batches"],
        "amortized_flops": statistics["amortized_flops"] + numpy.sum([child.amortized_sample_cost * train_loader.batch_size * child.seen_batches for child in children]),
        "diversity": statistics["diversity"] / statistics["unique_batches"],

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
