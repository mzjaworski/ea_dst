import training_utilities
import src.network.detail
import src.rules.warmup
import src.rules.regrowth
import src.rules.pruning
import src.rules.decay
import utilities
import settings

from typing import List, Tuple
import secrets
import torch
import numpy
import time
import json
import math
import argparse
import random
import sys

import graph_utilities.operations

def compute_reachability(model: torch.nn.Module) -> torch.nn.Module:
    graph, total_neurons = graph_utilities.operations.convert_to_graph(model)
    _, core_neurons = graph_utilities.operations.extract_functional_core(graph, total_neurons)
    model.reachability = float(numpy.mean([reachable / total for (reachable, total) in zip(core_neurons, total_neurons)]))
    return model

def stage_budgets(total_budget: int, num_children: int, keep: float) -> List[int]:
    S = int(math.floor(math.log(num_children, 1.0 / keep)))
    weights = [(1.0 / keep) ** s for s in range(S)]
    Z = sum(weights)

    batches = [int(total_budget * w / Z) for w in weights]
    batches[-1] += total_budget - sum(batches)  # fix rounding
    return batches

class StageScheduler:
    def __init__(self, total_budget: int, num_children: int, keep: float, alpha_range: Tuple[float, float] = (1.0, 1.0)):
        self.total_budget = total_budget
        self.num_children = num_children
        self.keep = keep

        S = int(math.floor(math.log(num_children, 1.0 / keep)))
        self.survivors = [max(1, int(round(num_children * (keep ** s)))) for s in range(S)]

        self.batches_per_stage = stage_budgets(total_budget, num_children, keep)
        self.alpha_range = numpy.linspace(alpha_range[0], alpha_range[1], S)

        self.stage_idx = 0
        self.consumed_in_stage = 0

    def __call__(self) -> Tuple[bool, int, float]: # phase_end, survivor_count, current_alpha
        if self.stage_idx >= len(self.survivors):
            return True, 1, self.alpha_range[-1]  # iteration finished

        self.consumed_in_stage += 1

        if self.consumed_in_stage >= self.batches_per_stage[self.stage_idx]:
            self.stage_idx += 1
            self.consumed_in_stage = 0
            if self.stage_idx >= len(self.survivors):
                return True, 1, self.alpha_range[-1]  # iteration finished

        return self.consumed_in_stage == 0, self.survivors[self.stage_idx], self.alpha_range[self.stage_idx - 1 if self.consumed_in_stage == 0 else self.stage_idx]

def soft_top_k(candidates: List, select: int, decay_rate: float) -> List:
    data_count = len(candidates)

    if decay_rate is None or data_count <= 1:
        return candidates[:select]

    candidate_indices = numpy.arange(data_count)
    percentiles = candidate_indices / (data_count - 1)

    weights = numpy.exp(-decay_rate * percentiles)
    probabilities = weights / weights.sum()

    selected_indices = numpy.random.choice(data_count, size=select, replace=False, p=probabilities)
    return [candidates[i] for i in selected_indices]


def collect_masks(model):
    mask_chunks = []

    for module in model.modules():
        if isinstance(module, src.network.detail.MaskedLayer):
            mask_chunks.append(module.mask.float().view(-1))

    return torch.cat(mask_chunks, dim=0)

# magnitude based ordering
def rank_models(candidates: List[torch.nn.Module], alpha: float = 1.0, eps: float = 1e-8) -> List[torch.nn.Module]:
    loss = numpy.array([c.composite_loss for c in candidates])
    diff = numpy.abs(numpy.diff(loss, axis=1))

    pop_loss = numpy.median(loss, axis=0)
    pop_diff = numpy.median(diff, axis=0)

    for i, c in enumerate(candidates):
        level = loss[i][1:] / (pop_loss[1:] + eps)
        slope = diff[i] / (pop_diff + eps)
        c.rank = float(numpy.mean(alpha * level + (1.0 - alpha) * slope))

    return candidates

# # rank based ordering
# def rank_models(candidates: List[torch.nn.Module], alpha: float = 1.0, eps: float = 1e-8) -> List[torch.nn.Module]:
#
#     if len(candidates) < 2 or len(candidates[0].composite_loss) < 2:
#         for c in candidates:
#             c.rank = 0.0
#         return candidates
#
#     mean_loss = numpy.array([c.composite_loss for c in candidates])
#     diff_loss = numpy.array([[1.0 - c.flow_entropy()] for c in candidates])#numpy.diff(mean_loss, axis=1)
#
#     # align loss with change
#     # mean_loss = mean_loss[:, 1:]
#
#     mean_rank = numpy.argsort(numpy.argsort(mean_loss, axis=0), axis=0)
#     diff_rank = numpy.argsort(numpy.argsort(diff_loss, axis=0), axis=0)
#
#     # normalize in [0, 1]
#     N = len(candidates)
#     mean_rank = mean_rank / (N - 1 + eps)
#     diff_rank = diff_rank / (N - 1 + eps)
#
#     # combine based on alpha
#     final_rank = alpha * mean_rank + (1.0 - alpha) * diff_rank
#
#     for i, c in enumerate(candidates):
#         c.rank = float(numpy.mean(final_rank[i]))
#
#     return candidates

def train_epoch(model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader,
                warmup_strategy: src.rules.warmup.Warmup, pruning_strategies: List[src.rules.pruning.Pruning], regrowth_strategies: List[src.rules.regrowth.Regrowth], decay_strategy: src.rules.decay.DecayStrategy,
                device: torch.device, epochs: int, offspring: int, minibatch_budget: int, decay_rate: float):
    statistics = training_utilities.initialize_statistics()

    fitness_function = lambda c: c.rank
    criterion = criterion_factory()
    alpha = 1.0

    children = []
    for current_epoch in range(1, epochs + 1):
        statistics["current_epoch"] += 1
        statistics["epoch_start"] = time.perf_counter()

        for pruning_strategy in pruning_strategies:
            pruning_strategy.decay_pruned_fraction(decay_strategy.update(0)) # update fraction

        # apply learning rate decay to children
        if len(children):
            for child in children:
                child.lr_decay.step()

        # train for one epoch
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # update population
            if (statistics["total_batches"] / offspring) % minibatch_budget == 0:
                best_model = None

                # pick best model
                if len(children):
                    children = rank_models(children, alpha=alpha)
                    children.sort(key=fitness_function)
                    print([(child.index, fitness_function(child), float(numpy.mean(child.mean_loss))) for child in children])
                    best_model = soft_top_k(children, select=1, decay_rate=decay_rate)[0]
                    statistics["amortized_flops"] += numpy.sum([child.amortized_sample_cost * train_loader.batch_size * child.seen_batches for child in children])
                    children = []

                # generate new offspring
                for child_index in range(1, offspring + 1):
                    if current_epoch == 1:
                        child = training_utilities.spawn_network(model_factory, optimizer_factory, lr_decay_factory, best_model, warmup_strategy, None, None, train_loader, minibatch_budget, device)
                    else:
                        child = training_utilities.spawn_network(model_factory, optimizer_factory, lr_decay_factory, best_model, warmup_strategy, pruning_strategies, regrowth_strategies, train_loader, minibatch_budget, device)
                    child.index = child_index
                    children.append(child)

            # train population
            children, statistics = training_utilities.forward_and_backward_with_statistics(children, criterion, inputs, targets, statistics, current_epoch, batch_index)

        # log epoch end time
        statistics["epoch_end"] = time.perf_counter()
        statistics["total_time"] += statistics["epoch_end"] - statistics["epoch_start"]

        # sort before logging
        children = rank_models(children, alpha=alpha)
        children.sort(key=fitness_function)

        # log statistics
        log_entry = training_utilities.generate_log_entry(statistics, children, train_loader, test_loader, device)
        settings.logger.log(log_entry)

        # if current_epoch == 1:
        #     graph, total_neurons = graph_utilities.operations.convert_to_graph(children[0])
        #     graph_utilities.operations.draw_layered_graph(graph, total_neurons, filename="graph_start.png")
        # if current_epoch == 50:
        #     graph, total_neurons = graph_utilities.operations.convert_to_graph(children[0])
        #     graph_utilities.operations.draw_layered_graph(graph, total_neurons, filename="graph_middle.png")
        # if current_epoch == 100:
        #     graph, total_neurons = graph_utilities.operations.convert_to_graph(children[0])
        #     graph_utilities.operations.draw_layered_graph(graph, total_neurons, filename="graph_end.png")

    return children

def staged_train_epoch(model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader,
                warmup_strategy: src.rules.warmup.Warmup, pruning_strategies: List[src.rules.pruning.Pruning], regrowth_strategies: List[src.rules.regrowth.Regrowth], decay_strategy: src.rules.decay.DecayStrategy,
                device: torch.device, epochs: int, offspring: int, minibatch_budget: int, decay_rate: float, keep: float):
    statistics = training_utilities.initialize_statistics()

    fitness_function = lambda c: c.rank
    criterion = criterion_factory()
    alpha_low, alpha_high = [1.0, 1.0]
    current_alpha = alpha_low

    halving_scheduler = lambda: (False, 1, alpha_high)  # cheap hack to match expected output indicating we have to generate new population

    children = []
    for current_epoch in range(1, epochs + 1):
        statistics["current_epoch"] += 1
        statistics["epoch_start"] = time.perf_counter()

        for pruning_strategy in pruning_strategies:
            pruning_strategy.decay_pruned_fraction(decay_strategy.update(0))  # update fraction

        # apply learning rate decay to children
        if len(children):
            for child in children:
                child.lr_decay.step()

        # train for one epoch
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            halving_time, survivor_count, current_alpha = halving_scheduler()

            if halving_time:
                children = rank_models(children, alpha=current_alpha)
                children.sort(key=fitness_function)
                print([(child.index, fitness_function(child)) for child in children])
                statistics["amortized_flops"] += numpy.sum([child.amortized_sample_cost * train_loader.batch_size * child.seen_batches for child in children[(survivor_count if survivor_count != 1 else 0):]]) # add flops of discarded
                children = soft_top_k(children, select=survivor_count, decay_rate=decay_rate)
                children = [training_utilities.reset_tracked_fields(child) for child in children] # ensures we grade children on their performance within stage not across stages

            # do we schedule new gen?
            if survivor_count == 1:
                halving_scheduler = StageScheduler(minibatch_budget, offspring, keep=keep, alpha_range=(alpha_low, alpha_high))

                # pick best model
                best_model = children[0] if len(children) > 0 else None
                children = []

                # generate new offspring
                for child_index in range(1, offspring + 1):
                    child = training_utilities.spawn_network(model_factory, optimizer_factory, lr_decay_factory, best_model, warmup_strategy, pruning_strategies, regrowth_strategies, train_loader, minibatch_budget, device)
                    child.index = child_index
                    children.append(child)

            # train population
            children, statistics = training_utilities.forward_and_backward_with_statistics(children, criterion, inputs, targets, statistics, batch_index == 0)

        # log epoch end time
        statistics["epoch_end"] = time.perf_counter()
        statistics["total_time"] += statistics["epoch_end"] - statistics["epoch_start"]

        # sort before logging
        children = rank_models(children, alpha=current_alpha)
        children.sort(key=fitness_function)

        # log statistics
        log_entry = training_utilities.generate_log_entry(statistics, children, train_loader, test_loader, device)
        settings.logger.log(log_entry)

    return children

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='main', description='main program used to launch experiments')

    parser.add_argument('--configuration', default=None, type=json.loads, help="experiment configuration to run")
    parser.add_argument('--seed', default=secrets.randbits(32), type=int, help='training seed (for reproducibility)')

    # parse args
    args = parser.parse_args()

    # tell CuDNN to pick only deterministic algorithms in the backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # seed rngs
    rng = numpy.random.default_rng(seed=args.seed)
    random.seed(int(rng.integers(0, 2 ** 32 - 1)))
    numpy.random.seed(rng.integers(0, 2 ** 32 - 1))
    torch.manual_seed(rng.integers(0, 2 ** 32 - 1))
    torch.cuda.manual_seed(rng.integers(0, 2 ** 32 - 1))

    # if no pipeline is passed through command line, scan stdin
    if args.configuration is None:
        configuration = ""
        for line in sys.stdin:
            configuration += line
        args.configuration = json.loads(configuration)

    # spawn global settings object
    settings = settings.Settings(args.configuration)

    # instantiate referenced objects
    train_loader, test_loader = utilities.resolve_class(settings.dataset["type"])(**settings.dataset["kwargs"]).load()

    model_factory = lambda: utilities.resolve_class(settings.model["type"])(**settings.model["kwargs"])
    optimizer_factory = lambda x: utilities.resolve_class(settings.optimizer["type"])(x.parameters(), **settings.optimizer["kwargs"])
    lr_decay_factory = lambda x, **kwargs: utilities.resolve_class(settings.lr_decay["type"])(x, **settings.lr_decay["kwargs"], **kwargs) if settings.lr_decay is not None else None
    criterion_factory = lambda: utilities.resolve_class(settings.criterion["type"])(**settings.criterion["kwargs"])

    warmup = utilities.resolve_class(settings.warmup["type"])(dataset=train_loader, **settings.warmup["kwargs"])
    pruning = [utilities.resolve_class(settings.pruning["type"])(**settings.pruning["kwargs"])] if isinstance(settings.pruning, dict) else [utilities.resolve_class(entry["type"])(**entry["kwargs"]) for entry in settings.pruning]
    regrowth = [utilities.resolve_class(settings.regrowth["type"])(**settings.regrowth["kwargs"])] if isinstance(settings.regrowth, dict) else [utilities.resolve_class(entry["type"])(**entry["kwargs"]) for entry in settings.regrowth]
    decay = utilities.resolve_class(settings.decay["type"])(**settings.decay["kwargs"])

    # instantiate experiment handle and call it
    models = globals()[settings.mode["type"]](model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader, test_loader, warmup_strategy=warmup, pruning_strategies=pruning, regrowth_strategies=regrowth, decay_strategy=decay, **settings.mode["kwargs"])
    utilities.resolve_class(settings.serialize["type"])(**settings.serialize["kwargs"]).serialize([models[0]]) if settings.serialize is not None else None
