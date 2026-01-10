import json

from torch.utils.data import DataLoader
from itertools import cycle
from typing import List
import secrets
import torch
import numpy
import time

import src.network.detail
import src.rules.warmup
import src.rules.regrowth
import src.rules.pruning
import src.rules.decay

import argparse
import settings
import training_utilities
import utilities
import random
import sys

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

def rank_models(candidates: List[torch.nn.Module], alpha: float = 0.7, eps: float = 1e-8) -> List[torch.nn.Module]:

    if len(candidates) < 2 or len(candidates[0].mean_loss) < 2:
        for c in candidates:
            c.rank = 0.0
        return candidates

    # get loss and its change
    loss = numpy.array([c.mean_loss for c in candidates])
    diff = numpy.abs(numpy.diff(loss, axis=1))

    # align loss with change
    loss = loss[:, 1:]

    # compute per candidate averages
    loss_score = numpy.mean(loss, axis=1)
    diff_score = numpy.mean(diff, axis=1)

    # assign ranks
    loss_rank = numpy.argsort(numpy.argsort(loss_score))
    diff_rank = numpy.argsort(numpy.argsort(diff_score))

    # normalize in [0, 1]
    N = len(candidates)
    loss_rank = loss_rank / (N - 1 + eps)
    diff_rank = diff_rank / (N - 1 + eps)

    # combine based on alpha
    final_rank = alpha * loss_rank + (1.0 - alpha) * diff_rank

    for i, c in enumerate(candidates):
        c.rank = final_rank[i]

    return candidates

def train_epoch(model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader: DataLoader, test_loader: DataLoader,
                warmup_strategy: src.rules.warmup.Warmup, pruning_strategy: src.rules.pruning.Pruning, regrowth_strategy: src.rules.regrowth.Regrowth, decay_strategy: src.rules.decay.DecayStrategy,
                device: torch.device, epochs: int, offspring: int, minibatch_budget: int, decay_rate: float):

    statistics = {
        "amortized_flops": 0.0,
        "current_epoch": 0,
        "epoch_start": 0,
        "epoch_end": 0,
        "total_time": 0,
        "total_batches": 0,
    }

    fitness_function = lambda c: c.rank
    criterion = criterion_factory()
    alpha = 0.7

    children = []
    for current_epoch in range(1, epochs + 1):
        statistics["current_epoch"] += 1
        statistics["epoch_start"] = time.perf_counter()

        pruning_strategy.decay_pruned_fraction(decay_strategy.update(0)) # update fraction

        # if len(children):
        #     print(children[0].classifier.mask)

        # apply learning rate decay to children
        if len(children):
            for child in children:
                child.lr_decay.step()

        # train for one epoch
        for (inputs, targets) in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # update population
            if (statistics["total_batches"] / offspring) % minibatch_budget == 0:
                best_model = None

                # pick best model
                if len(children):
                    children = rank_models(children, alpha=alpha)
                    children.sort(key=fitness_function)
                    print([(child.index, fitness_function(child)) for child in children])
                    best_model = soft_top_k(children, select=1, decay_rate=decay_rate)[0]
                    statistics["amortized_flops"] += numpy.sum([child.amortized_sample_cost * train_loader.batch_size * child.seen_batches for child in children])
                    children = []

                # generate new offspring
                for child_index in range(1, offspring + 1):
                    child = training_utilities.spawn_network(model_factory, optimizer_factory, lr_decay_factory, best_model, warmup_strategy, pruning_strategy, regrowth_strategy, train_loader, minibatch_budget, device)
                    child.index = child_index
                    children.append(child)

            # train population
            for child in children:
                training_utilities.forward_and_backward_with_statistics(child, criterion, inputs, targets)
                statistics["total_batches"] += 1

        # log epoch end time
        statistics["epoch_end"] = time.perf_counter()
        statistics["total_time"] += statistics["epoch_end"] - statistics["epoch_start"]

        # sort before logging
        children = rank_models(children, alpha=alpha)
        children.sort(key=fitness_function)

        # log statistics
        log_entry = training_utilities.generate_log_entry(statistics, children, train_loader, test_loader, device)
        settings.logger.log(log_entry)

    return children[0]

from typing import List, Tuple
import math

def successive_halving_batches(total_budget: int, num_children: int, keep: float) -> List[int]:
    S = int(math.floor(math.log(num_children, 1.0 / keep)))
    weights = [(1.0 / keep) ** s for s in range(S)]
    Z = sum(weights)

    batches = [int(total_budget * w / Z) for w in weights]
    batches[-1] += total_budget - sum(batches)  # fix rounding
    return batches

class SuccessiveHalvingScheduler:
    def __init__(self, total_budget: int, num_children: int, keep: float, alpha_range: Tuple[float, float] = (1.0, 1.0)):
        self.total_budget = total_budget
        self.num_children = num_children
        self.keep = keep

        # Compute survivors per stage
        S = int(math.floor(math.log(num_children, 1.0 / keep)))
        self.survivors = [max(1, int(round(num_children * (keep ** s)))) for s in range(S)]

        # Compute minibatches per stage using weighted schedule
        self.batches_per_stage = successive_halving_batches(total_budget, num_children, keep)

        self.alpha_range = numpy.linspace(alpha_range[0], alpha_range[1], S)

        self.stage_idx = 0
        self.consumed_in_stage = 0

    def __call__(self) -> Tuple[bool, int, float]:
        """
        Consume one minibatch. Returns the number of models to keep for current stage.
        Last call returns 1 to signal iteration end.
        """
        if self.stage_idx >= len(self.survivors):
            return True, 1, self.alpha_range[-1]  # iteration finished

        self.consumed_in_stage += 1

        if self.consumed_in_stage >= self.batches_per_stage[self.stage_idx]:
            self.stage_idx += 1
            self.consumed_in_stage = 0
            if self.stage_idx >= len(self.survivors):
                return True, 1, self.alpha_range[-1]  # iteration finished

        return self.consumed_in_stage == 0, self.survivors[self.stage_idx], self.alpha_range[self.stage_idx - 1 if self.consumed_in_stage == 0 else self.stage_idx]



def train_epoch_v2(model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader: DataLoader, test_loader: DataLoader,
                warmup_strategy: src.rules.warmup.Warmup, pruning_strategy: src.rules.pruning.Pruning, regrowth_strategy: src.rules.regrowth.Regrowth, decay_strategy: src.rules.decay.DecayStrategy,
                device: torch.device, epochs: int, offspring: int, minibatch_budget: int, decay_rate: float, alpha_low: float, alpha_high: float, keep: float):
    fitness_function = lambda c: c.rank
    current_alpha = alpha_low

    criterion = criterion_factory()

    amortized_flops = 0.0
    total_batches = 0
    total_time = 0

    halving_scheduler = lambda: (True, 1, alpha_high) # cheap hack to match expected output
    children = [model_factory().to(device)]
    children[0].optimizer = optimizer_factory(children[0])

    for current_epoch in range(1, epochs + 1):
        pruning_strategy.decay_pruned_fraction(decay_strategy.update(0)) # update fraction
        pruned_fraction = pruning_strategy.pruned_fraction() # store new fraction

        # apply learning rate decay to children
        if current_epoch > 1:
            for child in children:
                child.lr_decay.step()

        # log epoch start time
        start = time.perf_counter()

        # train for one epoch
        for (inputs, targets) in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            halving_time, survivor_count, current_alpha = halving_scheduler()

            if halving_time and len(children) > 1:
                children = rank_models(children, alpha=current_alpha)
                children.sort(key=fitness_function)
                print([(child.index, fitness_function(child)) for child in children])
                amortized_flops += numpy.sum([child.amortized_sample_cost * train_loader.batch_size * child.seen_batches for child in children[survivor_count if survivor_count != 1 else 0:]]) # add flops of discarded
                children = soft_top_k(children, select=survivor_count, decay_rate=decay_rate)

                # reset tracked metrics
                for child in children:
                    child.quantile_loss = []
                    child.mean_loss = []
                    child.std_loss = []
                    child.gradients = []

            # do we schedule new gen?
            if survivor_count == 1:
                halving_scheduler = SuccessiveHalvingScheduler(minibatch_budget, offspring, keep=keep, alpha_range=(alpha_low, alpha_high))

                # pick best model
                best_model = children[0]
                children = []

                # generate new offspring
                for child_index in range(1, offspring + 1):
                    child = model_factory().to(device)
                    child.load_state_dict(best_model.state_dict())

                    child = update_mask(child, warmup_strategy, pruning_strategy, regrowth_strategy)

                    # fields related to profiling / state tracking
                    child.index = child_index
                    child.quantile_loss = []
                    child.mean_loss = []
                    child.std_loss = []
                    child.gradients = []

                    # fields required to estimate (amortized) FLOP cost of operations (slightly adjusted "Rigging the Lottery: Making All Tickets Winners" logic as original idea wouldn't fit here)
                    child.amortized_sample_cost = src.utilities.estimate_amortized_flops(child, warmup_strategy, train_loader, minibatch_budget)
                    child.seen_batches = 0

                    child.train()

                    # fields related to optimization state
                    child.optimizer = optimizer_factory(child)
                    child.optimizer.load_state_dict(best_model.optimizer.state_dict())
                    child.lr_decay = lr_decay_factory(child.optimizer)

                    children.append(child)

            # train population
            for child in children:
                outputs = child(inputs)
                loss = criterion(outputs, targets)

                with torch.no_grad():
                    child.quantile_loss.append(loss.quantile(q=0.5).item())
                    child.mean_loss.append(loss.mean().item())
                    child.std_loss.append(loss.std().item())
                    child.seen_batches += 1
                    total_batches += 1

                child.optimizer.zero_grad()
                loss.mean().backward()

                gradients = []
                for current_layer in child.model:
                    if not isinstance(current_layer, src.network.detail.MaskedLayer):
                        continue

                    if current_layer.layer.weight.grad is not None:
                        gradients.append(current_layer.layer.weight.grad[current_layer.mask == 1].flatten())
                child.gradients.append(torch.cat(gradients).cpu().numpy())

                child.optimizer.step()

        # log epoch end time
        end = time.perf_counter()
        total_time += end - start

        # sort before logging
        children = rank_models(children, alpha=current_alpha)
        children.sort(key=fitness_function)

        # create log entry
        log_entry = {
            # global info
            "epoch": current_epoch,
            "total_time": total_time,
            "epoch_time": end - start,
            "total_batches": total_batches,
            "amortized_flops": amortized_flops + numpy.sum([child.amortized_sample_cost * train_loader.batch_size * child.seen_batches for child in children]),

            # best member statistics
            "current_best": children[0].index,
            "quantile_loss": numpy.mean(children[0].quantile_loss),
            "mean_loss": numpy.mean(children[0].mean_loss),
            "std_loss": numpy.mean(children[0].std_loss),
            "mean_gradient": numpy.mean(numpy.mean(children[0].gradients, axis=0)),
            "std_gradient": numpy.mean(numpy.std(children[0].gradients, axis=0)),

            # population statistics
            "p_quantile_loss": numpy.mean([numpy.mean(child.quantile_loss) for child in children]),
            "p_mean_loss": numpy.mean([numpy.mean(child.mean_loss) for child in children]),
            "p_std_loss": numpy.mean([numpy.mean(child.std_loss) for child in children]),
            "p_mean_gradient": numpy.mean([numpy.mean(numpy.mean(child.gradients, axis=0)) for child in children]),
            "p_std_gradient": numpy.mean([numpy.mean(numpy.std(child.gradients, axis=0)) for child in children]),

            # miscellaneous
            "expected_sparsity": children[0].expected_sparsity(),
            "realized_sparsity": children[0].realized_sparsity(),
            "active_weights": children[0].active_weights(),
            "total_weights": children[0].total_weights(),
            "learning_rate": children[0].optimizer.param_groups[0]["lr"],
            "pruned_fraction": pruned_fraction,
        }

        # get accuracy at top-k of the best model under current fitness_function
        for (level, value) in top_k_accuracy(children[0], test_loader, device, [1, 3, 5]).items():
            log_entry[level] = value

        # store log entry
        settings.logger.log(log_entry)

    return children[0]



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
    pruning = utilities.resolve_class(settings.pruning["type"])(**settings.pruning["kwargs"])
    regrowth = utilities.resolve_class(settings.regrowth["type"])(**settings.regrowth["kwargs"])
    decay = utilities.resolve_class(settings.decay["type"])(**settings.decay["kwargs"])

    # instantiate experiment handle and call it
    mode = globals()[settings.mode["type"]](model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader, test_loader, warmup_strategy=warmup, pruning_strategy=pruning, regrowth_strategy=regrowth, decay_strategy=decay, **settings.mode["kwargs"])
