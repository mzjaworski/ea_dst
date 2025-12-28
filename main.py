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
import utilities
import random
import sys

def update_mask(model: torch.nn.Module, warmup_strategy: src.rules.warmup.Warmup, prunning_strategy: src.rules.pruning.Pruning, regrowth_strategy: src.rules.regrowth.Regrowth) -> torch.nn.Module:
    model = warmup_strategy.apply(model)

    # iterate over all layers
    for layer in model.model:
        # we are interested only in custom MaskedLayers
        if not isinstance(layer, src.network.detail.MaskedLayer):
            continue

        # apply removal/regrowth only for layers that actually require it
        if layer.expected_sparsity() > 0.0:
            pruned_count, pruned_layer = prunning_strategy.apply(layer)
            regrowth_strategy.apply(pruned_count, pruned_layer)
            # not assigning back should be safe as we deal with references

    return model

def top_k_accuracy(model: torch.nn.Module, testloader: DataLoader, device: torch.device, top_k: list[int]) -> dict[str, float]:
    model.eval()
    max_k = max(top_k)

    correct = {k: 0 for k in top_k}
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, _ = model(inputs)
            _, preds = outputs.topk(max_k, dim=1)
            preds = preds.t()

            matches = preds.eq(targets.view(1, -1))

            for k in top_k:
                correct[k] += matches[:k].any(dim=0).sum().item()

            total += targets.size(0)

    return {f"accuracy-top{k}": 100.0 * correct[k] / total for k in top_k}

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

# def normalized_aulc(candidates: List, eps: float = 0.00001) -> List:
#     loss = numpy.array([c.quantile_loss for c in candidates])
#     per_batch = numpy.mean(loss, axis=0)
#
#     for i, candidate in enumerate(candidates):
#         candidate.normalized_aulc = loss[i] / (per_batch + eps)
#
#     return candidates

# def normalized_aulc(candidates: List, alpha: float = 1.0, eps: float = 1e-8) -> List:
#     loss = numpy.array([c.quantile_loss for c in candidates])
#     diff = numpy.abs(numpy.diff(loss, axis=1))
#
#     pop_loss = numpy.mean(loss, axis=0)
#     pop_diff = numpy.mean(diff, axis=0)
#
#     for i, c in enumerate(candidates):
#         level = loss[i][1:] / (pop_loss[1:] + eps)
#         slope = diff[i] / (pop_diff + eps)
#         c.normalized_aulc = numpy.mean(alpha * level + (1.0 - alpha) * slope)
#
#     return candidates

def normalized_aulc(candidates: List, alpha: float = 1.0, split: int = 20, eps: float = 1e-8) -> List:
    if len(candidates) < 2:
        for i, c in enumerate(candidates):
            c.normalized_aulc = 0.0
        return candidates

    loss = numpy.array([c.mean_loss for c in candidates])  # (N, T)
    diff = numpy.abs(numpy.diff(loss, axis=1))  # (N, T-1)

    N, T = loss.shape
    split = min(split, T)
    part_len = T // split

    scores = numpy.zeros((N, split))

    for s in range(split):
        start = s * part_len
        end = (s + 1) * part_len if s < split - 1 else T

        loss_s = loss[:, start:end]
        diff_s = diff[:, start:end - 1]

        pop_loss = loss_s.mean(axis=0)
        pop_diff = diff_s.mean(axis=0)

        for i in range(N):
            level = loss_s[i][1:] / (pop_loss[1:] + eps)
            slope = diff_s[i] / (pop_diff + eps) if diff_s.shape[1] > 0 else 0.0

            scores[i, s] = numpy.mean(alpha * level + (1.0 - alpha) * slope)

    # rank within each split
    ranks = numpy.argsort(numpy.argsort(scores, axis=0), axis=0)
    ranks = ranks / (N - 1 + eps)

    # average rank across splits
    final_rank = ranks.mean(axis=1)

    for i, c in enumerate(candidates):
        c.normalized_aulc = final_rank[i]

    return candidates




def train_epoch(model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader: DataLoader, test_loader: DataLoader,
                warmup_strategy: src.rules.warmup.Warmup, pruning_strategy: src.rules.pruning.Pruning, regrowth_strategy: src.rules.regrowth.Regrowth, decay_strategy: src.rules.decay.DecayStrategy,
                device: torch.device, epochs: int, offspring: int, minibatch_budget: int, decay_rate: float):
    fitness_function = lambda c: numpy.mean(c.normalized_aulc)
    criterion = criterion_factory()

    pruned_fraction = pruning_strategy.pruned_fraction()
    amortized_flops = 0.0
    total_batches = 0
    total_time = 0

    children = []
    for current_epoch in range(1, epochs + 1):

        # if current_epoch == 35 or current_epoch == 70:
        #     pruned_fraction /= 2.0

        pruning_strategy.decay_pruned_fraction(src.rules.decay.ConstantDecay(pruned_fraction).update(0)) # restore fraction
        pruning_strategy.decay_pruned_fraction(decay_strategy.update(0)) # update fraction
        pruned_fraction = pruning_strategy.pruned_fraction() # store new fraction

        # apply learning rate decay to children
        if len(children):
            for child in children:
                child.lr_decay.step()

        # log epoch start time
        start = time.perf_counter()

        # train for one epoch
        for (inputs, targets) in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # update population
            if (total_batches / offspring) % minibatch_budget == 0:
                best_model = None

                # pick best model
                if len(children):
                    children = normalized_aulc(children)

                    children.sort(key=lambda c: numpy.mean(c.mean_loss))
                    print([(child.index, (lambda c: numpy.mean(c.mean_loss))(child)) for child in children])

                    children.sort(key=lambda c: numpy.mean(c.quantile_loss))
                    print([(child.index, (lambda c: numpy.mean(c.quantile_loss))(child)) for child in children])

                    children.sort(key=fitness_function)
                    print(f"metric : {[(child.index, fitness_function(child)) for child in children]}")

                    best_model = soft_top_k(children, select=1, decay_rate=decay_rate)[0]
                    amortized_flops += numpy.sum([child.amortized_sample_cost * train_loader.batch_size * child.seen_batches for child in children])
                    children = []
                else:
                    best_model = model_factory().to(device)
                    best_model.optimizer = optimizer_factory(best_model) # ensure it exists for downstream task

                # generate new offspring
                for child_index in range(1, offspring + 1):
                    pruning_strategy.decay_pruned_fraction(src.rules.decay.ConstantDecay((pruned_fraction / offspring) * child_index).update(0))

                    child = model_factory().to(device)
                    child.load_state_dict(best_model.state_dict())

                    child = update_mask(child, warmup_strategy, pruning_strategy, regrowth_strategy)

                    # fields related to profiling / state tracking
                    child.index = child_index
                    child.quantile_loss = []
                    child.mean_loss = []
                    child.std_loss = []
                    child.neuron_dormancy = []

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
                outputs, score = child(inputs)
                loss = criterion(outputs, targets)

                with torch.no_grad():
                    child.quantile_loss.append(loss.quantile(q=0.5).item())
                    child.mean_loss.append(loss.mean().item())
                    child.std_loss.append(loss.std().item())
                    child.neuron_dormancy.append(score)
                    child.seen_batches += 1
                    total_batches += 1

                child.optimizer.zero_grad()
                loss.mean().backward()
                child.optimizer.step()

        # log epoch end time
        end = time.perf_counter()
        total_time += end - start

        # sort before logging
        children = normalized_aulc(children)
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
            "neuron_dormancy": numpy.mean(children[0].neuron_dormancy),

            # population statistics
            "p_quantile_loss": numpy.mean([numpy.mean(child.quantile_loss) for child in children]),
            "p_mean_loss": numpy.mean([numpy.mean(child.mean_loss) for child in children]),
            "p_std_loss": numpy.mean([numpy.mean(child.std_loss) for child in children]),
            "p_neuron_dormancy": numpy.mean([numpy.mean(child.neuron_dormancy) for child in children]),

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

def transplant_mask(from_model: torch.nn.Module, to_model: torch.nn.Module, weight_reinitialization: str = "zero") -> torch.nn.Module:
    reinitialization = src.rules.weight.Reinitialization.lookup(weight_reinitialization)

    from_layers = []
    to_layers = []

    for m in from_model.modules():
        if isinstance(m, src.network.detail.MaskedLayer):
            from_layers.append(m)

    for m in to_model.modules():
        if isinstance(m, src.network.detail.MaskedLayer):
            to_layers.append(m)

    assert len(from_layers) == len(to_layers), "MaskedLayer count mismatch"

    for src_layer, tgt_layer in zip(from_layers, to_layers):
        assert src_layer.mask.shape == tgt_layer.mask.shape, "Mask shape mismatch"

        grow = (src_layer.mask.bool() & (~tgt_layer.mask.bool())).view(-1).nonzero(as_tuple=True)
        tgt_layer.layer.weight.data = reinitialization(tgt_layer.layer.weight.data, grow)

        tgt_layer.mask.data.copy_(src_layer.mask.data)
        tgt_layer.mask.requires_grad = False   # freeze transplanted mask

    return to_model

def partial_sort(lst, start=1, end=None, *, key=None, reverse=False):
    lst[start:end] = sorted(lst[start:end], key=key, reverse=reverse)

def train_epoch_v2(model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader: DataLoader, test_loader: DataLoader,
                warmup_strategy: src.rules.warmup.Warmup, pruning_strategy: src.rules.pruning.Pruning, regrowth_strategy: src.rules.regrowth.Regrowth, decay_strategy: src.rules.decay.DecayStrategy,
                device: torch.device, epochs: int, offspring: int, minibatch_budget: int, decay_rate: float):

    def initialize_fields(model: torch.nn.Module) -> torch.nn.Module:
        # fields related to profiling / state tracking
        model.mean_loss = []
        model.quantile_loss = []
        model.seen_batches = 0

        # fields related to optimization state
        model.optimizer = optimizer_factory(model)
        model.lr_decay = lr_decay_factory(model.optimizer)

        return model

    fitness_function = lambda c: numpy.mean(c.normalized_aulc)
    criterion = criterion_factory()

    amortized_flops = 0.0
    total_batches = 0
    total_time = 0

    main_network = initialize_fields(model_factory().to(device))
    children = []

    for current_epoch in range(1, epochs + 1):
        pruning_strategy.decay_pruned_fraction(decay_strategy.update(0))
        pruned_fraction = pruning_strategy.pruned_fraction()

        # reset fields
        main_network.quantile_loss = []
        main_network.mean_loss = []

        # log epoch start time
        start = time.perf_counter()

        # train for one epoch
        for (inputs, targets) in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # update main network mask
            if main_network.seen_batches % minibatch_budget == 0:
                weight_donor = model_factory().to(device)
                children = []

                # generate new offspring
                for child_index in range(1, offspring + 1):
                    clone = model_factory().to(device)
                    clone.load_state_dict(main_network.state_dict())
                    clone = update_mask(clone, warmup_strategy, pruning_strategy, regrowth_strategy)

                    child = initialize_fields(model_factory().to(device))
                    child.load_state_dict(weight_donor.state_dict())
                    child.index = child_index

                    child = transplant_mask(clone, child, weight_reinitialization="zero")

                    # if child_index != 1:
                    #     child = transplant_mask(clone, child, weight_reinitialization="zero")
                    # else:
                    #     child = transplant_mask(main_network, child, weight_reinitialization="zero")

                    child.train()

                    child.optimizer = optimizer_factory(child)
                    child.optimizer.load_state_dict(main_network.optimizer.state_dict())
                    child.lr_decay = lr_decay_factory(child.optimizer)

                    children.append(child)

                # train offspring
                for batch_index, (child_inputs, child_targets) in enumerate(cycle(train_loader)):
                    child_inputs, child_targets = child_inputs.to(device), child_targets.to(device)

                    if batch_index >= minibatch_budget:
                        break

                    for child in children:
                        outputs, score = child(child_inputs)
                        loss = criterion(outputs, child_targets)

                        with torch.no_grad():
                            child.quantile_loss.append(loss.quantile(q=0.5).item())
                            child.mean_loss.append(loss.mean().item())
                            child.seen_batches += 1
                            total_batches += 1

                        child.optimizer.zero_grad()
                        loss.mean().backward()
                        child.optimizer.step()

                # pick best and transplant mask back to main network
                children = normalized_aulc(children)
                children.sort(key=fitness_function)
                print([(child.index, fitness_function(child)) for child in children])
                main_network = transplant_mask(soft_top_k(children, select=1, decay_rate=decay_rate)[0], main_network, weight_reinitialization="zero")

            main_network.train()

            # train main network
            outputs, score = main_network(inputs)
            loss = criterion(outputs, targets)

            with torch.no_grad():
                main_network.quantile_loss.append(loss.quantile(q=0.5).item())
                main_network.mean_loss.append(loss.mean().item())
                main_network.seen_batches += 1
                total_batches += 1

            main_network.optimizer.zero_grad()
            loss.mean().backward()
            main_network.optimizer.step()

        # log epoch end time
        end = time.perf_counter()
        total_time += end - start

        # apply learning rate decay to main_network
        main_network.lr_decay.step()

        # sort before logging
        children = normalized_aulc(children)
        children.sort(key=fitness_function)

        # create log entry
        log_entry = {
            # global info
            "epoch": current_epoch,
            "total_time": total_time,
            "epoch_time": end - start,
            "total_batches": total_batches,

            # best member statistics
            "mean_loss": numpy.mean(main_network.mean_loss),
            "quantile_loss": numpy.mean(main_network.quantile_loss),

            # population statistics
            "p_mean_loss": numpy.mean([numpy.mean(child.mean_loss) for child in children]),
            "p_quantile_loss": numpy.mean([numpy.mean(child.quantile_loss) for child in children]),

            # miscellaneous
            "expected_sparsity": main_network.expected_sparsity(),
            "realized_sparsity": main_network.realized_sparsity(),
            "active_weights": main_network.active_weights(),
            "total_weights": main_network.total_weights(),
            "learning_rate": main_network.optimizer.param_groups[0]["lr"],
            "pruned_fraction": pruned_fraction,
        }

        # get accuracy at top-k of the best model under current fitness_function
        for (level, value) in top_k_accuracy(main_network, test_loader, device, [1, 3, 5]).items():
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
