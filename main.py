import json

from torch.utils.data import DataLoader
from itertools import cycle
from typing import List
import secrets
import torch
import numpy

import mask.network
import mask.warmup
import mask.regrowth
import mask.pruning
import mask.decay

import argparse
import settings
import utilities
import sys

def update_mask(model: torch.nn.Module, warmup_strategy: mask.warmup.Warmup, prunning_strategy: mask.pruning.Pruning, regrowth_strategy: mask.regrowth.Regrowth) -> torch.nn.Module:
    model = warmup_strategy.apply(model)

    for layer in model.model:
        if not isinstance(layer, mask.network.MaskedLayer):
            continue

        pruned_count, pruned_layer = prunning_strategy.apply(layer)
        regrowth_strategy.apply(pruned_count, pruned_layer)
        # not assigning back should be safe as we deal with references

    return model

def evaluate(model: torch.nn.Module, testloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return 100 * correct / total

def soft_top_k(candidates: List, select: int, decay_rate: float) -> List:

    if decay_rate is None:
        return candidates[:select]

    num_candidates = len(candidates)
    candidate_indices = numpy.arange(num_candidates)

    percentiles = candidate_indices / (num_candidates - 1) if num_candidates > 1 else numpy.zeros(1)

    weights = numpy.exp(-decay_rate * percentiles)
    probabilities = weights / weights.sum()
    selected_indices = numpy.random.choice(num_candidates, size=select, replace=False, p=probabilities)

    return [candidates[i] for i in selected_indices]

def train_sgd_batch(model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader: DataLoader, test_loader: DataLoader,
                    warmup_strategy: mask.warmup.Warmup, pruning_strategy: mask.pruning.Pruning, regrowth_strategy: mask.regrowth.Regrowth, decay_strategy: mask.decay.DecayStrategy,
                    device: torch.device, epochs: int, minibatch_budget: int):
    model = model_factory()
    optimizer = optimizer_factory(model)
    lr_decay = lr_decay_factory(optimizer)
    criterion = criterion_factory()
    total_batches = 0

    model.train()
    model.to(device)

    for epoch in range(1, epochs + 1):
        pruned_fraction = pruning_strategy.decay_pruned_fraction(decay_strategy.update(0))

        model.train()
        model = update_mask(model, warmup_strategy, pruning_strategy, regrowth_strategy)

        # # TODO for debugging mask update rules
        # for name, param in model.named_parameters():
        #     if name == 'model.9.mask':
        #         print(name, param)

        total_mean = 0.0
        total_variance = 0.0

        for batch_index, (inputs, targets) in enumerate(cycle(train_loader)):
            if batch_index >= minibatch_budget:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            with torch.no_grad():
                total_mean += loss.mean().item()
                total_variance += loss.var().item()
                total_batches += 1

            optimizer.zero_grad()
            loss.mean().backward()

            optimizer.step()

        total_mean /= minibatch_budget
        total_variance /= minibatch_budget

        # log stats
        acc = evaluate(model, test_loader, device)
        expected_sparsity = model.expected_sparsity()
        realized_sparsity = model.realized_sparsity()
        active_weights = model.active_weights()
        total_weights = model.total_weights()

        if lr_decay is not None:
            lr_decay.step()

        settings.logger.log({
            "epoch": epoch,
            "batches": total_batches,
            "mean_loss": total_mean,
            "variance_loss": total_variance,
            "test_acc": acc,
            "pruned_fraction": pruned_fraction,
            "expected_sparsity": expected_sparsity,
            "realized_sparsity": realized_sparsity,
            "active_weights": active_weights,
            "total_weights": total_weights
        })

    return model

def train_ea_batch(model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader: DataLoader, test_loader: DataLoader,
                   warmup_strategy: mask.warmup.Warmup, pruning_strategy: mask.pruning.Pruning, regrowth_strategy: mask.regrowth.Regrowth, decay_strategy: mask.decay.DecayStrategy,
                   device: torch.device, generations: int, offspring: int, decay_rate: float, minibatch_budget: int):

    best_model = model_factory().to(device)
    criterion = criterion_factory()
    best_model.index = 1

    best_model.mean_loss = float("inf")
    total_batches = 0
    indicator = 1.0

    rng = numpy.random.default_rng(seed=args.seed)

    for gen in range(1, generations + 1):
        pruned_fraction = pruning_strategy.decay_pruned_fraction(decay_strategy.update(indicator))
        best_model.index = 0

        children = []
        for idx in range(offspring):
            child = model_factory().to(device)
            child.load_state_dict(best_model.state_dict())

            child = update_mask(child, warmup_strategy, pruning_strategy, regrowth_strategy)
            child.train()

            child.index = idx + 1
            child.mean_loss = 0.0
            child.variance_loss = 0.0

            optimizer = optimizer_factory(child)
            lr_decay = lr_decay_factory(optimizer)

            if lr_decay is not None:
                optimizer.step() # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
                for _ in range(1, gen): # decay learning rate
                    lr_decay.step()

            for batch_index, (inputs, targets) in enumerate(cycle(train_loader)):
                if batch_index >= minibatch_budget:
                    break

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = child(inputs)
                loss = criterion(outputs, targets)

                with torch.no_grad():
                    child.mean_loss += loss.mean().item()
                    child.variance_loss += loss.var().item()
                    total_batches += 1

                optimizer.zero_grad()
                loss.mean().backward()

                optimizer.step()

            child.mean_loss /= minibatch_budget
            child.variance_loss /= minibatch_budget
            total_batches += minibatch_budget
            children.append(child)

        # sort based on fitness
        children.sort(key=lambda c: c.mean_loss)

        # pick winner
        best_model = soft_top_k(children, select=1, decay_rate=decay_rate)[0]
        indicator = 1.0 if best_model.index == 1 else 0.0

        # log stats
        test_acc = evaluate(best_model, test_loader, device)
        expected_sparsity = best_model.expected_sparsity()
        realized_sparsity = best_model.realized_sparsity()
        active_weights = best_model.active_weights()
        total_weights = best_model.total_weights()

        settings.logger.log({
            "epoch": gen,
            "batches": total_batches,
            "survivor": best_model.index,
            "indicator": indicator,
            "mean_loss": best_model.mean_loss,
            "variance_loss": best_model.variance_loss,
            "p_mean_loss": numpy.mean([child.mean_loss for child in children]),
            "p_variance_loss": numpy.mean([child.variance_loss for child in children]),
            "test_acc": test_acc,
            "pruned_fraction": pruned_fraction,
            "expected_sparsity": expected_sparsity,
            "realized_sparsity": realized_sparsity,
            "active_weights": active_weights,
            "total_weights": total_weights
        })

    return best_model

def train_ea_batch_v2(model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader: DataLoader, test_loader: DataLoader,
                   warmup_strategy: mask.warmup.Warmup, pruning_strategy: mask.pruning.Pruning, regrowth_strategy: mask.regrowth.Regrowth, decay_strategy: mask.decay.DecayStrategy,
                   device: torch.device, generations: int, offspring: int, decay_rate: float, minibatch_budget: int):

    best_model = model_factory().to(device)
    criterion = criterion_factory()
    best_model.index = 1

    best_model.mean_loss = float("inf")
    total_batches = 0
    indicator = 1.0

    for gen in range(1, generations + 1):
        pruned_fraction = pruning_strategy.decay_pruned_fraction(decay_strategy.update(indicator))
        best_model.index = 0

        children = []
        for idx in range(offspring + 1):
            child = model_factory().to(device)
            child.load_state_dict(best_model.state_dict())

            if idx:
                child = update_mask(child, warmup_strategy, pruning_strategy, regrowth_strategy)
            child.train()

            child.index = idx + 1
            child.mean_loss = 0.0
            child.variance_loss = 0.0

            optimizer = optimizer_factory(child)
            lr_decay = lr_decay_factory(optimizer)

            if lr_decay is not None:
                optimizer.step() # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
                for _ in range(1, gen): # decay learning rate
                    lr_decay.step()

            for batch_index, (inputs, targets) in enumerate(cycle(train_loader)):
                if batch_index >= minibatch_budget:
                    break

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = child(inputs)
                loss = criterion(outputs, targets)

                with torch.no_grad():
                    child.mean_loss += loss.mean().item()
                    child.variance_loss += loss.var().item()
                    total_batches += 1

                optimizer.zero_grad()
                loss.mean().backward()

                optimizer.step()

            child.mean_loss /= minibatch_budget
            child.variance_loss /= minibatch_budget
            total_batches += minibatch_budget
            children.append(child)

        # sort based on fitness
        children.sort(key=lambda c: c.mean_loss)

        # pick winner
        best_model = soft_top_k(children, select=1, decay_rate=decay_rate)[0]
        indicator = 1.0 if best_model.index == 1 else 0.0

        # log stats
        test_acc = evaluate(best_model, test_loader, device)
        expected_sparsity = best_model.expected_sparsity()
        realized_sparsity = best_model.realized_sparsity()
        active_weights = best_model.active_weights()
        total_weights = best_model.total_weights()

        settings.logger.log({
            "epoch": gen,
            "batches": total_batches,
            "survivor": best_model.index,
            "indicator": indicator,
            "mean_loss": best_model.mean_loss,
            "variance_loss": best_model.variance_loss,
            "p_mean_loss": numpy.mean([child.mean_loss for child in children]),
            "p_variance_loss": numpy.mean([child.variance_loss for child in children]),
            "test_acc": test_acc,
            "pruned_fraction": pruned_fraction,
            "expected_sparsity": expected_sparsity,
            "realized_sparsity": realized_sparsity,
            "active_weights": active_weights,
            "total_weights": total_weights
        })

    return best_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='main', description='main program used to launch experiments')

    parser.add_argument('--configuration', default=None, type=json.loads, help="experiment configuration to run")
    parser.add_argument('--seed', default=secrets.randbits(32), type=int, help='training seed (for reproducibility)')

    # parse args
    args = parser.parse_args()

    # seed rngs
    rng = numpy.random.default_rng(seed=args.seed)
    numpy.random.seed(rng.integers(0, 2 ** 32 - 1))
    torch.manual_seed(rng.integers(0, 2 ** 32 - 1))

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

    if not isinstance(settings.decay, list):
        decay = utilities.resolve_class(settings.decay["type"])(**settings.decay["kwargs"])
    else:
        decay = []
        for entry in settings.decay:
            decay.append(utilities.resolve_class(entry["type"])(**entry["kwargs"]))

    # instantiate experiment handle and call it
    mode = globals()[settings.mode["type"]](model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader, test_loader, warmup_strategy=warmup, pruning_strategy=pruning, regrowth_strategy=regrowth, decay_strategy=decay, **settings.mode["kwargs"])
