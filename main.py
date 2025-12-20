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
import random
import sys

def update_mask(model: torch.nn.Module, warmup_strategy: mask.warmup.Warmup, prunning_strategy: mask.pruning.Pruning, regrowth_strategy: mask.regrowth.Regrowth) -> torch.nn.Module:
    model = warmup_strategy.apply(model)

    # iterate over all layers
    for layer in model.model:
        # we are interested only in custom MaskedLayers
        if not isinstance(layer, mask.network.MaskedLayer):
            continue

        # apply removal/regrowth only for layers that actually require it
        if layer.expected_sparsity() > 0.0:
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
    data_count = len(candidates)

    if decay_rate is None or data_count <= 1:
        return candidates[:select]

    candidate_indices = numpy.arange(data_count)
    percentiles = candidate_indices / (data_count - 1)

    weights = numpy.exp(-decay_rate * percentiles)
    probabilities = weights / weights.sum()

    selected_indices = numpy.random.choice(data_count, size=select, replace=False, p=probabilities)
    return [candidates[i] for i in selected_indices]

def train_epoch(model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader: DataLoader, test_loader: DataLoader,
                warmup_strategy: mask.warmup.Warmup, pruning_strategy: mask.pruning.Pruning, regrowth_strategy: mask.regrowth.Regrowth, decay_strategy: mask.decay.DecayStrategy,
                device: torch.device, epochs: int, offspring: int, minibatch_budget: int, decay_rate: float):
    criterion = criterion_factory()
    total_batches = 0

    children = []
    for current_epoch in range(1, epochs + 1):
        pruned_fraction = pruning_strategy.decay_pruned_fraction(decay_strategy.update(0))

        # apply learning rate decay to children
        if len(children):
            for child in children:
                child.lr_decay.step()

        for (inputs, targets) in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # update population
            if (total_batches / offspring) % minibatch_budget == 0:
                best_model = None

                # pick best model
                if len(children):
                    children.sort(key=lambda c: c.mean_loss / c.seen_batches)
                    # print([child.mean_loss / child.seen_batches for child in children])
                    best_model = soft_top_k(children, select=1, decay_rate=decay_rate)[0]
                    children = []
                else:
                    best_model = model_factory().to(device)
                    best_model.optimizer = optimizer_factory(best_model) # ensure it exists for downstream task

                # generate new offspring
                for child_index in range(1, offspring + 1):
                    child = model_factory().to(device)
                    child.load_state_dict(best_model.state_dict())
                    child = update_mask(child, warmup_strategy, pruning_strategy, regrowth_strategy)

                    child.index = child_index
                    child.seen_batches = 0.0

                    child.mean_loss = 0.0
                    child.variance_loss = 0.0

                    child.train()
                    child.optimizer = optimizer_factory(child)
                    child.optimizer.load_state_dict(best_model.optimizer.state_dict())
                    child.lr_decay = lr_decay_factory(child.optimizer)

                    children.append(child)

            # train population
            for child in children:
                outputs = child(inputs)
                loss = criterion(outputs, targets)

                with torch.no_grad():
                    child.mean_loss += loss.mean().item()
                    child.variance_loss += loss.var().item()
                    child.seen_batches += 1
                    total_batches += 1

                child.optimizer.zero_grad()
                loss.mean().backward()
                child.optimizer.step()

        children.sort(key=lambda c: c.mean_loss / c.seen_batches)

        # log stats
        test_acc = evaluate(children[0], test_loader, device)
        expected_sparsity = children[0].expected_sparsity()
        realized_sparsity = children[0].realized_sparsity()
        active_weights = children[0].active_weights()
        total_weights = children[0].total_weights()

        settings.logger.log({
            "epoch": current_epoch,
            "batches": total_batches,
            "survivor": children[0].index,
            "learning_rate": children[0].optimizer.param_groups[0]["lr"],
            "mean_loss": children[0].mean_loss / children[0].seen_batches,
            "variance_loss": children[0].variance_loss / children[0].seen_batches,
            "p_mean_loss": numpy.mean([child.mean_loss / child.seen_batches for child in children]),
            "p_variance_loss": numpy.mean([child.variance_loss / child.seen_batches for child in children]),
            "test_acc": test_acc,
            "pruned_fraction": pruned_fraction,
            "expected_sparsity": expected_sparsity,
            "realized_sparsity": realized_sparsity,
            "active_weights": active_weights,
            "total_weights": total_weights
        })

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

    if not isinstance(settings.decay, list):
        decay = utilities.resolve_class(settings.decay["type"])(**settings.decay["kwargs"])
    else:
        decay = []
        for entry in settings.decay:
            decay.append(utilities.resolve_class(entry["type"])(**entry["kwargs"]))

    # instantiate experiment handle and call it
    mode = globals()[settings.mode["type"]](model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader, test_loader, warmup_strategy=warmup, pruning_strategy=pruning, regrowth_strategy=regrowth, decay_strategy=decay, **settings.mode["kwargs"])
