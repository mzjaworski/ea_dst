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
    num_candidates = len(candidates)
    candidate_indices = numpy.arange(num_candidates)

    selection_weights = numpy.exp(-decay_rate * candidate_indices)
    selection_probabilities = selection_weights / selection_weights.sum()
    selected_indices = numpy.random.choice(num_candidates, size=select, replace=False, p=selection_probabilities)

    return [candidates[i] for i in selected_indices]  # return selected candidates

def train_sgd_batch(model_factory, optimizer_factory, criterion_factory, train_loader: DataLoader, test_loader: DataLoader,
                    warmup_strategy: mask.warmup.Warmup, pruning_strategy: mask.pruning.Pruning, regrowth_strategy: mask.regrowth.Regrowth,
                    device: torch.device, epochs: int, minibatch_budget: int):
    model = model_factory()
    optimizer = optimizer_factory(model)
    criterion = criterion_factory()
    total_batches = 0

    model.train()
    model.to(device)

    for epoch in range(epochs):
        model.train()

        model = update_mask(model, warmup_strategy, pruning_strategy, regrowth_strategy)

        # # TODO for debugging mask update rules
        # for name, param in model.named_parameters():
        #     if name == 'model.9.mask':
        #         print(name, param)

        total_loss = 0

        for batch_index, (inputs, targets) in enumerate(cycle(train_loader)):
            if batch_index >= minibatch_budget:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

        total_loss /= minibatch_budget
        acc = evaluate(model, test_loader, device)

        settings.logger.log({"epoch": epoch, "batches": total_batches, "loss": total_loss, "test_acc": acc})

    return model

def train_ea_batch(model_factory, optimizer_factory, criterion_factory, train_loader: DataLoader, test_loader: DataLoader,
                   warmup_strategy: mask.warmup.Warmup, pruning_strategy: mask.pruning.Pruning, regrowth_strategy: mask.regrowth.Regrowth,
                   device: torch.device, generations: int, offspring: int, decay_rate: float, minibatch_budget: int):

    best_model = model_factory().to(device)
    criterion = criterion_factory()

    best_model.fitness = float("inf")
    total_batches = 0

    for gen in range(1, generations + 1):
        best_model.index = 0

        children = []
        for idx in range(offspring):
            child = model_factory().to(device)
            child.load_state_dict(best_model.state_dict())

            child = update_mask(child, warmup_strategy, pruning_strategy, regrowth_strategy)
            child.train()

            child.index = idx + 1
            child.fitness = 0.0

            optimizer = optimizer_factory(child)
            for batch_index, (inputs, targets) in enumerate(cycle(train_loader)):
                if batch_index >= minibatch_budget:
                    break

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = child(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                child.fitness += loss.item()

            child.fitness /= minibatch_budget
            children.append(child)

            # settings.logger.log({"epoch": gen, "child": child.index, "loss": child.fitness})
        total_batches += minibatch_budget

        # sort based on fitness
        children.sort(key=lambda c: c.fitness)

        best_model = soft_top_k(children, select=1, decay_rate=decay_rate)[0]
        test_acc = evaluate(best_model, test_loader, device)

        settings.logger.log({"epoch": gen, "child": "survivor", "batches": total_batches, "index": best_model.index, "loss": best_model.fitness, "test_acc": test_acc})

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
    criterion_factory = lambda: utilities.resolve_class(settings.criterion["type"])(**settings.criterion["kwargs"])

    warmup = utilities.resolve_class(settings.warmup["type"])(dataset=train_loader, **settings.warmup["kwargs"])
    pruning = utilities.resolve_class(settings.pruning["type"])(**settings.pruning["kwargs"])
    regrowth = utilities.resolve_class(settings.regrowth["type"])(**settings.regrowth["kwargs"])

    # instantiate experiment handle and call it
    mode = globals()[settings.mode["type"]](model_factory, optimizer_factory, criterion_factory, train_loader, test_loader, warmup_strategy=warmup, pruning_strategy=pruning, regrowth_strategy=regrowth, **settings.mode["kwargs"])
