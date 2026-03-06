import training_utilities
import src.rules.warmup
import src.rules.regrowth
import src.rules.pruning
import src.rules.decay
import utilities
import settings

from typing import List
import secrets
import torch
import numpy
import time
import json
import argparse
import random
import sys

def rank_models(candidates: List[torch.nn.Module]) -> List[torch.nn.Module]: # ranks using z-score
    for i, c in enumerate(candidates):
        c.fitness = float(numpy.mean(c.mean_loss))
    return candidates

def train_epoch(model_factory, optimizer_factory, lr_decay_factory, criterion_factory, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader,
                warmup_strategy: src.rules.warmup.Warmup, pruning_strategies: List[src.rules.pruning.Pruning], regrowth_strategies: List[src.rules.regrowth.Regrowth], decay_strategy: src.rules.decay.DecayStrategy,
                device: torch.device, epochs: int, parents: int, offspring: int, minibatch_budget: int, warmup: int, redistribute: bool = False, noise: float = 0.05):
    statistics = training_utilities.initialize_statistics()

    fitness_function = lambda c: c.fitness
    criterion = criterion_factory()

    warmup_strategy.update_factor(parents / offspring) # for gradient warmups we need to discount dense backward as its not computed ofr every pop member and just for parents
    children = [training_utilities.spawn_network(model_factory, optimizer_factory, lr_decay_factory, None, warmup_strategy, pruning_strategies, regrowth_strategies, train_loader, minibatch_budget, device, cycles=warmup) for _ in range(parents)]
    for child_index, child in enumerate(children):
        child.index = child_index
        statistics["amortized_flops"] += warmup * child.constant_cost # count warmup cost

    for current_epoch in range(1, epochs + 1):
        survivors = []

        statistics["current_epoch"] += 1
        statistics["epoch_start"] = time.perf_counter()

        for pruning_strategy in pruning_strategies:
            pruning_strategy.decay_pruned_fraction(decay_strategy.update(0)) # update fraction

        # train for one epoch
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # update population
            if (statistics["total_batches"] / offspring) % minibatch_budget == 0:

                # pick best model
                if len(children):
                    children = rank_models(children)
                    children.sort(key=fitness_function)
                    # print([(child.index, fitness_function(child)) for child in children])

                    statistics["amortized_flops"] += numpy.sum([(child.amortized_sample_cost * train_loader.batch_size * child.seen_batches) + child.constant_cost for child in children])
                    survivors = children[:parents]
                    children = []

                    # run warmup procedure
                    for survivor in survivors:
                        warmup_strategy.apply(survivor)

                    print([(1.0 - module.realized_sparsity()) for module in survivors[0].modules() if isinstance(module, src.network.detail.MaskedLayer)])

                # generate new offspring
                for child_index in range(1, offspring + 1):
                    survivor_index = numpy.random.randint(0, len(survivors))
                    child = training_utilities.spawn_network(model_factory, optimizer_factory, lr_decay_factory, survivors[survivor_index], warmup_strategy, pruning_strategies, regrowth_strategies, train_loader, minibatch_budget, device, cycles=1, redistribute=redistribute, noise=noise)
                    child.index = child_index
                    children.append(child)

            # train population
            children, statistics = training_utilities.forward_and_backward_with_statistics(children, criterion, inputs, targets, statistics)

        # apply learning rate decay to children - doing it here prevents pytorch warnings
        if len(children):
            for child in children:
                child.lr_decay.step()

        # log epoch end time
        statistics["epoch_end"] = time.perf_counter()
        statistics["total_time"] += statistics["epoch_end"] - statistics["epoch_start"]

        # sort before logging
        children = rank_models(children)
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
