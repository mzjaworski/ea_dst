from torch.utils.data import DataLoader
from itertools import cycle
from typing import List
import torchvision
import torch
import numpy
from network import SparseANN, SparseCNN
import mask_mutation as mask

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

def soft_top_k(candidates: List, select: int = 1, decay_rate: float = 1.5) -> List:
    num_candidates = len(candidates)
    candidate_indices = numpy.arange(num_candidates)

    selection_weights = numpy.exp(-decay_rate * candidate_indices)
    selection_probabilities = selection_weights / selection_weights.sum()
    selected_indices = numpy.random.choice(num_candidates, size=select, replace=False, p=selection_probabilities)

    return [candidates[i] for i in selected_indices]  # return selected candidates

def train_sgd_batch(model, trainset, testset, train_batch: int = 64, test_batch: int = 128, device: torch.device = "cuda", epochs: int = 128,
                    pruning_strategy: mask.Pruning = mask.NullPruning(), regrowth_strategy: mask.Regrowth = mask.NullRegrowth(), learning_rate: float = 0.01, minibatch_budget: int = 256):
    train_loader = DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=2, pin_memory=True)

    model.train()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)
    criterion = torch.nn.CrossEntropyLoss()
    total_batches = 0

    for epoch in range(epochs):
        model.train()

        mask.update_mask(model, pruning_strategy, regrowth_strategy)

        # TODO for debugging mask update rules
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

        print(f" ::: [Epoch {epoch+1:03d}] ::: batches: {total_batches:06d}, loss: {total_loss:7.4f}, test acc: {acc:.2f}%")

    return model

def train_ea_batch(model_factory, trainset, testset, train_batch: int = 64, test_batch: int = 128, device: torch.device = "cuda", generations: int = 128,
                   pruning_strategy: mask.Pruning = mask.NullPruning(), regrowth_strategy: mask.Regrowth = mask.NullRegrowth(), learning_rate: float = 0.01, minibatch_budget: int = 256, offspring: int = 10):
    train_loader = DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=2, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_model = model_factory().to(device)
    best_model.fitness = float("inf")

    total_batches = 0
    for gen in range(1, generations + 1):
        best_model.index = 0

        children = []
        for idx in range(offspring):
            child = model_factory().to(device)
            child.load_state_dict(best_model.state_dict())

            mask.update_mask(child, pruning_strategy, regrowth_strategy)
            child.train()

            child.index = idx + 1
            child.fitness = 0.0

            optimizer = torch.optim.SGD(child.parameters(), lr=learning_rate, momentum=0.0)
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

            print(f" --- Gen {gen:01d} | child {child.index:02d} | loss: {child.fitness:7.4f}")
        total_batches += minibatch_budget

        # sort based on fitness
        children.sort(key=lambda c: c.fitness)

        best_model = soft_top_k(children, select=1, decay_rate=1.75)[0]
        test_acc = evaluate(best_model, test_loader, device)

        print(f" ::: [Gen {gen:03d}] ::: parent | batches: {total_batches:06d}, survivor: {best_model.index:02d}, loss: {best_model.fitness:7.4f}, test acc: {test_acc:.2f}%")

    return best_model

def load_dataset() -> (DataLoader, DataLoader):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        torchvision.transforms.Lambda(lambda x: torch.flatten(x))  # TODO comment it out if you use SparseCNN
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return trainset, testset

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = torch.rand(1).item()
    sparsity = 0.5
    train_batch = 32
    test_batch = 512
    epochs = 50000
    offspring = 10

    pruning_strategy = mask.MagnitudePruning(pruned_fraction=0.3)
    regrowth_strategy = mask.UniformRegrowth()

    torch.manual_seed(seed)
    train_loader, test_loader = load_dataset()

    # model_factory = lambda: SparseCNN(in_channels=3, num_classes=10, num_blocks=5, hidden_channels=128, sparsity=sparsity)
    model_factory = lambda: SparseANN(3*32*32, 500, 3, sparsity=sparsity, layernorm=True)

    # train_sgd_batch(model_factory(), train_loader, test_loader, train_batch=train_batch, test_batch=test_batch, device=device, epochs=epochs, pruning_strategy=pruning_strategy, regrowth_strategy=regrowth_strategy)
    train_ea_batch(model_factory, train_loader, test_loader, train_batch=train_batch, test_batch=test_batch, device=device, generations=epochs, offspring=offspring, pruning_strategy=pruning_strategy, regrowth_strategy=regrowth_strategy)
