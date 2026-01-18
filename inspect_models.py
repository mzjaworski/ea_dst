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
import pathlib

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import os



def leading_eigenvector_scalar(adj, max_iters=50, tol=1e-6, mode="mean"):
    """
    Compute a single summary scalar of leading eigenvector centrality.
    adj: [n x n] torch tensor, symmetric
    mode: "mean" or "max"
    """
    n = adj.size(0)
    v = torch.rand(n, device=adj.device)
    v = v / v.norm()

    for _ in range(max_iters):
        v_next = adj @ v
        v_next = v_next / (v_next.norm() + 1e-12)
        if torch.norm(v_next - v) < tol:
            break
        v = v_next

    v = v / v.max()  # normalize 0-1

    if mode == "mean":
        return v.mean().item()
    elif mode == "max":
        return v.max().item()
    else:
        raise ValueError("mode must be 'mean' or 'max'")

def compute_eigenvector_centrality(layers, mode="mean"):
    single_layer_centrality = []
    layer_pair_centrality = []

    for layer in layers:
        mask = layer.mask.float()
        adj = (mask @ mask.T)
        adj = (adj > 0).float()
        adj.fill_diagonal_(0)
        single_layer_centrality.append(leading_eigenvector_scalar(adj, mode=mode))

    for i in range(len(layers) - 1):
        mask1 = layers[i].mask.float()
        mask2 = layers[i+1].mask.float()
        composite = mask2 @ mask1
        adj = (composite.T @ composite)
        adj = (adj > 0).float()
        adj.fill_diagonal_(0)
        layer_pair_centrality.append(leading_eigenvector_scalar(adj, mode=mode))

    return {
        "single_layer": single_layer_centrality,  # scalar per layer
        "layer_pairs": layer_pair_centrality      # scalar per consecutive layer pair
    }



def compute_actual_clustering(layers):
    """
    Compute the actual graph-theory clustering coefficient for each layer
    and for each pair of consecutive layers.

    Args:
        layers: list of objects with `.mask` attribute (torch.Tensor, out x in, binary)

    Returns:
        dict with:
            "single_layer": list of avg clustering per layer
            "layer_pairs": list of avg clustering per consecutive layer pair
    """
    single_layer_clust = []
    pair_layer_clust = []

    # --- Single-layer clustering ---
    for layer in layers:
        mask = layer.mask.float()  # [out, in]
        # build output-output adjacency (connect outputs that share inputs)
        adj = (mask @ mask.T)  # [out, out]
        adj = (adj > 0).float()
        adj.fill_diagonal_(0)

        # triangles per node: diag(A^3)/2 for undirected
        triangles = torch.diagonal(adj @ adj @ adj) / 2
        deg = adj.sum(dim=1)
        clustering = torch.zeros_like(deg)
        mask_deg = deg > 1
        clustering[mask_deg] = (2 * triangles[mask_deg]) / (deg[mask_deg] * (deg[mask_deg] - 1))
        single_layer_clust.append(clustering.mean().item())

    # --- Consecutive layer pair clustering ---
    for i in range(len(layers) - 1):
        mask1 = layers[i].mask.float()
        mask2 = layers[i+1].mask.float()

        # build composite adjacency (inputs of layer1 connected if they reach same output in layer2)
        composite = mask2 @ mask1  # out2 x in1
        adj = (composite.T @ composite)  # square adjacency over inputs of layer1
        adj = (adj > 0).float()
        adj.fill_diagonal_(0)

        triangles = torch.diagonal(adj @ adj @ adj) / 2
        deg = adj.sum(dim=1)
        clustering = torch.zeros_like(deg)
        mask_deg = deg > 1
        clustering[mask_deg] = (2 * triangles[mask_deg]) / (deg[mask_deg] * (deg[mask_deg] - 1))
        pair_layer_clust.append(clustering.mean().item())

    return {
        "single_layer": single_layer_clust,
        "layer_pairs": pair_layer_clust
    }

def plot_layer_statistics(statistics, against: str = "accuracy", dataset: str = "none") -> None:

    layer_count = len(statistics[0]['layers'])
    layer_fields = list(statistics[0]['layers'][0].keys())
    other_fields = [key for key in list(statistics[0].keys()) if key not in ["path", "layers", against, "accuracy"]]

    pathlib.Path(f"data/network_analysis/{dataset}").mkdir(parents=True, exist_ok=True)

    for layer_field in layer_fields:
        for layer_index in range(layer_count):
            x = []
            y = []

            for group in statistics:
                x.append(group[against])
                y.append(group['layers'][layer_index][layer_field])

            plt.scatter(x, y)
            plt.xlabel(against)
            plt.ylabel(layer_field)
            plt.title(f"{against}-{layer_field} plot")
            plt.grid(True)

            plt.savefig(f"data/network_analysis/{dataset}/layer_{layer_index}|{against}_vs_{layer_field}.png", dpi=300, bbox_inches="tight")
            plt.close()

    for other_field in other_fields:
        x = []
        y = []

        for group in statistics:
            x.append(group[against])
            y.append(group[other_field])

        plt.scatter(x, y)
        plt.xlabel(against)
        plt.ylabel(other_field)
        plt.title(f"{against}-{other_field} plot")
        plt.grid(True)

        plt.savefig(f"data/network_analysis/{dataset}/{against}_vs_{other_field}.png", dpi=300, bbox_inches="tight")
        plt.close()

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
    criterion_factory = lambda: utilities.resolve_class(settings.criterion["type"])(**settings.criterion["kwargs"])

    model_directory = "/".join(settings.serialize["kwargs"]["path"].split("/")[:-2])
    device = "cuda:0"

    statistics = []
    for model_index, model_path in enumerate(pathlib.Path(model_directory).glob('**/*.pt')):

        # print(model_index)
        model = model_factory().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        statistics.append({"path": str(model_path)})

        statistics[-1]["accuracy"] = {}
        for (level, value) in training_utilities.top_k_accuracy(model, test_loader, device, [1]).items():
            statistics[-1]["accuracy"] = value

        statistics[-1]["layers"] = []

        layers = []
        for layer in model.modules():
            # we are interested only in custom MaskedLayers
            if not isinstance(layer, src.network.detail.MaskedLayer):
                continue
            layers.append(layer)

        clustering = compute_eigenvector_centrality(layers)
        for pair_index, value in enumerate(clustering["single_layer"]):
            statistics[-1][f"eigenvector_single_{pair_index}"] = value
        for pair_index, value in enumerate(clustering["layer_pairs"]):
            statistics[-1][f"eigenvector_pairs_{pair_index}"] = value

        for layer in layers:
            metrics = {}

            # compute number of isolated inputs/outputs
            if isinstance(layer.layer, torch.nn.Linear):

                deg_in = layer.mask.sum(dim=0)
                deg_out = layer.mask.sum(dim=1)

                metrics["i_in"] = round((deg_in == 0).sum().item() / layer.mask.size(1), 4)
                metrics["i_out"] = round((deg_out == 0).sum().item() / layer.mask.size(0), 4)
                metrics["deg_in_mean"] = round(deg_in[deg_in > 0].float().mean().item(), 4)
                metrics["deg_out_mean"] = round(deg_out[deg_out > 0].float().mean().item(), 4)

                metrics["in_overlap"] = round((layer.mask @ layer.mask.T > 1).float().mean().item(), 4)
                metrics["out_overlap"] = round((layer.mask.T @ layer.mask > 1).float().mean().item(), 4)

                u, s, v = torch.linalg.svd(layer.mask.float(), full_matrices=False)
                metrics["effective_rank"] = round((s.sum() ** 2 / (s ** 2).sum()).item(), 4)

            else:
                raise RuntimeError("unsupported layer type")

            statistics[-1]["layers"].append(metrics)
        # statistics[-1]["layers"] = statistics[-1]["layers"][-1:]

    statistics.sort(key=lambda x: x["accuracy"], reverse=True)
    for entry in statistics:
        print(entry)

    plot_layer_statistics(statistics, against="accuracy", dataset=settings.dataset["type"].split(".")[-1].lower())
