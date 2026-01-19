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
from tqdm import tqdm

from graph_tool.all import *
import hashlib

def generate_filename(s: str, length: int = 32) -> str:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h[:length]

# llm generated
def gini(x: numpy.ndarray) -> float:
    if x.size == 0:
        return numpy.nan

    x = numpy.asarray(x, dtype=numpy.float64)
    if numpy.all(x == 0):
        return 0.0

    x = numpy.sort(x)
    n = x.size
    cumx = numpy.cumsum(x)

    return (n + 1 - 2 * numpy.sum(cumx) / cumx[-1]) / n

# llm generated
def build_accessor_map(obj, prefix=""):
    fields = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            fields.update(build_accessor_map(v, path))
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            path = f"{prefix}.{idx}" if prefix else str(idx)
            fields.update(build_accessor_map(v, path))
    else:
        fields[prefix] = lambda x, p=prefix: _get_by_path(x, p)

    return fields

# llm generated
def _get_by_path(obj, path):
    keys = path.split(".")
    val = obj
    for key in keys:
        if key.isdigit():
            val = val[int(key)]
        else:
            val = val[key]
    return val

def plot_dataset(statistics, folder_name: str = "none") -> None:
    pathlib.Path(f"data/network_analysis/plots/{folder_name}").mkdir(parents=True, exist_ok=True)

    # gather x axis
    x_field_name = "accuracy"
    x = [x[x_field_name] for x in statistics]

    # scan for fields
    accessor_map = build_accessor_map(statistics[0]["metrics"])

    # plot for all fields treated as y
    for y_field_name, accessor in accessor_map.items():
        y = [accessor(x["metrics"]) for x in statistics]

        plt.scatter(x, y)
        plt.xlabel(x_field_name)
        plt.ylabel(y_field_name)
        plt.grid(True)

        plt.savefig(f"data/network_analysis/plots/{folder_name}/{x_field_name}_vs_{y_field_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

def convert_to_graph(model: torch.nn.Module) -> Tuple[Graph, List[int]]:

    # collect only custom layers
    layers = []
    for layer in model.modules():
        if not isinstance(layer, src.network.detail.MaskedLayer):
            continue
        if not isinstance(layer.layer, torch.nn.Linear):
            raise RuntimeError("currently only linear layers are supported!")
        layers.append(layer)

    if not layers:
        raise RuntimeError("No masked layers found!")

    # create graph
    graph = Graph(directed=True)
    vertices = []

    # create vertices for input layer
    input_vertices = [graph.add_vertex() for _ in range(layers[0].mask.shape[1])]
    vertices.append(input_vertices)

    # now create vertices for every layer output
    for layer in layers:
        layer_vertices = [graph.add_vertex() for _ in range(layer.mask.shape[0])]
        vertices.append(layer_vertices)

    # add edges according to each layers mask
    for layer_index, layer in enumerate(layers):
        mask = layer.mask
        out_dim, in_dim = mask.shape

        for i in range(out_dim):
            for j in range(in_dim):
                if mask[i, j]:
                    graph.add_edge(vertices[layer_index][j], vertices[layer_index + 1][i])

    return graph, [len(layer) for layer in vertices]

def global_statistics(graph: Graph, vertex_property, layer_sizes: list, connected_only: bool = False) -> dict:
    layer_start = 0

    values = []
    for layer_size in layer_sizes:
        for vertex_index in range(layer_start, layer_start + layer_size):
            vertex = graph.vertex(vertex_index)

            if connected_only and (vertex.in_degree() == 0 and vertex.out_degree() == 0):
                continue

            values.append(vertex_property[vertex])
        layer_start += layer_size
    values = numpy.array(values)

    return {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'median': float(numpy.median(values)),
        'gini': float(gini(values))
    }

def local_statistics(graph: Graph, vertex_property, layer_sizes: list, connected_only: bool = False) -> List[dict]:
    layer_start = 0
    local_stats = []

    for layer_size in layer_sizes:

        layer_values = []
        for vertex_index in range(layer_start, layer_start + layer_size):
            vertex = graph.vertex(vertex_index)

            if connected_only and (vertex.in_degree() == 0 and vertex.out_degree() == 0):
                continue

            layer_values.append(vertex_property[vertex])
        layer_values = numpy.array(layer_values)

        local_stats.append({
            'mean': float(layer_values.mean()),
            'std': float(layer_values.std()),
            'median': float(numpy.median(layer_values)),
            'gini': float(gini(layer_values))
        })

        layer_start += layer_size

    return local_stats


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
    device = "cuda:0"

    model_directory = "/".join(settings.serialize["kwargs"]["path"].split("/")[:-2])
    folder_name = settings.dataset["type"].split(".")[-1].lower()

    dataset_file_name = generate_filename("|".join(str(f) for f in pathlib.Path(model_directory).glob("**/*.pt")))
    dataset_path = pathlib.Path(f"data/network_analysis/dataset/{folder_name}/{dataset_file_name}.json")
    pathlib.Path(f"data/network_analysis/dataset/{folder_name}").mkdir(parents=True, exist_ok=True)

    dataset = []
    if dataset_path.exists(): # did we already process this data once?
        with open(dataset_path) as f:
            dataset = json.load(f)
    else: # no we process it for the first time

        for model_index, model_path in tqdm(enumerate(pathlib.Path(model_directory).glob('**/*.pt'))):
            data = {"path": str(model_path)}

            # load model
            model = model_factory().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # get its accuracy
            for (level, value) in training_utilities.top_k_accuracy(model, test_loader, device, [1]).items():
                data["accuracy"] = value

            # check if we already converted this network before
            graph, neuron_counts = convert_to_graph(model)

            # spawn major fields
            data["metrics"] = {}
            data["metrics"]["global"] = {}
            data["metrics"]["local"] = {}

            # not defined for DAG-s
            # eigenvalue, eigenvector = graph_tool.centrality.eigenvector(graph)
            # data["metrics"]["global"]['eigenvalue'] = eigenvalue
            # data["metrics"]["global"]['eigenvector'] = global_statistics(graph, eigenvector)

            in_degree = graph.degree_property_map("in")
            data["metrics"]["global"]['in_degree'] = global_statistics(graph, in_degree, neuron_counts, connected_only=True)
            data["metrics"]["local"]['in_degree'] = local_statistics(graph, in_degree, neuron_counts, connected_only=True)

            out_degree = graph.degree_property_map("out")
            data["metrics"]["global"]['out_degree'] = global_statistics(graph, out_degree, neuron_counts, connected_only=True)
            data["metrics"]["local"]['out_degree'] = local_statistics(graph, out_degree, neuron_counts, connected_only=True)

            total_degree = graph.degree_property_map("total")
            data["metrics"]["global"]['total_degree'] = global_statistics(graph, total_degree, neuron_counts, connected_only=True)
            data["metrics"]["local"]['total_degree'] = local_statistics(graph, total_degree, neuron_counts, connected_only=True)

            pagerank = graph_tool.centrality.pagerank(graph)
            data["metrics"]["global"]['pagerank'] = global_statistics(graph, pagerank, neuron_counts, connected_only=True)
            data["metrics"]["local"]['pagerank'] = local_statistics(graph, pagerank, neuron_counts, connected_only=True)

            vertex_betweenness, _ = graph_tool.centrality.betweenness(graph)
            data["metrics"]["global"]['betweenness'] = global_statistics(graph, vertex_betweenness, neuron_counts, connected_only=True)
            data["metrics"]["local"]['betweenness'] = local_statistics(graph, vertex_betweenness, neuron_counts, connected_only=True)

            closeness = graph_tool.centrality.closeness(graph)
            data["metrics"]["global"]['closeness'] = global_statistics(graph, closeness, neuron_counts, connected_only=True)
            data["metrics"]["local"]['closeness'] = local_statistics(graph, closeness, neuron_counts, connected_only=True)

            katz_centrality = graph_tool.centrality.katz(graph)
            data["metrics"]["global"]['katz_centrality'] = global_statistics(graph, katz_centrality, neuron_counts, connected_only=True)
            data["metrics"]["local"]['katz_centrality'] = local_statistics(graph, katz_centrality, neuron_counts, connected_only=True)

            shortest_distance = graph_tool.topology.shortest_distance(graph)
            data["metrics"]["global"]['shortest_distance'] = global_statistics(graph, shortest_distance, neuron_counts, connected_only=True)
            data["metrics"]["local"]['shortest_distance'] = local_statistics(graph, shortest_distance, neuron_counts, connected_only=True)

            k_core = graph_tool.topology.kcore_decomposition(graph)
            data["metrics"]["global"]['k_core'] = global_statistics(graph, k_core, neuron_counts, connected_only=True)
            data["metrics"]["local"]['k_core'] = local_statistics(graph, k_core, neuron_counts, connected_only=True)

            dataset.append(data)

        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

    dataset.sort(key=lambda x: x["accuracy"], reverse=True)
    for entry in dataset:
        print(entry)

    plot_dataset(dataset, folder_name=folder_name)
