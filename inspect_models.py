import training_utilities
import utilities
import settings

from typing import List, Tuple, Union, Callable, Any
import secrets
import torch
import numpy
import json
import argparse
import random
import sys
import pathlib

import matplotlib.pyplot as plt
from tqdm import tqdm

from graph_tool.all import *
import hashlib
import scipy
from statsmodels.stats.multitest import multipletests
import pickle
import graph_utilities.operations

def generate_filename(s: str, length: int = 32) -> str:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h[:length]

def load_or_convert(model: torch.nn.Module, folder_name: str, model_path: str) -> Tuple[graph_tool.Graph, List[int]]:
    base_dir = pathlib.Path(f"data/network_analysis/network/{folder_name}")
    base_dir.mkdir(parents=True, exist_ok=True)

    stem = generate_filename(str(model_path))
    graph_path = base_dir / f"{stem}.xml.gz"
    meta_path  = base_dir / f"{stem}_meta.pkl"

    if graph_path.exists() and meta_path.exists():
        _graph = load_graph(str(graph_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return _graph, meta

    _graph, meta = graph.operations.convert_to_graph(model)

    _graph.save(str(graph_path))
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    return _graph, meta

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

def extract_metrics(datasets: List[List[dict]], accessors: List[Callable[[dict], Any]]) -> List[List[Any]]:
    results = [[] for _ in range(len(accessors))]

    # extract data
    for accessor_index, accessor in enumerate(accessors):
        try:
            for dataset_index, dataset in enumerate(datasets):
                for entry_index, entry in enumerate(dataset):
                    results[accessor_index].append(accessor(entry))
        except:
            pass #

    # generate default labels
    results.append([f"dataset_{dataset_index}" for dataset_index, dataset in enumerate(datasets) for entry in dataset])

    return results

def layerwise_profile(metrics: List[List[float]], performances: List[float], dataset_names: List[str], metric_name: str,
                      quantiles: Tuple[float, float] = (0.25, 0.75), y_limits: Tuple[Union[float, None], Union[float, None]] = (None, None)) -> plt.Figure:
    metrics = numpy.asarray(metrics)
    performances = numpy.asarray(performances)
    dataset_names = numpy.asarray(dataset_names)

    figure, axis = plt.subplots(figsize=(8, 5))

    palette = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"]
    unique_datasets = numpy.unique(dataset_names)
    color_map = {ds: palette[i % len(palette)] for i, ds in enumerate(unique_datasets)}

    for dataset_name in unique_datasets:
        idx = (dataset_names == dataset_name)

        perf = performances[idx]
        metric = metrics[idx]

        q_low = numpy.quantile(perf, quantiles[0])
        q_high = numpy.quantile(perf, quantiles[1])

        low = perf <= q_low
        high = perf >= q_high

        color = color_map[dataset_name]

        if low.any():
            axis.plot(metric[low].mean(axis=0), linestyle="--", linewidth=2, c=color, label=f"{dataset_name} (q={quantiles[0]} perf)")

        if high.any():
            axis.plot(metric[high].mean(axis=0), linestyle="-", linewidth=2, c=color, label=f"{dataset_name} (q={quantiles[1]} perf)")

    axis.set_xlabel("layer index")
    axis.set_ylabel(metric_name)
    axis.set_title(f"structural profile: {metric_name}")

    if y_limits != (None, None):
        axis.set_ylim(*y_limits)

    axis.legend()
    figure.tight_layout()

    return figure

def save_figure(figure: plt.Figure, folder: str = "", name: str = "") -> None:
    pathlib.Path(f"data/network_analysis/plots/{folder_name}").mkdir(parents=True, exist_ok=True)
    figure.savefig(f"data/network_analysis/plots/{folder_name}/{name}", dpi=300)

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

    if not isinstance(args.configuration, list):
        args.configuration = [args.configuration]

    all_statistics = []
    for configuration in args.configuration:

        # spawn global settings object
        config = settings.Settings(configuration)

        # instantiate referenced objects
        train_loader, test_loader = utilities.resolve_class(config.dataset["type"])(**config.dataset["kwargs"]).load()

        model_factory = lambda: utilities.resolve_class(config.model["type"])(**config.model["kwargs"])
        optimizer_factory = lambda x: utilities.resolve_class(config.optimizer["type"])(x.parameters(), **config.optimizer["kwargs"])
        criterion_factory = lambda: utilities.resolve_class(config.criterion["type"])(**config.criterion["kwargs"])
        device = "cuda:0"

        model_directory = "/".join(config.serialize["kwargs"]["path"].split("/")[:-2])
        folder_name = config.dataset["type"].split(".")[-1].lower()

        statistics_file_name = generate_filename("|".join(str(f) for f in pathlib.Path(model_directory).glob("**/*.pt")))
        statistics_path = pathlib.Path(f"data/network_analysis/statistics/{folder_name}/{statistics_file_name}.json")
        pathlib.Path(f"data/network_analysis/statistics/{folder_name}").mkdir(parents=True, exist_ok=True)

        dataset = []
        if False and statistics_path.exists(): # did we already process this data once?
            with open(statistics_path) as f:
                dataset = json.load(f)
        else: # no we process it for the first time

            for model_index, model_path in tqdm(enumerate(pathlib.Path(model_directory).glob('**/*.pt'))):
                data = {"path": str(model_path)}

                # load model
                model = model_factory().to(device)
                model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
                model.eval()

                # get its accuracy
                for (level, value) in training_utilities.top_k_accuracy(model, test_loader, device, [1]).items():
                    data["accuracy"] = value

                def derive_metrics(graph: graph_tool.Graph, neuron_counts: List[int], mask: Union[numpy.ndarray, None]) -> dict:

                    if mask is not None:
                        original_neuron_counts = neuron_counts
                        graph, neuron_counts = graph_utilities.operations.extract_from_mask(graph, neuron_counts, mask)

                    metrics = {
                        "neurons": [i for i in range(sum(neuron_counts))],
                        "layer_neurons": neuron_counts,
                        "in_degree": graph.degree_property_map("in").get_array().tolist(),
                        "out_degree": graph.degree_property_map("out").get_array().tolist(),
                        "total_degree": graph.degree_property_map("total").get_array().tolist(),
                        "pagerank": graph_tool.centrality.pagerank(graph).get_array().tolist(),
                        "betweenness": graph_tool.centrality.betweenness(graph)[0].get_array().tolist(),
                        "katz_centrality": graph_tool.centrality.katz(graph).get_array().tolist(),
                        "k_core": graph_tool.topology.kcore_decomposition(graph).get_array().tolist(),
                        "global": {},
                        "local": {}
                    }

                    # compute summary statistics
                    # TODO mean, std, median, quantiles

                    if mask is not None:
                        # store mask
                        metrics["mask"] = mask.tolist()

                        # compute and store path participation mass
                        metrics["global"]["ppm"] = numpy.mean(mask)
                        metrics["local"]["ppm"] = graph_utilities.operations.per_layer(original_neuron_counts, mask, numpy.mean)

                    return metrics

                # convert network and compute its metrics
                graph, neuron_counts = graph_utilities.operations.convert_to_graph(model)
                data["full"] = derive_metrics(graph, neuron_counts, None)
                data["forward"] = derive_metrics(graph, neuron_counts, graph_utilities.operations.reachable_from_input(graph, neuron_counts))
                data["backward"] = derive_metrics(graph, neuron_counts, graph_utilities.operations.reachable_from_output(graph, neuron_counts))
                data["functional"] = derive_metrics(graph, neuron_counts, graph_utilities.operations.functional_core(graph, neuron_counts))

                dataset.append(data)

            # with open(statistics_path, "w") as f:
            #     json.dump(dataset, f)

        dataset.sort(key=lambda x: x["accuracy"], reverse=True)
        all_statistics.append(dataset)
        settings.SingletonMeta._instances.pop(settings.Settings, None)

    # generate plots
    if len(all_statistics):

        # path participation mass
        local_ppm, accuracy, names = extract_metrics(all_statistics, [lambda x: x["functional"]["local"]["ppm"], lambda x: x["accuracy"]])
        figure = layerwise_profile(local_ppm, accuracy, names, "path participation mass", quantiles = (0.25, 0.75), y_limits = (None, None))
        save_figure(figure, folder=folder_name, name="path_participation_mass.png")



