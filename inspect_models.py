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
        except Exception as e:
            print(e)

    # generate default labels
    results.append([f"dataset_{dataset_index}" for dataset_index, dataset in enumerate(datasets) for entry in dataset])

    return results

def scatter_plot(x: List[float], y: List[float], dataset_names: List[str], x_label: str, y_label: str, title: str) -> plt.Figure:
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    dataset_names = numpy.asarray(dataset_names)

    figure, axis = plt.subplots(figsize=(8, 5))

    palette = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"]
    unique_datasets = numpy.unique(dataset_names)
    color_map = {ds: palette[i % len(palette)] for i, ds in enumerate(unique_datasets)}

    for dataset_name in unique_datasets:
        idx = (dataset_names == dataset_name)
        axis.scatter(x[idx], y[idx], color=color_map[dataset_name], label=dataset_name)

    axis.set_title(f"scatter plot: {title}")

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    axis.grid(True)
    axis.legend()

    return figure

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
        color = color_map[dataset_name]

        perf = performances[idx]
        metric = metrics[idx]

        q_low = numpy.quantile(perf, quantiles[0])
        q_high = numpy.quantile(perf, quantiles[1])

        low = perf <= q_low
        high = perf >= q_high

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
    pathlib.Path(f"data/network_analysis/plots/{folder}").mkdir(parents=True, exist_ok=True)
    figure.savefig(f"data/network_analysis/plots/{folder}/{name}", dpi=300)
    plt.close(figure)

def generate_histogram(metric: List[Union[int, float]], names: List[str], x_label: str, y_label: str, title: str) -> plt.Figure:
    flattened_values = numpy.concatenate(metric)
    dataset_names = numpy.array(names)

    vmin = numpy.min(flattened_values)
    vmax = numpy.max(flattened_values)

    # check if data is float or int
    if numpy.all(numpy.equal(flattened_values, numpy.floor(flattened_values))):
        bins = numpy.arange(vmin, vmax + 2)
    else:
        bins = numpy.linspace(vmin, vmax, 51)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    figure, axis = plt.subplots(figsize=(8, 5))

    palette = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"]
    unique_datasets = numpy.unique(dataset_names)
    color_map = {ds: palette[i % len(palette)] for i, ds in enumerate(unique_datasets)}

    for dataset_name in unique_datasets:
        indices = [i for i, m in enumerate(dataset_names == dataset_name) if m]

        values = []
        for index in indices:
            values.append(metric[index])

        histograms = numpy.array([numpy.histogram(sub, bins=bins, density=True)[0] for sub in values])
        avg_hist = histograms.mean(axis=0)
        avg_hist /= (avg_hist.sum() * bin_width)

        axis.bar(bin_centers, avg_hist, width=bin_width, alpha=0.6, color=color_map[dataset_name], edgecolor='black', label=f"{dataset_name}")
    axis.set_title(f"histogram: {title}")

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    axis.grid(True)
    axis.legend()

    return figure

# llm generated
def stacked_bar_plots(metrics: List[List[List[float]]], metric_names: List[str], names: List[str]) -> List[plt.Figure]:
    metrics = numpy.array(metrics)
    metric_names = numpy.array(metric_names)
    dataset_names = numpy.array(names)

    global_max = metrics.max()

    palette = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#fdae6b", "#a1d99b", "#e7cb94", "#bcbddc"]
    unique_datasets = numpy.unique(dataset_names)
    bar_width = 0.15
    entry_gap = 0.1

    figures = []

    for dataset_name in unique_datasets:
        idx = (dataset_names == dataset_name)

        data = metrics[:, idx, :].mean(axis=1).T
        num_entries, num_metrics = data.shape

        fig, ax = plt.subplots(figsize=(10, 5))

        entry_centers = []
        entry_labels = []

        current_x = 0
        for e in range(num_entries):
            x_positions = current_x + numpy.arange(num_metrics) * bar_width
            ax.bar(
                x_positions,
                data[e],
                width=bar_width,
                color=palette[:num_metrics],
                edgecolor="black",
                linewidth=0.7,
                label=metric_names if e == 0 else None
            )

            entry_center = x_positions[0] + (num_metrics * bar_width) / 2
            entry_centers.append(entry_center)
            entry_labels.append(f"Layer {e + 1}")

            current_x += num_metrics * bar_width + entry_gap

        ax.set_xticks(entry_centers)
        ax.set_xticklabels(entry_labels, rotation=45, ha="right")
        ax.set_ylim(0, global_max * 1.05)
        ax.set_ylabel("Value (normalized)")
        ax.set_title(f"Dataset: {dataset_name}")
        ax.legend(metric_names, ncol=min(len(metric_names), 4))

        fig.tight_layout()
        figures.append(fig)

    return figures

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
                        "raw": {
                            "neurons": [i for i in range(sum(neuron_counts))],
                            "layer_neurons": neuron_counts,
                            "in_degree": graph.degree_property_map("in").get_array().tolist(),
                            "out_degree": graph.degree_property_map("out").get_array().tolist(),
                            "total_degree": graph.degree_property_map("total").get_array().tolist(),
                            "pagerank": graph_tool.centrality.pagerank(graph).get_array().tolist(),
                            "betweenness": graph_tool.centrality.betweenness(graph)[0].get_array().tolist(),
                            "katz_centrality": graph_tool.centrality.katz(graph).get_array().tolist(),
                            "k_core": graph_tool.topology.kcore_decomposition(graph).get_array().tolist(),
                        },
                        "global": {},
                        "local": {}
                    }

                    # compute summary statistics
                    for metric_name in ["in_degree", "out_degree", "total_degree", "pagerank", "betweenness", "katz_centrality", "k_core"]:
                        metrics["global"][metric_name] = {
                            "mean": numpy.mean(metrics["raw"][metric_name]),
                            "median": numpy.median(metrics["raw"][metric_name]),
                            "std": numpy.std(metrics["raw"][metric_name]),
                            "min": numpy.min(metrics["raw"][metric_name]),
                            "max": numpy.max(metrics["raw"][metric_name])
                        }
                        metrics["local"][metric_name] = {
                            "mean": graph_utilities.operations.per_layer(neuron_counts, metrics["raw"][metric_name], numpy.mean),
                            "median": graph_utilities.operations.per_layer(neuron_counts, metrics["raw"][metric_name], numpy.median),
                            "std": graph_utilities.operations.per_layer(neuron_counts, metrics["raw"][metric_name], numpy.std),
                            "min": graph_utilities.operations.per_layer(neuron_counts, metrics["raw"][metric_name], numpy.min),
                            "max": graph_utilities.operations.per_layer(neuron_counts, metrics["raw"][metric_name], numpy.max)
                        }

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


    # ------------------------------------------------ generate plots ------------------------------------------------ #
    if len(all_statistics):
        metrics = ["in_degree", "out_degree", "total_degree", "pagerank", "betweenness", "katz_centrality", "k_core"]
        statistics = ["mean", "median", "std", "min", "max"]

        # global statistics
        x_field = "accuracy"
        for metric in metrics:
            for statistic in statistics:
                x, y, names = extract_metrics(all_statistics, [ lambda x: x[x_field], lambda x: x["functional"]["global"][metric][statistic]])
                figure = scatter_plot(x, y, names, x_label=x_field, y_label=f"{statistic}_{metric}", title=f"{x_field} vs {statistic}_{metric}")
                save_figure(figure, folder=f"{folder_name}/statistics", name=f"global.{x_field}_vs_{statistic}_{metric}.png")

        # path participation mass
        local_ppm, accuracy, names = extract_metrics(all_statistics, [lambda x: x["functional"]["local"]["ppm"], lambda x: x["accuracy"]])
        figure = layerwise_profile(local_ppm, accuracy, names, "path participation mass", quantiles = (0.25, 0.75), y_limits = (None, None))
        save_figure(figure, folder=folder_name, name="path_participation_mass.png")

        # connected neuron distributions
        full, forward, backward, functional, names = extract_metrics(all_statistics, [lambda x: x["full"]["raw"]["layer_neurons"], lambda x: x["forward"]["raw"]["layer_neurons"], lambda x: x["backward"]["raw"]["layer_neurons"], lambda x: x["functional"]["raw"]["layer_neurons"]])
        figures = stacked_bar_plots([full, forward, backward, functional], ["full", "forward", "backward", "functional"], names)
        for (figure, name) in zip(figures, numpy.unique(numpy.array(names))):
            save_figure(figure, folder=f"{folder_name}/bar_plots", name=f"layer_naurons.{name}.png")

        # histograms
        for metric in metrics:

            # global
            statistics, names = extract_metrics(all_statistics,[lambda x: x["full"]["raw"][metric]])
            figure = generate_histogram(statistics, names, x_label=metric, y_label="frequency", title=f"{metric} frequency")
            save_figure(figure, folder=f"{folder_name}/histograms", name=f"full.global.{metric}.png")

            # local
            for index in range(len(all_statistics[0][0]["full"]["raw"]["layer_neurons"])): # a bit hacky but works
                statistics, names = extract_metrics(all_statistics, [lambda x: graph_utilities.operations.per_layer(x["full"]["raw"]["layer_neurons"], x["full"]["raw"][metric], lambda x: x)[index]])
                figure = generate_histogram(statistics, names, x_label=metric, y_label="frequency", title=f"{metric} frequency")
                save_figure(figure, folder=f"{folder_name}/histograms", name=f"full.local.{index}.{metric}.png")
