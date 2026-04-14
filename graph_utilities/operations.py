from typing import List, Tuple, Any, Dict
import matplotlib.pyplot as plt
from graph_tool.all import *
import numpy
import torch
from PIL import Image
import os

import src.network.detail

# converts feed forward network to a graph
def convert_to_graph(model: src.network.detail.NetworkInterface) -> Tuple[graph_tool.Graph, List[int]]:
    # ensure reachability matrices are populated inside of MaskedLayer
    model.compute_reachabilities()

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
    graph = graph_tool.Graph(directed=True)
    connectivity = graph.new_edge_property("string")
    vertices = []

    # create vertices for input layer
    input_vertices = [graph.add_vertex() for _ in range(layers[0].mask.shape[1])]
    vertices.append(input_vertices)

    # now create vertices for every layer output
    for layer in layers:
        layer_vertices = [graph.add_vertex() for _ in range(layer.mask.shape[0])]
        vertices.append(layer_vertices)

    # reachability modes; n = none, f = forward, b = backward, fb = forward+backward
    reachability_map = {0: "n", 1: "f", 2: "b", 3: "fb"}

    # add edges according to each layers mask
    for layer_index, layer in enumerate(layers):

        # indexing native type is orders of magnitude faster
        forward = layer.forward_reachability.tolist()
        backward = layer.backward_reachability.tolist()
        mask = layer.mask.tolist()

        out_dim, in_dim = layer.mask.shape

        for i in range(out_dim):
            for j in range(in_dim):
                if mask[i][j]:
                    edge = graph.add_edge(vertices[layer_index][j], vertices[layer_index + 1][i])
                    connectivity[edge] = reachability_map[(int(forward[i][j]) * 1) + (int(backward[i][j]) * 2)]

    graph.edge_properties["connectivity"] = connectivity

    return graph, [len(layer) for layer in vertices]

def downsample_image(input_path: str, output_path: str = None, max_size: tuple = (1920, 1920), quality: int = 85) -> str:
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_small.jpg"

    with Image.open(input_path) as img:
        img = img.convert("RGB")
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img.save(output_path, format="JPEG", quality=quality, optimize=True)

    return output_path

# llm assisted
def layered_graph_figure(model: src.network.detail.NetworkInterface, filename: str, width: int = 8000, height: int = 6000, margin: float = 100.0, vertex_size: float = 12, edge_pen_width: float = 1.0, colors: Dict[str, Tuple[float, float, float, float]] = None):
    from graph_tool.all import Graph, graph_draw
    graph, neuron_counts = convert_to_graph(model)

    g = Graph(directed=graph.is_directed())

    vertex_map = {v: g.add_vertex() for v in graph.vertices()}

    connectivity_in = graph.edge_properties["connectivity"]
    connectivity_out = g.new_edge_property("string")

    for e in graph.edges():
        new_e = g.add_edge(vertex_map[e.source()], vertex_map[e.target()])
        connectivity_out[new_e] = connectivity_in[e]

    g.edge_properties["connectivity"] = connectivity_out

    edge_color = g.new_edge_property("vector<double>")

    if colors is None:
        colors = default_color_map(highlight="no")

    for e in g.edges():
        edge_color[e] = colors.get(connectivity_out[e], (0, 0, 0, 1))

    pos = g.new_vertex_property("vector<double>")

    n_layers = len(neuron_counts)
    if n_layers == 1:
        x_spacing = width - 2 * margin
    else:
        x_spacing = (width - 2 * margin) / (n_layers - 1)

    vertices = list(g.vertices())
    v_idx = 0
    for layer_idx, n in enumerate(neuron_counts):
        x = margin + layer_idx * x_spacing
        if n == 1:
            y_spacing = 0
        else:
            y_spacing = (height - 2 * margin) / (n - 1)

        y0 = (height - (n - 1) * y_spacing) / 2
        for j in range(n):
            pos[vertices[v_idx]] = (x, y0 + j * y_spacing)
            v_idx += 1

    graph_draw(g, pos=pos, vertex_size=vertex_size, edge_pen_width=edge_pen_width, edge_color=edge_color, output_size=(width, height), output=filename, bg_color=[1, 1, 1, 1])
    downsample_image(filename)

# sloppy copy from training_utilities
def edge_reachability(model: torch.nn.Module) -> List[Tuple[float, float, float]]:
    modules = [module for module in model.modules() if isinstance(module, src.network.detail.MaskedLayer)]
    edge_reachabilities = []

    for module in modules:
        forward_reachability = torch.sum(module.forward_reachability * module.mask != 0).item()
        backward_reachability = torch.sum(module.backward_reachability * module.mask != 0).item()
        functional_reachability = torch.sum((module.forward_reachability.to(torch.bool) & module.backward_reachability.to(torch.bool)) * module.mask != 0).item()
        edge_reachabilities.append((forward_reachability, backward_reachability, functional_reachability))

    return edge_reachabilities

# sloppy copy from training_utilities
def normalized_edge_reachability(model: torch.nn.Module) -> List[Tuple[float, float, float]]:
    modules = [module for module in model.modules() if isinstance(module, src.network.detail.MaskedLayer)]
    edge_reachabilities = []

    for module in modules:
        forward_reachability = torch.sum(module.forward_reachability * module.mask != 0).item() / module.active_weights() if module.active_weights() > 0 else 0
        backward_reachability = torch.sum(module.backward_reachability * module.mask != 0).item() / module.active_weights() if module.active_weights() > 0 else 0
        functional_reachability = torch.sum((module.forward_reachability.to(torch.bool) & module.backward_reachability.to(torch.bool)) * module.mask != 0).item() / module.active_weights() if module.active_weights() > 0 else 0
        edge_reachabilities.append((forward_reachability, backward_reachability, functional_reachability))

    return edge_reachabilities


# llm assisted
def edge_reachability_figure(model: src.network.detail.NetworkInterface, filename: str) -> None:

    # plotting helper
    def plot_grouped_bars(data: List[tuple], filename: str, metric_names: List[str]) -> None:

        data = numpy.array(data)  # shape: (num_entries, 3)
        num_entries, num_metrics = data.shape
        global_max = data.max()

        palette = ["#6baed6", "#fd8d3c", "#74c476"]  # one color per metric
        bar_width = 0.25
        entry_gap = 0.15

        fig, ax = plt.subplots(figsize=(10, 5))

        entry_centers = []
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

            # center of the entry for x-tick
            entry_center = x_positions[0] + (num_metrics * bar_width) / 2
            entry_centers.append(entry_center)

            current_x += num_metrics * bar_width + entry_gap

        ax.set_xticks(entry_centers)
        ax.set_xticklabels([f"Entry {i + 1}" for i in range(num_entries)], rotation=45, ha="right")
        ax.set_ylim(0, global_max * 1.05)
        ax.set_ylabel("Value")
        ax.set_title("Triplet Metrics (Grouped)")
        ax.legend(metric_names, ncol=num_metrics)
        fig.tight_layout()

        # Save the figure
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    model.compute_reachabilities()
    reachabilities = normalized_edge_reachability(model)
    plot_grouped_bars(reachabilities, filename, metric_names=["Forward", "Backward", "Functional"])

def default_color_map(highlight: str = "f", alpha: float = 1.0) -> Dict[str, Tuple[float, float, float, float]]:

    match highlight:
        case "no":
            return {
                "n": (0.575, 0.575, 0.575, alpha), # gray
                "f": (0.575, 0.575, 0.575, alpha), # gray
                "b": (0.575, 0.575, 0.575, alpha), # gray
                "fb": (0.575, 0.575, 0.575, alpha), # gray
            }
        case "f":
            return {
                "n": (0.575, 0.575, 0.575, alpha / 1.5), # gray
                "f": (0.20, 0.60, 0.35, alpha), # dark green
                "b": (0.575, 0.575, 0.575, alpha / 1.5), # gray
                "fb": (0.20, 0.60, 0.35, alpha), # gray
            }
        case "b":
            return {
                "n": (0.575, 0.575, 0.575, alpha / 1.5), # gray
                "f": (0.575, 0.575, 0.575, alpha / 1.5), # gray
                "b": (0.70, 0.25, 0.25, alpha), # dark red/orange
                "fb": (0.70, 0.25, 0.25, alpha), # gray
            }
        case "fb":
            return {
                "n": (0.575, 0.575, 0.575, alpha / 1.5), # gray
                "f": (0.575, 0.575, 0.575, alpha / 1.5), # gray
                "b": (0.575, 0.575, 0.575, alpha / 1.5), # gray
                "fb": (0.35, 0.35, 0.65, alpha), # dark muted purple/blue blend
            }
        case _:
            raise RuntimeError("unsupported color scheme")
