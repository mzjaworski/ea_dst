from typing import List, Tuple, Any
from graph_tool.all import *
import numpy
import torch

import src.network.detail

def convert_to_graph(model: torch.nn.Module) -> Tuple[graph_tool.Graph, List[int]]:

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
        mask = layer.mask.tolist() # indexing native type is orders of magnitude faster
        out_dim, in_dim = layer.mask.shape

        for i in range(out_dim):
            for j in range(in_dim):
                if mask[i][j]:
                    graph.add_edge(vertices[layer_index][j], vertices[layer_index + 1][i])

    return graph, [len(layer) for layer in vertices]

def reachable_from(graph: Graph, source: int) -> numpy.ndarray:
    class ReachabilityVisitor(BFSVisitor):
        def __init__(self, reached: numpy.ndarray):
            self.reached = reached

        def discover_vertex(self, u):
            self.reached[int(u)] = True

    visitor = ReachabilityVisitor(numpy.zeros(graph.num_vertices(), dtype=bool))
    bfs_search(graph, graph.vertex(source), visitor)

    return visitor.reached

def reachable_from_input(graph: Graph, neuron_counts: list[int]) -> numpy.ndarray:
    g = graph.copy()

    phantom = g.add_vertex()
    n_inputs = neuron_counts[0]

    for v in range(n_inputs):
        g.add_edge(phantom, g.vertex(v))

    reached = reachable_from(g, int(phantom))

    return reached[:-1]

def reachable_from_output(graph: Graph, neuron_counts: list[int]) -> numpy.ndarray:
    g = graph.copy()
    g.set_reversed(True)

    phantom = g.add_vertex()
    offset = sum(neuron_counts[:-1])

    for v in range(neuron_counts[-1]):
        g.add_edge(phantom, g.vertex(offset + v))

    reached = reachable_from(g, int(phantom))

    return reached[:-1]

def functional_core(graph: Graph, neuron_counts: list[int]) -> numpy.ndarray:
    return reachable_from_input(graph, neuron_counts) & reachable_from_output(graph, neuron_counts)

def extract_from_mask(graph: Graph, neuron_counts: list[int], mask: numpy.ndarray) -> Tuple[Graph, List[int]]:
    core_graph = Graph(directed=graph.is_directed())
    core_neuron_counts = []

    # propagate vertices
    vertex_map = {}
    for v in graph.vertices():
        idx = int(v)
        if mask[idx]:
            vertex_map[idx] = int(core_graph.add_vertex())

    # propagate edges
    for e in graph.edges():
        u = int(e.source())
        v = int(e.target())
        if mask[u] and mask[v]:
            core_graph.add_edge(core_graph.vertex(vertex_map[u]), core_graph.vertex(vertex_map[v]))

    # update number of neuron_counts
    start = 0
    for layer_size in neuron_counts:
        core_neuron_counts.append(int(numpy.sum(mask[start : start + layer_size])))
        start += layer_size

    return core_graph, core_neuron_counts

def extract_functional_core(graph: Graph, neuron_counts: list[int]) -> Tuple[Graph, List[int]]:
    return extract_from_mask(graph, neuron_counts, functional_core(graph, neuron_counts))

def extract_input_reachable(graph: Graph, neuron_counts: list[int]) -> Tuple[Graph, List[int]]:
    return extract_from_mask(graph, neuron_counts, reachable_from_input(graph, neuron_counts))

def extract_output_reachable(graph: Graph, neuron_counts: list[int]) -> Tuple[Graph, List[int]]:
    return extract_from_mask(graph, neuron_counts, reachable_from_output(graph, neuron_counts))

def per_layer(neuron_counts: list[int], data: numpy.ndarray, metric = numpy.sum) -> List[Any]:
    results = []

    start = 0
    for neuron_count in neuron_counts:
        results.append(metric(data[start : start + neuron_count]))
        start += neuron_count

    return results

def interlayer_connectivity(neuron_counts: list[int], forward_mask: numpy.ndarray, backward_mask: numpy.ndarray, operand = numpy.outer) -> List[numpy.ndarray]:
    assert len(forward_mask) == len(backward_mask)
    assert sum(neuron_counts) == len(forward_mask)

    result = []

    # first gather all matrices
    forward_layers = []
    backward_layers = []

    start = 0
    for layer_neuron_count in neuron_counts:
        forward_layers.append(forward_mask[start : start + layer_neuron_count])
        backward_layers.append(backward_mask[start : start + layer_neuron_count])
        start += layer_neuron_count

    # now assemble reachability matrices per each layer
    for layer_index in range(len(forward_layers) - 1):
        result.append(operand(forward_layers[layer_index + 1], backward_layers[layer_index]))

    return result

def edge_reachability(model: torch.nn.Module, edge_connectivity: List[numpy.ndarray]) -> List[int]:

    # collect only custom layers
    layers = []
    for layer in model.modules():
        if not isinstance(layer, src.network.detail.MaskedLayer):
            continue
        if not isinstance(layer.layer, torch.nn.Linear):
            raise RuntimeError("currently only linear layers are supported!")
        layers.append(layer)

    assert len(edge_connectivity) == len(layers)

    # compute overlap between supplied theoretical edge connectivity and actual network
    edge_overlap = []
    for (layer, connectivity_matrix) in zip(layers, edge_connectivity):
        mask = layer.mask.cpu().numpy()
        edge_overlap.append(float(numpy.sum(mask.astype(bool) & connectivity_matrix.astype(bool)).astype(float) / numpy.sum(mask)))

    return edge_overlap

# llm generated
def draw_layered_graph(graph: Graph, neuron_counts: list[int], filename: str, width: int = 8000, height: int = 6000, margin: float = 100.0, vertex_size: float = 12, edge_pen_width: float = 1.0):
    g = Graph(directed=graph.is_directed())

    vertex_map = {v: g.add_vertex() for v in graph.vertices()}
    for e in graph.edges():
        g.add_edge(vertex_map[e.source()], vertex_map[e.target()])

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

    graph_draw(g, pos=pos, vertex_size=vertex_size, edge_pen_width=edge_pen_width, output_size=(width, height), output=filename)
