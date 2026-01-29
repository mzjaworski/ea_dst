from typing import List, Tuple
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
