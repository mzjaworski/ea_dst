from typing import Tuple, List
import src.network.custom
import src.network.detail
import training_utilities
from tqdm import tqdm
import numpy


def predict_connectivity(layer_sizes: List[Tuple[int, int]], sparsities: List[float]):
    L = len(layer_sizes)

    p = [1 - s for s in sparsities]

    # |V_l|
    V = [layer_sizes[0][0]] + [n_out for _, n_out in layer_sizes]

    # vertex part
    R_fwd_v = numpy.zeros(L + 1)
    R_fwd_v[0] = V[0]

    for l in range(1, L + 1):
        R_fwd_v[l] = V[l] * (1 - (1 - p[l-1]) ** R_fwd_v[l-1])

    R_bwd_v = numpy.zeros(L + 1)
    R_bwd_v[L] = V[L]

    for l in reversed(range(L)):
        R_bwd_v[l] = V[l] * (1 - (1 - p[l]) ** R_bwd_v[l+1])

    # edge part
    R_fwd_e = numpy.zeros(L)
    R_bwd_e = numpy.zeros(L)
    R_fcnl_e = numpy.zeros(L)

    for l in range(L):
        V_prev = V[l]
        V_curr = V[l+1]

        R_fwd_e[l] = p[l] * V_curr * R_fwd_v[l] / (V[l] * V[l+1] * p[l])
        R_bwd_e[l] = p[l] * V_prev * R_bwd_v[l+1] / (V[l] * V[l+1] * p[l])
        R_fcnl_e[l] = p[l] * R_fwd_v[l] * R_bwd_v[l+1] / (V[l] * V[l+1] * p[l])

    return R_fwd_e.tolist(), R_bwd_e.tolist(), R_fcnl_e.tolist()

layers = [[3072, 1024], [1024, 512], [512, 512], [512, 768], [768, 512], [512, 368], [368, 512], [512, 100]]
samples = 1

for sparsity_level in numpy.linspace(0.99, 0.9999, 10):
    print(f"# ----------------------------------------------- {round(sparsity_level, 4):.4f} ----------------------------------------------- #")

    sparsities = None
    averages = None

    for _ in range(samples):
        network = src.network.custom.SparseANN(layers=layers, sparsity=sparsity_level, act="relu", layernorm=True, method="erk")
        network.compute_reachabilities()

        layer_sparsities = [module.realized_sparsity() for module in network.modules() if isinstance(module, src.network.detail.MaskedLayer)]
        sparsities = [a + b for (a, b) in zip(layer_sparsities, sparsities)] if sparsities is not None else layer_sparsities

        current = training_utilities.edge_reachability(network)
        averages = [(x[0] + y[0], x[1] + y[1], x[2] + y[2]) for x, y in zip(averages, current)] if averages is not None else current

    averages = [(round(x / samples, 4), round(y / samples, 4), round(z / samples, 4)) for (x, y, z) in averages]
    sparsities = [sparsity / samples for sparsity in sparsities]

    forward, backward, functional = predict_connectivity(layers, sparsities)

    print([(round(a, 4), round(b, 4), round(c, 4)) for (a, b, c) in zip(forward, backward, functional)])
    print(averages)
