from typing import Tuple, List
import src.network.custom
import src.network.detail
import training_utilities
import pandas
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

layers = [[3072, 1024], [1024, 512], [512, 512], [512, 100]]
round_to = 3
samples = 100

column_names = [[f"l{layer_index+1} predicted forward", f"l{layer_index+1} realized forward",
                 f"l{layer_index+1} predicted backward", f"l{layer_index+1} realized backward",
                 f"l{layer_index+1} predicted functional", f"l{layer_index+1} realized functional"] for layer_index in range(len(layers))]
column_names = numpy.array(column_names).flatten().tolist()

data = pandas.DataFrame(columns=column_names)
data.index.name = "sparsity"

for sparsity_level in numpy.linspace(0.99, 0.9999, 10):

    sparsities = None
    averages = [[] for _ in range(len(layers))]

    for _ in range(samples):
        network = src.network.custom.SparseANN(layers=layers, sparsity=[sparsity_level, sparsity_level, sparsity_level, 0.0], act="relu", layernorm=True, method="uniform")
        network.compute_reachabilities()

        layer_sparsities = [module.realized_sparsity() for module in network.modules() if isinstance(module, src.network.detail.MaskedLayer)]
        sparsities = [a + b for (a, b) in zip(layer_sparsities, sparsities)] if sparsities is not None else layer_sparsities

        current = training_utilities.edge_reachability(network)

        for entry_index in range(len(current)):
            averages[entry_index].append(current[entry_index])

    realized_averages = [tuple(numpy.asarray(array, dtype=float).mean(axis=0)) for array in averages]
    realized_std = [tuple(numpy.asarray(array, dtype=float).std(axis=0)) for array in averages]

    realized_sparsities = [sparsity / samples for sparsity in sparsities] # average
    forward, backward, functional = predict_connectivity(layers, realized_sparsities)

    predicted_sparsities = [(fwd, bwd, func) for (fwd, bwd, func) in zip(forward, backward, functional)]

    # format data entries
    row_data = []
    for entry_index in range(len(forward)):
        row_data.append(f"{round(predicted_sparsities[entry_index][0], round_to):.{round_to}f}")
        row_data.append(f"{round(realized_averages[entry_index][0], round_to):.{round_to}f} ± {round(realized_std[entry_index][0], round_to):.{round_to}f}")
        row_data.append(f"{round(predicted_sparsities[entry_index][1], round_to):.{round_to}f}")
        row_data.append(f"{round(realized_averages[entry_index][1], round_to):.{round_to}f} ± {round(realized_std[entry_index][1], round_to):.{round_to}f}")
        row_data.append(f"{round(predicted_sparsities[entry_index][2], round_to):.{round_to}f}")
        row_data.append(f"{round(realized_averages[entry_index][2], round_to):.{round_to}f} ± {round(realized_std[entry_index][2], round_to):.{round_to}f}")

    data.loc[f"{round(sparsity_level, round_to+1):.{round_to+1}f}"] = row_data

# print(data)
print(data.to_latex())