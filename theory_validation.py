from typing import Tuple, List
import src.network.custom
import src.network.detail
import training_utilities
from tqdm import tqdm
import pandas
import numpy
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import PercentFormatter

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

def generate_artifacts(data: pandas.DataFrame, layers: int, folder: str) -> None:
    def read_from_row(row: str) -> Tuple[float, float]:
        parts = [p.strip() for p in row.split("±")]
        return tuple(float(p) for p in parts)

    def read_column(data: pandas.DataFrame, index: int) -> List[List[float]]:

        if index == 0:
            column_data = data.index.astype(str).tolist()
        else:
            column_data = data.iloc[:, index - 1].astype(str).tolist()

        extracted = []
        for cell in column_data:
            if "±" in cell:
                extracted.append(read_from_row(cell))
            else:
                extracted.append([float(cell)])

        if not extracted:
            raise RuntimeError("Nothing to extract")

        results = []
        if len(extracted[0]) == 2:
            mean = numpy.array([row[0] for row in extracted])
            deviation = numpy.array([row[1] for row in extracted])
            results.append((mean - deviation).tolist())
            results.append(mean.tolist())
            results.append((mean + deviation).tolist())
        else:
            results.append(numpy.array(extracted).flatten().tolist())

        return results

    def read_group(data: pandas.DataFrame, index: int) -> List[List[float]]:
        index_1 = index * 2 + 1
        index_2 = index_1 + 1
        return read_column(data, index=index_1) + read_column(data, index=index_2)

    def derive_bounds(data: List[float], margin: int = 8) -> Tuple[int, int]:
        y = numpy.array(data)
        x = numpy.arange(len(y), dtype=float)

        if numpy.all(y == 1) or numpy.all(y == 0):
            raise RuntimeError("constant range")

        y = (y - y.min()) / (y.max() - y.min())
        x = (x - x.min()) / (x.max() - x.min())
        center = numpy.argmax(y * x)

        return int(center - margin), int(center + margin)

    def add_inset(ax, x, lines, x1, x2, bbox=None):
        x = numpy.asarray(x)
        lower = numpy.asarray(lines[1])
        upper = numpy.asarray(lines[3])

        mask = (x >= x1) & (x <= x2)
        y1 = numpy.min(lower[mask])
        y2 = numpy.max(upper[mask])

        # Use bbox if provided, else default relative coords
        if bbox is not None:
            axins = inset_axes(
                ax,
                width="42.5%", height="28.33%",
                loc="lower left",
                bbox_to_anchor=bbox,
                bbox_transform=ax.transAxes,
                borderpad=0
            )
        else:
            axins = ax.inset_axes([0.05, 0.65, 0.3, 0.3])  # default inset location

        # Plot inset data
        _, = axins.plot(x, lines[0])
        line_real, = axins.plot(x, lines[2])
        axins.fill_between(x, lower, upper, color=line_real.get_color(), alpha=0.2)

        # Set limits
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        # Remove ticks and tick labels
        axins.set_xticks([])
        axins.set_yticks([])
        axins.xaxis.set_ticklabels([])
        axins.yaxis.set_ticklabels([])

        for spine in axins.spines.values():
            spine.set_visible(True)

        if hasattr(ax, "indicate_inset_zoom"):
            ax.indicate_inset_zoom(axins, edgecolor="0.5")
        else:
            from mpl_toolkits.axes_grid1.inset_locator import mark_inset
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    x = read_column(data, index=0)[0]

    os.makedirs(folder, exist_ok=True)

    with open(folder + "table.txt", "w") as f:
        f.write(data.to_latex())

    for layer_index in range(layers):
        for group_index, group_label in enumerate(["forward", "backward", "functional"]):
            group_offset = 3 * layer_index + group_index

            lines = read_group(data, group_offset)

            fig, ax = plt.subplots()

            _, = ax.plot(x, lines[0], label="predicted")
            line_real, = ax.plot(x, lines[2], label="realized")
            ax.fill_between(x, lines[1], lines[3], color=line_real.get_color(), alpha=0.2)

            plt.title(f"layer {layer_index} - {group_label} connectivity")
            plt.xlabel("sparsity [%]")
            plt.ylabel("connectivity [%]")

            try:
                lhs, rhs = derive_bounds(lines[0], margin=6)
                add_inset(ax, x, lines, x[lhs], x[rhs], (0.05, 0.05, 1, 1))
                ax.legend(loc="lower left", bbox_to_anchor=(0.05, 0.38), bbox_transform=ax.transAxes)
            except Exception as e:
                ax.legend(loc="lower left", bbox_to_anchor=(0.05, 0.05), bbox_transform=ax.transAxes)
                print(e)

            ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

            plt.savefig(folder + f"/layer_{layer_index}_{group_label}")
            plt.close()

layers = [[3072, 1024], [1024, 512], [512, 512], [512, 100]]
round_to = 3
samples = 100
levels = 100

column_names = [[f"l{layer_index+1} predicted forward", f"l{layer_index+1} realized forward",
                 f"l{layer_index+1} predicted backward", f"l{layer_index+1} realized backward",
                 f"l{layer_index+1} predicted functional", f"l{layer_index+1} realized functional"] for layer_index in range(len(layers))]
column_names = numpy.array(column_names).flatten().tolist()

data = pandas.DataFrame(columns=column_names)
data.index.name = "sparsity"

for sparsity_level in tqdm(numpy.linspace(0.99, 0.9999, levels)):

    sparsities = None
    averages = [[] for _ in range(len(layers))]

    for _ in range(samples):
        network = src.network.custom.SparseANN(layers=layers, sparsity=[sparsity_level, sparsity_level, sparsity_level, sparsity_level], act="relu", layernorm=True, method="uniform")
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

generate_artifacts(data, layers=len(layers), folder="./data/theory_validation/uniform/")
