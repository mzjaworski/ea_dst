import graph_utilities.operations
import src.network.custom
import src.network.detail
import torch

from pathlib import Path
from typing import List
import argparse
import pandas
import json
import os

def get_files(filepaths: List[str]) -> List[str]:
    files = []

    for path in filepaths:
        abs_path = os.path.abspath(path)

        if os.path.isfile(abs_path):
            files.append(abs_path)
        elif os.path.isdir(abs_path):
            for root, _, dir_files in os.walk(abs_path, followlinks=False):
                for file in dir_files:
                    full_path = os.path.join(root, file)
                    files.append(full_path)
        else:
            raise RuntimeError(f"{abs_path} is neither a valid file or an existing directory")

    return files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='make_plots', description='script used to automate making plots')
    parser.add_argument('--groups', default='{"DST": [], "EA": []}', type=json.loads, help="groups of log files to use for plotting")
    parser.add_argument('--save_to', default=None, type=str, help="place to store tables")
    parser.add_argument("--caption", default=None, type=str, help="caption for table")

    # parse args
    args = parser.parse_args()

    rows = []

    for setup_name, group_files in args.groups.items():
        for file_group in group_files:
            files = get_files([file_group])
            sparsity = Path(file_group).parts[-1]

            for file in files:
                network = src.network.custom.SparseANN()
                network.load_state_dict(torch.load(file, map_location="cpu"))

                network.compute_reachabilities()
                total = graph_utilities.operations.edge_reachability(network)

                forward = sum(fwd for (fwd, _, _) in total) / network.active_weights() * 100.0
                backward = sum(bwd for (_, bwd, _) in total) / network.active_weights() * 100.0
                functional = sum(fn for (_, _, fn) in total) / network.active_weights() * 100.0

                rows.append({
                    "sparsity": sparsity,
                    "method": setup_name,
                    "forward": forward,
                    "backward": backward,
                    "functional": functional,
                })

    df = pandas.DataFrame(rows)

    data = (
        df.groupby(["sparsity", "method"])
        .agg({
            "forward": ["mean", "std"],
            "backward": ["mean", "std"],
            "functional": ["mean", "std"],
        })
        .reset_index()
    )

    data.columns = [
        "sparsity", "method",
        "fwd_mean", "fwd_std",
        "bwd_mean", "bwd_std",
        "func_mean", "func_std"
    ]

    data["forward [\\%]"] = data.apply(lambda r: f"${r['fwd_mean']:.2f} \\pm {r['fwd_std']:.2f}\\%$", axis=1)
    data["backward [\\%]"] = data.apply(lambda r: f"${r['bwd_mean']:.2f} \\pm {r['bwd_std']:.2f}\\%$", axis=1)
    data["functional [\\%]"] = data.apply(lambda r: f"${r['func_mean']:.2f} \\pm {r['func_std']:.2f}\\%$", axis=1)

    data = data[["sparsity", "method", "forward [\\%]", "backward [\\%]", "functional [\\%]"]]
    data = data.sort_values(["sparsity", "method"])
    data = data.set_index(["sparsity", "method"])

    latex = data.to_latex(
        escape=False,
        multirow=True,
        column_format="|l|l||c|c|c|"
    )

    if args.caption is not None:
        latex = f"\\begin{{table}}[ht]\n\\centering\n{latex}\\caption{{{args.caption}}}\n\\label{{tab:setup_results}}\n\\end{{table}}"

    if args.save_to is not None:
        directory = os.path.dirname(args.save_to)

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(args.save_to, "w") as f:
            f.write(latex)
    else:
        print(latex)