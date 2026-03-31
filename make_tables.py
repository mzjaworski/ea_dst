import pandas
import numpy

from typing import List
import argparse
import json
import re
import os
from pathlib import Path


def float_pattern(filepath: str, needle: str) -> List[str]:
    search_pattern = re.compile(rf'\[{re.escape(needle)}::([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\]')
    matches = []

    with open(filepath, 'r') as file:
        for line in file:
            match = search_pattern.search(line)
            if match:
                matches.append(match.group(1))

    return matches

def extract_needles_as_float(filepath: str, needle: str) -> List[float]:
    return [float(entry) for entry in float_pattern(filepath, needle)]

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

    parser.add_argument('--title', default=None, type=str, help="optional plot title")
    parser.add_argument('--groups', default='{"DST": [], "EA": []}', type=json.loads, help="groups of log files to use for plotting")
    parser.add_argument('--save_to', default=None, type=str, help="place to store tables")

    # parse args
    args = parser.parse_args()

    metrics = ["Accuracy", "Loss", "TFLOPs", "Active Weights", "Total Weights"]
    needles = ["acc-top_1", "mean_loss", "amortized_flops", "active_weights", "total_weights"]

    setup_order = list(args.groups.keys())

    # store results as dict[(sparsity, setup)] -> {metric: value}
    rows = {}

    for setup_name, group_files in args.groups.items():
        if not group_files:
            raise RuntimeError("nothing to plot!")

        for file_group in group_files:
            files = get_files([file_group])
            sparsity = Path(file_group).parts[-1]  # top-level row

            # initialize the row key
            row_key = (sparsity, setup_name)
            row_data = {}

            for needle, metric in zip(needles, metrics):
                matches = [extract_needles_as_float(path, needle) for path in files]
                matches = [m for m in matches if m is not None and len(m) > 0]

                if not matches:
                    value = "-"
                else:
                    last_values = numpy.array([m[-1] for m in matches])
                    if needle == "acc-top_1":
                        mean = numpy.mean(last_values)
                        std = numpy.std(last_values)
                        value = f"{mean:.2f} ± {std:.2f}\\%"
                    elif needle == "amortized_flops":
                        mean = numpy.mean(last_values)
                        value = f"{mean / 1e12:.2f}"
                    elif needle in ["active_weights", "total_weights"]:
                        mean = numpy.mean(last_values)
                        value = f"{int(mean)}"
                    elif needle == "mean_loss":
                        mean = numpy.mean(last_values)
                        std = numpy.std(last_values)
                        value = f"{mean:.2f} ± {std:.2f}"
                    else:
                        value = "-"

                row_data[metric] = value

            rows[row_key] = row_data

    index = pandas.MultiIndex.from_tuples(rows.keys(), names=["Sparsity", "Setup"])
    data = pandas.DataFrame(list(rows.values()), index=index, columns=metrics)

    data = data.copy()
    data.index = pandas.MultiIndex.from_arrays([
        data.index.get_level_values("Sparsity").astype(int),
        pandas.Categorical(data.index.get_level_values("Setup"), categories=setup_order, ordered=True)
    ], names=["Sparsity", "Setup"])

    data = data.sort_index(level=["Sparsity", "Setup"])
    latex = data.to_latex(column_format=("l|" * 2 + "c|" * len(needles))[:-1])

    caption = "Results of different setups at various sparsities"
    latex = f"\\begin{{table}}[ht]\n\\centering\n{latex}\\caption{{{caption}}}\n\\label{{tab:setup_results}}\n\\end{{table}}"

    if args.save_to is not None:
        directory = os.path.dirname(args.save_to)

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(args.save_to, "w") as f:
            f.write(latex)
    else:
        print(latex)