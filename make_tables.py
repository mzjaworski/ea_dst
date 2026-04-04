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

def format_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f} K"
    else:
        return str(n)

def format_flops_scientific(flops: float) -> str:
    if flops == 0:
        return "0"
    exponent = int(f"{flops:e}".split('e')[1])
    mantissa = flops / (10 ** exponent)
    return f"{mantissa:.2f} * 10^" + "{" + str(exponent) + "}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='make_plots', description='script used to automate making plots')

    parser.add_argument('--title', default=None, type=str, help="optional plot title")
    parser.add_argument('--groups', default='{"DST": [], "EA": []}', type=json.loads, help="groups of log files to use for plotting")
    parser.add_argument('--save_to', default=None, type=str, help="place to store tables")
    parser.add_argument("--caption", default=None, type=str, help="caption for table")
    parser.add_argument('--deltas', default=True, type=bool, action=argparse.BooleanOptionalAction, help='add deltas or not')

    # parse args
    args = parser.parse_args()

    metrics = ["Accuracy", "Loss", "Training FLOPs", "Parameters (Active)", "Parameters (Total)"]
    needles = ["acc-top_1", "mean_loss", "amortized_flops", "active_weights", "total_weights"]

    setup_order = list(args.groups.keys())

    rows = {}
    dst_means = {}

    for setup_name, group_files in args.groups.items():
        if not group_files:
            raise RuntimeError("nothing to plot!")

        for file_group in group_files:
            files = get_files([file_group])
            sparsity = Path(file_group).parts[-1]

            row_key = (sparsity, setup_name)
            row_data = {}

            last_acc_values = []
            last_loss_values = []

            for needle, metric in zip(needles, metrics):
                matches = [extract_needles_as_float(path, needle) for path in files]
                matches = [m for m in matches if m is not None and len(m) > 0]

                if not matches:
                    value = "-"
                else:
                    last_values = numpy.array([m[-1] for m in matches])

                    if metric == "Accuracy":
                        last_acc_values = last_values
                    elif metric == "Loss":
                        last_loss_values = last_values
                    if needle == "acc-top_1":
                        mean = numpy.mean(last_values)
                        std = numpy.std(last_values)
                        value = f"${mean:.2f} \\pm {std:.2f}\\%$"
                    elif needle == "amortized_flops":
                        mean = numpy.mean(last_values)
                        value = f"${format_flops_scientific(mean)}$"
                    elif needle in ["active_weights", "total_weights"]:
                        mean = numpy.mean(last_values)
                        value = f"${format_number(mean)}$"
                    elif needle == "mean_loss":
                        mean = numpy.mean(last_values)
                        std = numpy.std(last_values)
                        value = f"${mean:.2f} \\pm {std:.2f}$"
                    else:
                        value = "-"

                row_data[metric] = value

                if args.deltas:
                    if metric in ["Accuracy", "Loss"]:
                        delta_val = "-"

                        if setup_name == "DST":
                            if metric == "Accuracy":
                                dst_means.setdefault(sparsity, {})["Accuracy"] = numpy.mean(last_acc_values)
                            else:
                                dst_means.setdefault(sparsity, {})["Loss"] = numpy.mean(last_loss_values)
                        else:
                            delta_values = 100.0 * (last_values - dst_means[sparsity][metric]) / dst_means[sparsity][metric]
                            delta_val = f"${numpy.mean(delta_values):.2f} \\pm {numpy.std(delta_values):.2f}\\%$"

                        row_data[f"{metric} $\\Delta$ [\\%]"] = delta_val
            rows[row_key] = row_data

    ordered_columns = []

    for m in metrics:
        ordered_columns.append(m)
        if args.deltas:
            if m in ["Accuracy", "Loss"]:
                ordered_columns.append(f"{m} $\\Delta$ [\\%]")

    data = pandas.DataFrame(list(rows.values()), index=pandas.MultiIndex.from_tuples(rows.keys(), names=["Sparsity [\\%]", "Method"]))
    data = data[ordered_columns]

    data.index = pandas.MultiIndex.from_arrays([
        data.index.get_level_values("Sparsity [\\%]"),
        pandas.Categorical(data.index.get_level_values("Method"), categories=setup_order, ordered=True)
    ], names=["Sparsity [\\%]", "Method"])
    data = data.sort_index(level=["Sparsity [\\%]", "Method"])

    latex = data.to_latex(column_format=("l|" * 2 + "c|" * len(ordered_columns))[:-1])

    if args.caption is not None:
        latex = f"\\begin{{table}}[ht]\n\\centering\n{latex}\\caption{{{args.caption}}}\n\\label{{tab:setup_results}}\n\\end{{table}}"

    if args.save_to is not None:
        directory = os.path.dirname(args.save_to)

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(args.save_to, "w") as f:
            f.write(latex)

        data.to_csv(args.save_to + ".csv")
    else:
        print(latex)