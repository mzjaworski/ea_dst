import pandas
import numpy

from typing import List
import argparse
import json
import re
import os

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

    # parse args
    args = parser.parse_args()

    metrics = ["Top-1 Accuracy", "Top-3 Accuracy", "Top-5 Accuracy", "TFLOP", "train loss", "Active Weights", "Total Weights"]
    needles = ["acc-top_1", "acc-top_3", "acc-top_5", "amortized_flops", "mean_loss", "active_weights", "total_weights"]

    data = pandas.DataFrame(columns=metrics)
    data.index.name = "setup"

    for group_name, group_files in args.groups.items():

        if not len(group_files):
            raise RuntimeError("nothing to plot!")

        # in case user passed directories retrieve all files inside, otherwise noop
        group_files = get_files(group_files)

        row_data = []
        for (needle, metric) in zip(needles, metrics):
            matches = [extract_needles_as_float(path, needle) for path in group_files]

            if needle in ["acc-top_1", "acc-top_3", "acc-top_5"]:
                last_values = numpy.array(matches)[:, -1]
                mean = numpy.mean(last_values)
                std = numpy.std(last_values)
                row_data.append(f"{round(mean, 2):.2f} ± {round(std, 2):.2f}%")
            elif needle in ["amortized_flops"]:
                last_values = numpy.array(matches)[:, -1]
                mean = numpy.mean(last_values)
                row_data.append(f"{round(mean / 1_000_000_000_000, 2):.2f}") # decimal
            elif needle in ["active_weights", "total_weights"]:
                last_values = numpy.array(matches)[:, -1]
                mean = numpy.mean(last_values)
                row_data.append(f"{int(mean)}")
            elif needle in ["mean_loss"]:
                last_values = numpy.array(matches)[:, -1]
                mean = numpy.mean(last_values)
                std = numpy.std(last_values)
                row_data.append(f"{round(mean, 2):.2f} ± {round(std, 2):.2f}")
            else:
                row_data.append("-")

        data.loc[group_name] = row_data

    # print(data)
    print(data.to_latex())