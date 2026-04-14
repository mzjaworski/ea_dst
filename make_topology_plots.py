from pathlib import Path
from typing import List
import argparse
import json
import os

import graph_utilities.operations
import src.network.custom
import src.network.detail
import torch


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

    return sorted(files)

def determine_input_size(main: str) -> int:
    if "cifar100" in main:
        return 3072
    elif "cifar10" in main:
        return 3072
    elif "tiny_imagenet" in main:
        return 12288
    else:
        raise RuntimeError("cannot determine input size!")

def determine_output_size(main: str) -> int:
    if "cifar100" in main:
        return 100
    elif "cifar10" in main:
        return 10
    elif "tiny_imagenet" in main:
        return 200
    else:
        raise RuntimeError("cannot determine output size!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='make_plots', description='script used to automate making plots')

    parser.add_argument('--model_path', default=None, type=str, help="groups of log files to use for plotting")
    parser.add_argument('--save_to', default=None, type=str, help="optional place to save current plot")
    parser.add_argument('--dpi', default=100, type=int, help='dots per inch (dpi) of the image')

    # parse args
    args = parser.parse_args()

    if args.model_path is None:
        raise RuntimeError("nothing to plot!")
    if args.save_to is None:
        raise RuntimeError("no path to save!")

    if not Path(args.model_path).exists():
        print("model_path does not point to an existing file!")

    network = src.network.custom.SparseANN(layers=[(determine_input_size(args.model_path), 1024), (1024, 512), (512, 512), (512, determine_output_size(args.model_path))])
    network.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    network.compute_reachabilities()

    graph_utilities.operations.layered_graph_figure(network, filename=args.save_to)
