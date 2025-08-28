import matplotlib.pyplot as plt
from typing import List
import argparse
import json
import re
import os

def extract_needles(filepath: str, needle: str) -> List[str]:
    search_pattern = re.compile(rf'\[{re.escape(needle)}::([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\]')
    matches = []

    with open(filepath, 'r') as file:
        for line in file:
            match = search_pattern.search(line)
            if match:
                matches.append(match.group(1))

    return matches

def extract_needles_as_float(filepath: str, needle: str) -> List[float]:
    return [float(entry) for entry in extract_needles(filepath, needle)]

def average_extracted_needles(matches: List[List[float]]) -> List[float]:
    # identify file with max elements
    maximum_length = max(len(matches) for matches in matches)
    longest_list = next(matches for matches in matches if len(matches) == maximum_length)

    # pad using its values shorter lists
    for i in range(len(matches)):
        current = matches[i]
        if len(current) < maximum_length:
            padding = longest_list[len(current):]
            matches[i] = current + padding

    # now average per index
    return [sum(matches[i] for matches in matches) / len(matches) for i in range(maximum_length)]

def bernoulli_extracted_needles_sma(matches: List[List[float]], class_value: float = 1.0, window_size: int = 17) -> List[float]:

    # identify file with max elements
    maximum_length = max(len(m) for m in matches)
    longest_list = next(m for m in matches if len(m) == maximum_length)

    # pad shorter lists using values from the longest list
    padded_matches = []
    for m in matches:
        if len(m) < maximum_length:
            m = m + longest_list[len(m):]
        padded_matches.append(m)

    # compute Bernoulli probabilities per index
    probabilities = []
    for i in range(maximum_length):
        count_class = sum(1 for m in padded_matches if m[i] == class_value)
        probabilities.append(count_class / len(padded_matches))

    # apply simple moving average (last `window_size` entries)
    smoothed = []
    for i in range(maximum_length):
        start = max(0, i - window_size + 1)
        window = probabilities[start:i+1]
        smoothed.append(sum(window) / len(window))

    return smoothed

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
    parser.add_argument('--needles', default='["loss"]', type=json.loads, help="fields to extract from logs")
    parser.add_argument('--save_to', default=None, type=str, help="optional place to save current plot")

    # parse args
    args = parser.parse_args()

    if not len(args.needles):
        raise RuntimeError("nothing to plot!")

    if len(args.needles) > 2:
        raise RuntimeError("currently at most two needles are supported!")

    plt.figure(figsize=(10, 5))

    xlabel = args.needles[1] if len(args.needles) > 1 else "epochs"
    ylabel = args.needles[0]

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if args.title is None:
        args.title = f"{xlabel} - {ylabel}"

    plt.title(args.title)

    for group_name, group_files in args.groups.items():

        if not len(group_files):
            raise RuntimeError("nothing to plot!")

        # in case user passed directories retrieve all files inside, otherwise noop
        group_files = get_files(group_files)
        data = []

        for needle in args.needles:
            matches = [extract_needles_as_float(path, needle) for path in group_files]

            if not len(matches):
                raise RuntimeError(f"couldn't match any '{needle}' in the specified directories {group_files}")

            if needle in ["survivor", "indicator"]:
                data.append(bernoulli_extracted_needles_sma(matches))
            else:
                data.append(average_extracted_needles(matches))

        if len(data) == 1:
            data.append(list(range(len(data[0]))))

        plt.plot(data[1], data[0], label=group_name, linewidth=2)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if args.save_to is not None:
        plt.savefig(args.save_to)
        plt.close()
    else:
        plt.show()
