import matplotlib.pyplot as plt
from typing import List
import argparse
import json
import re
import os

def extract_needles(filepath: str, needle: str) -> List[str]:
    search_pattern = re.compile(rf'\[{re.escape(needle)}::([0-9.]+)\]')
    matches = []

    with open(filepath, 'r') as file:
        for line in file:
            match = search_pattern.search(line)
            if match:
                matches.append(match.group(1))

    return matches

def extract_needles_as_float(filepath: str, needle: str) -> List[float]:
    return [float(entry) for entry in extract_needles(filepath, needle)]

def average_extracted_needles(filepaths: List[str], needle: str) -> List[float]:
    all_matches = [extract_needles_as_float(path, needle) for path in filepaths]

    if not len(all_matches):
        raise RuntimeError(f"couldn't match any '{needle}' in the specified directories {filepaths}")

    # identify file with max elements
    maximum_length = max(len(matches) for matches in all_matches)
    longest_list = next(matches for matches in all_matches if len(matches) == maximum_length)

    # pad using its values shorter lists
    for i in range(len(all_matches)):
        current = all_matches[i]
        if len(current) < maximum_length:
            padding = longest_list[len(current):]
            all_matches[i] = current + padding

    # now average per index
    return [sum(matches[i] for matches in all_matches) / len(all_matches) for i in range(maximum_length)]

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
    parser.add_argument('--needles', default='["loss"]', type=json.loads, help="fields to extract from logs")

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

    plt.title(f"{xlabel} - {ylabel}")

    for group_name, group_files in args.groups.items():

        if not len(group_files):
            raise RuntimeError("nothing to plot!")

        # in case user passed directories retrieve all files inside, otherwise noop
        group_files = get_files(group_files)
        data = []

        for needle in args.needles:
            data.append(average_extracted_needles(group_files, needle))

        if len(data) == 1:
            data.append(list(range(len(data[0]))))

        plt.plot(data[1], data[0], label=group_name, linewidth=2)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()
