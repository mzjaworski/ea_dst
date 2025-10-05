import matplotlib.pyplot as plt
from typing import List, Tuple
import scipy.stats
import argparse
import numpy
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

def average_with_confidence(series_list: List[List[float]], confidence: float = 0.95) -> Tuple[List[float], List[float], List[float]]:
    # identify file with max elements
    longest_length = max(len(s) for s in series_list)
    longest_series = max(series_list, key=len)
    
    # pad using its values shorter lists
    padded_series = []
    for s in series_list:
        if len(s) < longest_length:
            # pad with the values from the remainder of the longest series
            extra_values = longest_series[len(s):]
            padded_series.append(s + extra_values)
        else:
            padded_series.append(s)

    # convert to numpy and average
    data = numpy.array(padded_series, dtype=float)
    mean_values = data.mean(axis=0)

    # compute CI using students-t
    n = data.shape[0]
    alpha = 1.0 - confidence
    t_crit = scipy.stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    standard_error = scipy.stats.sem(data, axis=0, nan_policy="omit")
    margin = t_crit * standard_error

    lower_bound = mean_values - margin
    upper_bound = mean_values + margin

    return mean_values.tolist(), lower_bound.tolist(), upper_bound.tolist()

def simple_moving_average(values: List[float], window_size: int = 17) -> List[float]:
    """
    Apply simple moving average smoothing over a list of values.
    """
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        window = values[start:i+1]
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
    parser.add_argument('--ci', default=True, type=bool, action=argparse.BooleanOptionalAction, help='plot ci or not')
    parser.add_argument('--dpi', default=300, type=int, help='dots per inch (dpi) of the image')

    # parse args
    args = parser.parse_args()

    if not len(args.needles):
        raise RuntimeError("nothing to plot!")

    if len(args.needles) > 2:
        raise RuntimeError("currently at most two needles are supported!")

    plt.figure(figsize=(10, 5), dpi=args.dpi)

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
        
        means = []
        lower_bounds = []
        upper_bounds = []
        
        for needle in args.needles:
            matches = [extract_needles_as_float(path, needle) for path in group_files]
            
            if not len(matches):
                raise RuntimeError(f"couldn't match any '{needle}' in the specified directories {group_files}")
            
            mean, lower_bound, upper_bound = average_with_confidence(matches, confidence=0.95)
            
            means.append(mean)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            
        if len(means) == 1:
            means.append(list(range(len(means[0]))))
        
        axis = plt.gca()
        axis.plot(means[1], means[0], label=group_name, linewidth=2)
        
        if args.ci and len(lower_bounds) and len(upper_bounds):
            axis.fill_between(means[1], lower_bounds[0], upper_bounds[0], color=axis.lines[-1].get_color(), alpha=0.35)
        

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if args.save_to is not None:

        directory = os.path.dirname(args.save_to)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        plt.savefig(args.save_to, dpi=args.dpi)
        plt.close()
    else:
        plt.show()
