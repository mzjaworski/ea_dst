import argparse
import json
import sys
import utilities

# small utility script used to download all datasets
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='prepare_datasets', description='helper program used to download dataset files')
    parser.add_argument('--datasets', default=None, type=json.loads, help="list of dataset objects to prepare, ie. initialize as it may require downloading them")

    # parse args
    args = parser.parse_args()

    # if no pipeline is passed through command line, scan stdin
    if args.datasets is None:
        datasets = ""
        for line in sys.stdin:
            datasets += line
        args.datasets = json.loads(datasets)

    initialized = []
    for config in args.datasets:
        initialized.append(utilities.resolve_class(config["type"])(**config["kwargs"]).load())

    print("done! ")
