from concurrent.futures import ThreadPoolExecutor
import threading
import argparse
import json
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='main', description='main program used to launch experiments')

    parser.add_argument('--configuration', default=None, type=json.loads, help="experiment template to run")
    parser.add_argument('--devices', default='["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4"]', type=json.loads, help="devices on which we will run in parallel processing")
    parser.add_argument('--device_needle', default="DEVICE_LOCATION", type=str, help='string that will be replaced with passed device')
    parser.add_argument('--log_needle', default="LOG_IDENTIFIER", type=str, help='string that will be replaced with run ID')
    parser.add_argument('--runs', default=5, type=int, help='number of parallel experiments to run')

    # parse args
    args = parser.parse_args()

    # if no pipeline is passed through command line, scan stdin
    if args.configuration is None:
        configuration = ""
        for line in sys.stdin:
            configuration += line
        args.configuration = json.loads(configuration)

    # get number of runs

    # replace
    print(args.configuration)
