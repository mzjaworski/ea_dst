from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse
import json
import sys
from typing import List
import settings
import re
import copy
import os
import time
import subprocess

class GlobalStorage(metaclass=settings.SingletonMeta): # not a good design choice but it will do
    def __init__(self):
        thread_id = None

thread_local = threading.local()
thread_id_lock = threading.Lock()

def replace_all(haystack: str, needle: str, replacement: str) -> str:
    return re.sub(needle, replacement, haystack)

def thread_id():
    if not hasattr(thread_local, 'thread_id'):
        with thread_id_lock:
            thread_local.thread_id = next(GlobalStorage().thread_id)
    return thread_local.thread_id

def run_experiment(command: str, devices: List[str], needle: str) -> None:
    id = thread_id() % len(devices)
    
    try:
        command = replace_all(command, needle, devices[id])
        print(f"running: {command}")
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    except Exception as e:
        print(f"command: {command}\n failed with error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='main', description='main program used to launch experiments')

    parser.add_argument('--configuration', default=None, type=json.loads, help="experiment template to run")
    parser.add_argument('--devices', default='["cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]', type=json.loads, help="devices on which we will run in parallel processing")
    parser.add_argument('--device_needle', default="DEVICE_LOCATION", type=str, help='string that will be replaced with passed device')
    parser.add_argument('--log_needle', default="LOG_IDENTIFIER", type=str, help='string that will be replaced with run ID')
    parser.add_argument('--runs', default=10, type=int, help='number of parallel experiments to run')
    parser.add_argument('--workers', default=4 * 7, type=int, help='number of parallel workers')

    # parse args
    args = parser.parse_args()

    # if no pipeline is passed through command line, scan stdin
    if args.configuration is None:
        configuration = []
        for line in sys.stdin:
            configuration.append(line.strip())
        args.configuration = configuration

    if not isinstance(args.configuration, list):
        raise RuntimeError("configuration type should be list!")

    # load specified files
    for file_index in range(len(args.configuration)):
        with open(args.configuration[file_index], "r") as file:
            args.configuration[file_index] = json.load(file)

    # generate file ids and names
    file_ids = [i for i in range(args.runs)]
    file_names = [f"{i}.log" for i in file_ids]

    # assemble object generating per thread ids
    GlobalStorage().thread_id = iter(range(args.workers))

    # generate log locations
    commands = []
    for configuration in args.configuration:
        for run_index in range(args.runs):
            command = copy.deepcopy(configuration)
            directory, filename = os.path.split(command['log'])
            command['log'] = os.path.join(directory, replace_all(filename, args.log_needle, file_names[run_index])) # a bit excessive
            
            if "serialize" in command:
                if "kwargs" in command['serialize']:
                    if "path" in command['serialize']["kwargs"]:
                        command['serialize']["kwargs"]["path"] = replace_all(command['serialize']["kwargs"]["path"], args.log_needle, file_names[run_index])
            
            command = f"echo '{json.dumps(command)}' | python ./main.py" # not platform-agnostic
            commands.append(command)

    # run commands in thread pool
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []

        for command in commands:
            futures.append(executor.submit(run_experiment, command, args.devices, args.device_needle))

        for future in as_completed(futures):
            error = future.result()

    print("done")