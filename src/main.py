import csv
from itertools import islice
import json
import os
import pickle
import h5py
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
import multiprocessing as mp

from fetchers.eth_fetchers import fetch_block, fetch_block_trace, fetcher_prestate, fetcher_call
from parsers import create_conflict_graph, get_callTracer_additional_metrics, parse_callTracer_trace, parse_preStateTracer_trace
from graph_metrics import *

from plotters import plot_data
import plotters
from savers import save_to_file

from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from loaders import load_compressed_file, load_file
from fetchers.sui_fetchers import fetcher_sui
from fetchers.fetchers import fetch_serial
from parsers import parse_sui_trace

def process_prestate_trace(block_number, diffFalse, diffTrue):
    print(f"processing {block_number}...")
    if diffFalse is None or diffTrue is None:
        print(f"{block_number} data is missing!")
        return None
    reads, writes = parse_preStateTracer_trace(diffFalse, diffTrue)
    txs = [tx_trace["txHash"] for tx_trace in diffFalse]
    G = create_conflict_graph(txs, reads, writes)
    return get_graph_metrics(G, {"block_number": block_number, "txs": len(diffFalse)})

def process_call_trace(block_number, call_trace):
    print(f"processing {block_number}...")
    if call_trace is None:
        print(f"{block_number} data is missing!")
        return None
    metrics = {}
    metrics.update(get_callTracer_additional_metrics(call_trace))
    reads, writes = parse_callTracer_trace(call_trace)
    txs = [tx_trace["txHash"] for tx_trace in call_trace]
    G = create_conflict_graph(txs, reads, writes)
    metrics.update(get_graph_metrics(G, {"block_number": block_number, "txs": len(call_trace)}))
    return metrics

def process_sui_trace(checkpoint_number, checkpoint, txs):
    print(f"processing {checkpoint_number}...")
    if checkpoint is None or txs is None:
        print(f"{checkpoint_number} data is missing!")
        return None
    reads, writes = parse_sui_trace(txs)
    txs_ids = [tx_trace["digest"] for tx_trace in txs]
    G = create_conflict_graph(txs_ids, reads, writes)
    metrics = get_graph_metrics(G, {"block_number": checkpoint_number, "txs": len(txs)})
    return metrics

def generate_data(data_path, output_path, processor, limit = None):
    write_header = not os.path.exists(output_path)
    max_pending = 1000
    with open(output_path, mode="a", newline="") as file:
        with ProcessPoolExecutor() as pool:
            data_generator = load_compressed_file(data_path, limit)
            futures = [
                pool.submit(processor, *data) for data in islice(data_generator, max_pending)
            ]
            all_submitted = len(futures) < max_pending
            writer = csv.writer(file)
            i = 0
            while len(futures) > 0:
                future = futures[0]
                futures = futures[1:]
                result = future.result()
                if result is not None:
                    if write_header:
                        write_header = False
                        writer.writerow(result.keys())
                    writer.writerow(result.values())
                    print(f"wrote result {i} to output csv")
                    i += 1
                if not all_submitted:
                    try:
                        next_data = next(data_generator)
                        futures.append(pool.submit(processor, *next_data))
                    except StopIteration:
                        all_submitted = True

def get_files(folder_path, extension):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)]

def main():
    dirpath = f"E:\\sui"
    output_path = "metrics.csv"
    '''if os.path.exists(output_path):
        os.remove(output_path)
    for file in get_files(dirpath, ".h5"):
        generate_data(file, output_path, process_sui_trace)'''
    plot_data(output_path)


def download_files(start: int, end: int, dirpath: str, filesize: int):
    for begin in range(start, end, filesize):
        end = begin + filesize
        filename = f"{begin}_{end}.h5"
        traces_generator = fetch_serial(range(begin, end), fetcher_sui)
        save_to_file(os.path.join(dirpath, filename), traces_generator)

if __name__ == "__main__":
    # download_files(150000000, 151000000, "E:\\sui", 100)
    main()  
    