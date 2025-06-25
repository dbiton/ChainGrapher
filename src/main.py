import csv
from itertools import islice
import os
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
import itertools
from concurrent.futures._base import as_completed

load_dotenv()

from interfaces.iota_interface import IotaInterface
from interfaces.eth_call_interface import EthCallInterface
from interfaces.eth_prestate_interface import EthPerstateInterface
from interfaces.sui_interface import SuiInterface, USER_KINDS
from graph_metrics import get_graph_metrics
from plotters import plot_data, plot_graph
from savers import save_to_file
from loaders import load_compressed_file
from fetchers import fetch_parallel, fetch_serial

sui_interface = SuiInterface()
iota_interface = IotaInterface()
eth_interface = EthPerstateInterface()
crypto_interface = sui_interface

def process_trace(block_number, *trace_args):
    print(f"Processing {block_number}...")
    print(f"Getting additional metrics {block_number}...")
    trace_args = list(trace_args)
    # trace_args[1] = [tx for tx in trace_args[1] if crypto_interface._get_tx_type(tx) in USER_KINDS]
    metrics = crypto_interface.get_additional_metrics(block_number, trace_args)
    print(f"Creating conflict graph {block_number}...")
    G = crypto_interface.get_conflict_graph(trace_args)
    print(f"Getting graph metrics {block_number}...")
    metrics.update(get_graph_metrics(G))
    return metrics


def agg_load_compressed_file(dirpath, limit, k):
    generators = [load_compressed_file(filepath) for filepath in get_files(dirpath, ".h5")]
    it = itertools.chain.from_iterable(generators)
    while True:
        chunk = list(next(it, None) for _ in range(k))
        chunk = [x for x in chunk if x is not None]
        if not chunk:
            break
        agg_txs = sum([txs for (_, _, txs) in chunk], [])
        yield [chunk[0][0], chunk[0][1], agg_txs]


def generate_data(dirpath, output_path, limit=None):
    data_generator = agg_load_compressed_file(dirpath, limit, 1)
    write_header = not os.path.exists(output_path)
    max_pending = 6

    with open(output_path, mode="a", newline="") as file:
        with ProcessPoolExecutor() as pool:
            futures = {pool.submit(process_trace, *data): data for data in islice(data_generator, max_pending)}
            all_submitted = len(futures) < max_pending
            writer = csv.writer(file)
            i = 0
            while futures:
                for future in as_completed(futures):
                    result = future.result()
                    del futures[future]
                    if result is not None:
                        sorted_keys = sorted(result.keys())
                        sorted_values = [result[k] for k in sorted_keys]
                        if write_header:
                            write_header = False
                            writer.writerow(sorted_keys)
                        writer.writerow(sorted_values)
                        print(result)
                        print(f"wrote result {i} to output csv")
                        i += 1
                    if not all_submitted:
                        try:
                            next_data = next(data_generator)
                            new_future = pool.submit(process_trace, *next_data)
                            futures[new_future] = next_data
                        except StopIteration:
                            all_submitted = True
                    break  # Exit early to allow re-entering as_completed with updated futures


def get_files(folder_path, extension):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)]


def main():
    output_path = "metrics_sui_big.csv"
    dirpath = "C:\\Projects\\SuiGrapher"
    if os.path.exists(output_path):
        os.remove(output_path)
    generate_data(dirpath, output_path)
    plot_data(output_path, crypto_interface)
    

def download_files(start: int, end: int, dirpath: str, filesize: int):
    for begin in list(range(start, end, filesize)):
        end = begin + filesize
        filename = f"{begin}_{end}.h5"
        fetcher_multiple = fetch_serial
        if crypto_interface.fetch_parallel:
            fetcher_multiple = fetch_parallel
        traces_generator = fetcher_multiple(range(begin, end), crypto_interface.fetch)
        save_to_file(os.path.join(dirpath, filename), traces_generator)


if __name__ == "__main__":
    download_files(0, 150000000, "F:\\sui_checkpoints", 1000000)
    # main()
