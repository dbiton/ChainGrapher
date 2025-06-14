import csv
from itertools import islice
import os
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor

load_dotenv()

from interfaces.iota_interface import IotaInterface
from interfaces.eth_call_interface import EthCallInterface
from interfaces.eth_prestate_interface import EthPerstateInterface
from interfaces.sui_interface import SuiInterface
from graph_metrics import get_graph_metrics
from plotters import plot_data, plot_graph
from savers import save_to_file
from loaders import load_compressed_file
from fetchers import fetch_parallel, fetch_serial

sui_interface = SuiInterface()
iota_interface = IotaInterface()
eth_interface = EthPerstateInterface()
crypto_interface = eth_interface

def process_trace(block_number, *trace_args):
    print(f"Processing {block_number}...")
    metrics = crypto_interface.get_additional_metrics(block_number, trace_args)
    G = crypto_interface.get_conflict_graph(trace_args)
    metrics.update(get_graph_metrics(G))
    return metrics


def agg_load_compressed_file(data_path, limit, k):
    it = load_compressed_file(data_path, limit)
    while True:
        chunk = list(next(it, None) for _ in range(k))
        chunk = [x for x in chunk if x is not None]
        if not chunk:
            break
        agg_txs = sum([txs for (_, _, txs) in chunk], [])
        yield [chunk[0][0], chunk[0][1], agg_txs]


def generate_data(data_path, output_path, limit=None):
    write_header = not os.path.exists(output_path)
    max_pending = 1000
    with open(output_path, mode="a", newline="") as file:
        with ProcessPoolExecutor() as pool:
            data_generator = agg_load_compressed_file(data_path, limit, 10)
            futures = [
                pool.submit(process_trace, *data) for data in islice(data_generator, max_pending)
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
                        futures.append(pool.submit(process_trace, *next_data))
                    except StopIteration:
                        all_submitted = True


def get_files(folder_path, extension):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)]


def main():
    output_path = "metrics.csv"
    '''dirpath = "E:\\sui"
    if os.path.exists(output_path):
        os.remove(output_path)
    for file in get_files(dirpath, ".h5"):
        generate_data(file, output_path)'''
    plot_data(output_path)
    

def download_files(start: int, end: int, dirpath: str, filesize: int):
    for begin in reversed(list(range(start, end, filesize))):
        end = begin + filesize
        filename = f"{begin}_{end}.h5"
        fetcher_multiple = fetch_serial
        if crypto_interface.fetch_parallel:
            fetcher_multiple = fetch_parallel
        traces_generator = fetcher_multiple(range(begin, end), crypto_interface.fetch)
        save_to_file(os.path.join(dirpath, filename), traces_generator)


if __name__ == "__main__":
    download_files(22039000, 22040000, ".", 1000)
    # main()
