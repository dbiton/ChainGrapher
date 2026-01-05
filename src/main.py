import csv
import glob
import logging
import re
import sys
from itertools import islice
import os

import infixpy
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
import itertools
from concurrent.futures._base import as_completed
from infixpy import *

load_dotenv(override=False) # Prioritize environment first

import interfaces.solana_interface
from interfaces.iota_interface import IotaInterface
from interfaces.solana_interface import SolanaInterface
from interfaces.eth_call_interface import EthCallInterface
from interfaces.eth_prestate_interface import EthPerstateInterface
from interfaces.sui_interface import SuiInterface, USER_KINDS
from graph_metrics import get_graph_metrics
from plotters import plot_data, plot_graph, plot_data_dir, plot_data_filelist
from savers import save_to_file, CHUNK_SIZE
from loaders import load_compressed_file
from fetchers import fetch_parallel, fetch_serial

sui_interface = SuiInterface()
iota_interface = IotaInterface()
eth_interface = EthPerstateInterface()
solana_interface = SolanaInterface()
crypto_interface = solana_interface


main_func_registry={}

MAX_BLOCK_EXCLUDE,MIN_BLOCK_EXCLUDE=None,None
IGNORE_LIST=[]

def register_entrypoint(func, name):
    global main_func_registry
    key = func.__name__ if name is None else name
    main_func_registry[key] = func
    return func
def entrypoint(_func=None, name=None):
    if _func is not None and callable(_func):
        return register_entrypoint(_func,name)
    else:
        return lambda _func: register_entrypoint(_func, name)

def do_main_func(key):
    global main_func_registry
    main_func_registry[key]()

def process_trace(block_number, *trace_args):
    logging.info(f"Processing {block_number}...")
    logging.info(f"Getting additional metrics {block_number}...")
    trace_args = list(trace_args)
    # trace_args[1] = [tx for tx in trace_args[1] if crypto_interface._get_tx_type(tx) in USER_KINDS]
    metrics = crypto_interface.get_additional_metrics(block_number, trace_args)
    logging.info(f"Creating conflict graph {block_number}...")
    G = crypto_interface.get_conflict_graph(trace_args)
    logging.info(f"Getting graph metrics {block_number}...")
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
        #agg_txs = sum([txs for (_, _, txs) in chunk], [])
        #yield [chunk[0][0], chunk[0][1], agg_txs]
        yield from chunk


def generate_data(data_path, output_path):
    # data_generator = agg_load_compressed_file(dirpath, limit,1)
    data_generator = load_compressed_file(data_path)
    write_header = not os.path.exists(output_path)
    max_pending = 6

    with open(output_path, mode="w", newline="") as file:
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
                        file.flush()
                        # logging.info()
                        logging.info(f"wrote result {i} to output csv: {result}")
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

@entrypoint(name="metrics")
def do_metrics():
    # output_path = "metrics_sui_big.csv"
    dirpath = "./data/download/solana/"
    # if os.path.exists(output_path):
    #     os.remove(output_path)
    for  datapath, range in (
            Seq(os.listdir(f"{dirpath}/chunks"))
                .map(lambda x: re.match(r"((\d+)_(\d+)).h5", x))
                .filter(lambda x: x is not None)
                .filter(lambda x: (MIN_BLOCK_EXCLUDE is None) or (int(x.group(2)) >= MIN_BLOCK_EXCLUDE))
                .filter(lambda x: (MAX_BLOCK_EXCLUDE is None) or (int(x.group(3)) <= MAX_BLOCK_EXCLUDE))
                    .filter(lambda x: (x.group(2) not in IGNORE_LIST) and (x.group(3) not in IGNORE_LIST))
                .filter(lambda x: not os.path.exists(f"{dirpath}/metrics/{x.group(1)}.csv"))
                .map(lambda x:( f"{dirpath}/chunks/{x.group(0)}", x.group(1)))
                .sortby(lambda x:x[1])):
        generate_data(datapath, f"{dirpath}/metrics/temp_{range}.csv")
        os.rename(f"{dirpath}/metrics/temp_{range}.csv",f"{dirpath}/metrics/{range}.csv")

@entrypoint
def do_plots():
    all_files = (Seq(glob.glob(os.path.join("./data/download/solana/metrics", "*.csv")))
                 .map(lambda x: (x,re.match(r".+\\(\d+)_(\d+).csv", x)))
                 .filter(lambda x: x[1] is not None)
                 .filter(lambda x: (MIN_BLOCK_EXCLUDE is None) or (int(x[1].group(2)) >= MIN_BLOCK_EXCLUDE))
                 .filter(lambda x: (MAX_BLOCK_EXCLUDE is None) or (int(x[1].group(3)) <= MAX_BLOCK_EXCLUDE))
                 .filter(lambda x: (x.group(2) not in IGNORE_LIST) and (x.group(3) not in IGNORE_LIST))
                 .map(lambda x: x[0])
                 .sort()
                 .tolist()
                 )

    plot_data_filelist(all_files, crypto_interface)


def download_files(start: int, end: int, dirpath: str, filesize: int):
    assert (filesize % CHUNK_SIZE == 0)
    count = end - start
    assert (count % filesize == 0)
    for begin in list(range(start, end, filesize)):
        end = begin + filesize
        filename = f"{begin}_{end-1}.h5"
        fetcher_multiple = fetch_serial
        if crypto_interface.fetch_parallel:
            fetcher_multiple = fetch_parallel
        #run once to save files locally before making compressed file
        for _ in fetcher_multiple(range(begin, end), crypto_interface.fetch):
            pass

        # run again to load local files to make compressed file
        traces_generator = fetch_serial(range(begin, end), crypto_interface.fetch)
        save_to_file(os.path.join(dirpath ,filename), traces_generator)
        # remove allfiles from cache
        crypto_interface.remove_cached_files(range(begin, end))

@entrypoint
def do_download():
    logging.info("Starting download")
    # start_block = 390_000_000
    start_block = int(os.getenv("START_BLOCK"))
    # start_block = 385_280_000
    # start_block = 386_280_000
    # start_block = 388_420_000
    # count = 100_000
    count = int(os.getenv("BLOCK_COUNT"))
    dirpath="./data/download/solana/"
    interfaces.solana_interface.DIR_PATH = f"{dirpath}/inprog"
    current_start = (
        Seq(os.listdir(f"{dirpath}/chunks/"))
        .map(lambda x: re.match(r"\d+_(\d+).h5", x))
        .filter(lambda x: x is not None)
        .filter(lambda x: (MIN_BLOCK_EXCLUDE is None) or (int(x.group(2)) >= MIN_BLOCK_EXCLUDE))
        .filter(lambda x: (MAX_BLOCK_EXCLUDE is None) or (int(x.group(3)) <= MAX_BLOCK_EXCLUDE))
        .filter(lambda x: (x.group(2) not in IGNORE_LIST) and (x.group(3) not in IGNORE_LIST))
        .map(lambda x: int(x.group(1)))
        .chain([start_block])
        .reduce(max))
    download_files(start=current_start, end=start_block + count, dirpath=f"{dirpath}/chunks",
                   filesize=1_000)


if __name__ == "__main__":
    # logging.basicConfig(format='%(message)s', level=logging.BASIC_FORMAT)
    logging.basicConfig(level=logging.INFO)
    # do_download()
    # main()
    if os.getenv("MAX_BLOCK_EXCLUDE") is not None:
        MAX_BLOCK_EXCLUDE = int(os.getenv("MAX_BLOCK_EXCLUDE"))
    if os.getenv("MIN_BLOCK_EXCLUDE") is not None:
        MIN_BLOCK_EXCLUDE = int(os.getenv("MIN_BLOCK_EXCLUDE"))
    if os.getenv("IGNORE") is not None:
        IGNORE_LIST = {int(x.strip()) for x in os.getenv("IGNORE").split(",")}

    do_main_func(sys.argv[1])
