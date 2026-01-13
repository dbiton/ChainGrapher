import csv
import glob
import logging

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

import re
import sys
from itertools import islice
import os

import infixpy
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
import itertools
import concurrent.futures
from infixpy import *

import loaders

load_dotenv(override=False)  # Prioritize environment first

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
import fetchers

sui_interface = SuiInterface()
iota_interface = IotaInterface()
eth_interface = EthPerstateInterface()
solana_interface = SolanaInterface()
crypto_interface = solana_interface

main_func_registry = {}

MAX_BLOCK_EXCLUDE, MIN_BLOCK_EXCLUDE = None, None
IGNORE_LIST = []


def register_entrypoint(func, name):
    global main_func_registry
    key = func.__name__ if name is None else name
    main_func_registry[key] = func
    return func


def entrypoint(_func=None, name=None):
    if _func is not None and callable(_func):
        return register_entrypoint(_func, name)
    else:
        return lambda _func: register_entrypoint(_func, name)


def do_main_func(key):
    global main_func_registry
    main_func_registry[key]()


def process_trace(block_number, *trace_args):
    logger.info(f"Processing {block_number}...")
    logger.info(f"Getting additional metrics {block_number}...")
    trace_args = list(trace_args)
    # trace_args[1] = [tx for tx in trace_args[1] if crypto_interface._get_tx_type(tx) in USER_KINDS]
    metrics = crypto_interface.get_additional_metrics(block_number, trace_args)
    logger.info(f"Creating conflict graph {block_number}...")
    G = crypto_interface.get_conflict_graph(trace_args)
    logger.info(f"Getting graph metrics {block_number}...")
    metrics.update(get_graph_metrics(G))
    return metrics


@entrypoint(name="graph")
def plot_conflict_graph():
    block_number = 390000000  # int(sys.argv[2])
    b = loaders.get_single_block(no=block_number)
    G = crypto_interface.get_conflict_graph([b[1]['result']])
    plot_graph(G)
    print(G)


def agg_load_compressed_file(dirpath, limit, k):
    generators = [load_compressed_file(filepath) for filepath in get_files(dirpath, ".h5")]
    it = itertools.chain.from_iterable(generators)
    while True:
        chunk = list(next(it, None) for _ in range(k))
        chunk = [x for x in chunk if x is not None]
        if not chunk:
            break
        # agg_txs = sum([txs for (_, _, txs) in chunk], [])
        # yield [chunk[0][0], chunk[0][1], agg_txs]
        yield from chunk


def generate_data(data_path, output_path):
    # data_generator = agg_load_compressed_file(dirpath, limit,1)
    data_generator = load_compressed_file(data_path)
    write_header = True  # not os.path.exists(output_path)
    max_pending = int(os.getenv("MAX_PENDING", 6))

    with open(output_path, mode="w", newline="") as file:
        max_workers = int(os.getenv("MAX_METRIC_WORKERS", -1))
        if max_workers == -1:
            max_workers = None
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(process_trace, *data): data for data in islice(data_generator, max_pending)}
            all_submitted = len(futures) < max_pending
            writer = csv.writer(file)
            i = 0
            while futures:
                for future in concurrent.futures.as_completed(futures):
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
                        # logger.info()
                        logger.info(f"wrote result {i} to output csv: {result}")
                        i += 1
                    if not all_submitted:
                        try:
                            next_data = next(data_generator)
                            new_future = pool.submit(process_trace, *next_data)
                            futures[new_future] = next_data
                        except StopIteration:
                            all_submitted = True
                    break  # Exit early to allow re-entering as_completed with updated futures


def process_newdata(newcols, block_number, trace_args, df):
    logger.info(f"Processing {block_number}...")
    logger.info(f"Getting additional metrics {block_number}...")
    # trace_args = list(trace_args)
    # trace_args[1] = [tx for tx in trace_args[1] if crypto_interface._get_tx_type(tx) in USER_KINDS]
    logger.info(f"Creating conflict graph {block_number}...")
    G, txs, reads, writes = crypto_interface.get_conflict_graph_rwsets([trace_args])
    G: nx.Graph = G
    block = trace_args

    new_metrics = {}
    logger.info(f"Getting graph metrics {block_number}...")

    if 'edge-count' in newcols:
        new_metrics['edge_count'] = len(G)

    if 'no-ww-conflics' in newcols:
        G_W = crypto_interface.create_conflict_graph_from_writewrite_only(txs, writes)
        new_metrics['wwconflicts_count'] = len(G_W.edges)
        new_metrics['wwconflicts_exclusive'] = len(nx.difference(G_W, G).edges)

    field_stuff = {
        'fee': (lambda tx: tx['meta']['fee'],),
        'computeUnitsConsumed': (lambda tx: tx['meta']['computeUnitsConsumed'],),
        'costUnits': (lambda tx: tx['meta']['costUnits'],),
        'failed': (lambda tx: 0 if tx['meta']['err'] is None else 1,)
    }
    # "sumof::fee"
    # "sumof::computeUnitsConsumed"
    # "sumof::costUnits"
    # "sumof::failed"
    def sumof(field: str):
        nonlocal new_metrics
        func, = field_stuff[field]
        new_metrics["sumof_" + field] = sum(func(tx) for tx in block['transactions'])

    for field in Seq(newcols).map(lambda x: re.match(r"sumof::(\S+)", x)).filter(lambda x: x is not None).map(
            lambda x: x.group(1)):
        sumof(field)

    # crypto_interface.get_additional_metrics(block_number, trace_args)
    # metrics.update(get_graph_metrics(G))
    return df, new_metrics


def generate_csv_data_pairs(data_path, input_path):
    data_generator = load_compressed_file(data_path)
    dfs = pd.read_csv(input_path, dtype_backend='numpy_nullable').sort_values(by='block_number')
    df = iter(dfs.iterrows())

    # _, cols = next(df)
    df_i = dict(next(df)[1])
    for data_i in data_generator:
        if df_i['block_number'] == data_i[1]['parentSlot'] + 1:
            yield *data_i, df_i
            df_i = dict(next(df)[1])
# import concurrent.futures, threading
# class DummyExecutor(concurrent.futures.Executor):
#
#     def __init__(self,*args, **kwargs):
#         self._shutdown = False
#         self._shutdownLock = threading.Lock()
#
#     def submit(self, fn, *args, **kwargs):
#         with self._shutdownLock:
#             if self._shutdown:
#                 raise RuntimeError('cannot schedule new futures after shutdown')
#
#             f = concurrent.futures.Future()
#             try:
#                 result = fn(*args, **kwargs)
#             except BaseException as e:
#                 f.set_exception(e)
#             else:
#                 f.set_result(result)
#
#             return f
#
#     def shutdown(self, wait=True):
#         with self._shutdownLock:
#             self._shutdown = True
def generate_additional_data(data_path, input_path, output_path, newcols):
    datapair_generator = generate_csv_data_pairs(data_path, input_path)
    write_header = True

    max_pending = int(os.getenv("MAX_PENDING", 6))

    with open(output_path, mode="w", newline="") as file:
        max_workers = int(os.getenv("MAX_METRIC_WORKERS", -1))
        if max_workers == -1:
            max_workers = None
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        # ThreadPoolExecutor = DummyExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(process_newdata, newcols, *data): data for data in
                       islice(datapair_generator, max_pending)}
            all_submitted = len(futures) < max_pending
            writer = csv.writer(file)
            i = 0
            while futures:
                for future in concurrent.futures.as_completed(futures):
                    existingvalues, result = future.result()
                    del futures[future]
                    if result is not None:
                        result.update(existingvalues)
                        sorted_keys = sorted(result.keys())
                        sorted_values = [result[k] for k in sorted_keys]
                        if write_header:
                            write_header = False
                            writer.writerow(sorted_keys)
                        writer.writerow(sorted_values)
                        file.flush()
                        # logger.info()
                        logger.info(f"wrote result {i} to output csv: {result}")
                        i += 1
                    if not all_submitted:
                        try:
                            next_data = next(datapair_generator)
                            new_future = pool.submit(process_newdata, newcols, *next_data)
                            futures[new_future] = next_data
                        except StopIteration:
                            all_submitted = True
                    break  # Exit early to allow re-entering as_completed with updated futures


def get_files(folder_path, extension):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)]


@entrypoint
def do_more_metrics():
    # output_path = "metrics_sui_big.csv"
    dirpath = "./data/download/solana/"
    # if os.path.exists(output_path):
    #     os.remove(output_path)
    newcols = sys.argv[2:]
    for datapath, range in (
            Seq(os.listdir(f"{dirpath}/chunks"))
                    .map(lambda x: re.match(r"((\d+)_(\d+)).h5", x))
                    .filter(lambda x: x is not None)
                    .filter(lambda x: (MIN_BLOCK_EXCLUDE is None) or (int(x.group(2)) >= MIN_BLOCK_EXCLUDE))
                    .filter(lambda x: (MAX_BLOCK_EXCLUDE is None) or (int(x.group(3)) <= MAX_BLOCK_EXCLUDE))
                    .filter(lambda x: (int(x.group(2)) not in IGNORE_LIST) and (int(x.group(3)) not in IGNORE_LIST))
                    .filter(lambda x: not os.path.exists(f"{dirpath}/metrics/updated_{x.group(1)}.csv"))
                    .map(lambda x: (f"{dirpath}/chunks/{x.group(0)}", x.group(1)))
                    .sortby(lambda x: x[1])):
        generate_additional_data(datapath, f"{dirpath}/metrics/{range}.csv", f"{dirpath}/metrics/temp_{range}.csv",
                                 newcols)
        os.rename(f"{dirpath}/metrics/temp_{range}.csv", f"{dirpath}/metrics/updated_{range}.csv")


@entrypoint(name="metrics")
def do_metrics():
    # output_path = "metrics_sui_big.csv"
    dirpath = "./data/download/solana/"
    # if os.path.exists(output_path):
    #     os.remove(output_path)
    for datapath, range in (
            Seq(os.listdir(f"{dirpath}/chunks"))
                    .map(lambda x: re.match(r"((\d+)_(\d+)).h5", x))
                    .filter(lambda x: x is not None)
                    .filter(lambda x: (MIN_BLOCK_EXCLUDE is None) or (int(x.group(2)) >= MIN_BLOCK_EXCLUDE))
                    .filter(lambda x: (MAX_BLOCK_EXCLUDE is None) or (int(x.group(3)) <= MAX_BLOCK_EXCLUDE))
                    .filter(lambda x: (int(x.group(2)) not in IGNORE_LIST) and (int(x.group(3)) not in IGNORE_LIST))
                    .filter(lambda x: not os.path.exists(f"{dirpath}/metrics/{x.group(1)}.csv"))
                    .map(lambda x: (f"{dirpath}/chunks/{x.group(0)}", x.group(1)))
                    .sortby(lambda x: x[1])):
        generate_data(datapath, f"{dirpath}/metrics/temp_{range}.csv")
        os.rename(f"{dirpath}/metrics/temp_{range}.csv", f"{dirpath}/metrics/{range}.csv")


@entrypoint
def do_plots():
    all_files = (Seq(glob.glob(os.path.join("./data/download/solana/metrics", "*.csv")))
                 .map(lambda x: (x, re.match(r".+/((\d+)_(\d+)).csv", x)))
                 .filter(lambda x: x[1] is not None)
                 .filter(lambda x: (MIN_BLOCK_EXCLUDE is None) or (int(x[1].group(2)) >= MIN_BLOCK_EXCLUDE))
                 .filter(lambda x: (MAX_BLOCK_EXCLUDE is None) or (int(x[1].group(3)) <= MAX_BLOCK_EXCLUDE))
                 .filter(lambda x: (int(x[1].group(2)) not in IGNORE_LIST) and (int(x[1].group(3)) not in IGNORE_LIST))
                 .sortby(lambda x: x[1].group(1))
                 .map(lambda x: x[0])
                 .tolist()
                 )

    plot_data_filelist(all_files, crypto_interface)


def download_files(start: int, end: int, dirpath: str, filesize: int):
    assert (filesize % CHUNK_SIZE == 0)
    count = end - start
    assert (start < end)
    assert (count % filesize == 0)
    for begin in list(range(start, end, filesize)):
        end = begin + filesize
        filename = f"{begin}_{end - 1}.h5"
        fetcher_multiple = fetchers.fetch_parallel_2 if crypto_interface.fetch_parallel else fetchers.fetch_serial
        # run once to save files locally before making compressed file
        for _ in fetcher_multiple(range(begin, end), crypto_interface.fetch):
            pass

        # run again to load local files to make compressed file
        traces_generator = fetchers.fetch_serial(range(begin, end), crypto_interface.fetch)
        save_to_file(os.path.join(dirpath, filename), traces_generator)
        # remove allfiles from cache
        crypto_interface.remove_cached_files(range(begin, end))


@entrypoint
def do_download():
    logger.info("Starting download")
    # start_block = 390_000_000
    start_block = int(os.getenv("START_BLOCK"))
    # start_block = 385_280_000
    # start_block = 386_280_000
    # start_block = 388_420_000
    # count = 100_000
    count = int(os.getenv("BLOCK_COUNT"))
    dirpath = "./data/download/solana/"
    interfaces.solana_interface.DIR_PATH = f"{dirpath}/inprog"
    current_start = (
        Seq(os.listdir(f"{dirpath}/chunks/"))
        .map(lambda x: re.match(r"((\d+)_(\d+)).h5", x))
        .filter(lambda x: x is not None)
        .filter(lambda x: (MIN_BLOCK_EXCLUDE is None) or (int(x.group(2)) >= MIN_BLOCK_EXCLUDE))
        .filter(lambda x: (MAX_BLOCK_EXCLUDE is None) or (int(x.group(3)) <= MAX_BLOCK_EXCLUDE))
        .filter(lambda x: (int(x.group(2)) not in IGNORE_LIST) and (int(x.group(3)) not in IGNORE_LIST))
        .map(lambda x: int(x.group(3)) + 1)
        .chain([start_block])
        .reduce(max))
    download_files(start=current_start, end=start_block + count, dirpath=f"{dirpath}/chunks",
                   filesize=1_000)


if __name__ == "__main__":
    # logger.basicConfig(format='%(message)s', level=logger.BASIC_FORMAT)
    logging.basicConfig(level=logging.INFO)
    # do_download()
    # main()
    if os.getenv("MAX_BLOCK_EXCLUDE") is not None:
        MAX_BLOCK_EXCLUDE = int(os.getenv("MAX_BLOCK_EXCLUDE"))
    if os.getenv("MIN_BLOCK_EXCLUDE") is not None:
        MIN_BLOCK_EXCLUDE = int(os.getenv("MIN_BLOCK_EXCLUDE"))
    if os.getenv("IGNORE") is not None:
        IGNORE_LIST = {int(y) for x in os.getenv("IGNORE").split(",") if (y := x.strip()) != ""}

    do_main_func(sys.argv[1])
