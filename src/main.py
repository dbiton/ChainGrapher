import collections
import contextlib
import csv
import functools
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
EXACT_LIST = []
IGNORE_LIST = []

V="v2"
V_Old,V_new="v2","v3"


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

def lazy(exp, tuple_size=None):
    val = None
    def getter():
        nonlocal val
        if val is None:
            val = exp()
        return val
    if tuple_size is None:
        return getter
    else:
        return tuple(lambda: getter()[i] for i in range(tuple_size))

solana_std_program_ids = {
    # Core Native Programs
    "11111111111111111111111111111111": {
        "name": "System Program",
        "description": "Creates accounts, transfers SOL, and assigns program ownership.",
        "group": "Native"
    },
    "Vote111111111111111111111111111111111111111": {
        "name": "Vote Program",
        "description": "Manages validator voting stakes and state.",
        "group": "Native"
    },
    "Stake11111111111111111111111111111111111111": {
        "name": "Stake Program",
        "description": "Manages SOL staking for delegation to validators.",
        "group": "Native"
    },
    "Config1111111111111111111111111111111111111": {
        "name": "Config Program",
        "description": "Manages chain-wide configuration data.",
        "group": "Native"
    },
    "ComputeBudget111111111111111111111111111111": {
        "name": "Compute Budget Program",
        "description": "Sets compute unit limits and priority fees for transactions.",
        "group": "Native"
    },
    "AddressLookupTab1e1111111111111111111111111": {
        "name": "Address Lookup Table Program",
        "description": "Manages Address Lookup Tables (ALTs) for versioned transactions.",
        "group": "Native"
    },
    "Ed25519SigVerify111111111111111111111111111": {
        "name": "Ed25519 Signature Verify",
        "description": "Verifies Ed25519 signatures (used for cross-chain ops).",
        "group": "Native"
    },
    "KeccakSecp256k11111111111111111111111111111": {
        "name": "Secp256k1 Signature Verify",
        "description": "Verifies Secp256k1 signatures (Ethereum/Bitcoin compatibility).",
        "group": "Native"
    },

    # SPL Programs
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA": {
        "name": "Token Program",
        "description": "The standard for fungible and non-fungible tokens on Solana.",
        "group": "SPL"
    },
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL": {
        "name": "Associated Token Program",
        "description": "Deterministically maps a wallet address to its token accounts.",
        "group": "SPL"
    },
    "MemoSq4gqABAXmK96DPVE9PCrmJ4y5yLPbXIcKBWu3": {
        "name": "Memo Program",
        "description": "Attaches a UTF-8 string (memo) to a transaction (Version 2).",
        "group": "SPL"
    },
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb": {
        "name": "Token-2022 Program",
        "description": "The new Token Extensions program with advanced features.",
        "group": "SPL"
    },
    "namesLPneVptA9Z5rqUDD9tMTWEJwofgaYwp8cawRkX": {
        "name": "Name Service",
        "description": "Manages .sol domain names (Bonfida).",
        "group": "SPL"
    },

    # Loaders
    "BPFLoaderUpgradeab1e11111111111111111111111": {
        "name": "BPF Upgradeable Loader",
        "description": "The standard loader for most user programs.",
        "group": "Loader"
    },
    "BPFLoader2111111111111111111111111111111111": {
        "name": "BPF Loader 2",
        "description": "Legacy loader.",
        "group": "Loader"
    },
    "BPFLoader1111111111111111111111111111111111": {
        "name": "BPF Loader 1",
        "description": "Legacy loader.",
        "group": "Loader"
    },

    # Ecosystem
    "worm2ZoG2kUd4vFXhvjh93UUH596ayRfgQ2MgjNMTth": {
        "name": "Wormhole Core Bridge",
        "description": "Cross-chain bridge infrastructure.",
        "group": "Ecosystem"
    },
    "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s": {
        "name": "Metaplex Token Metadata",
        "description": "Manages NFT metadata standards.",
        "group": "Ecosystem"
    }
}
solana_std_program_idx = sorted(solana_std_program_ids.keys())
solana_std_program_csvheader = "StandardPrograms___"+"__".join(solana_std_program_idx)+"___"
def process_newdata(newcols, block_number, trace_args, df):
    logger.info(f"Processing {block_number}...")
    logger.info(f"Getting additional metrics {block_number}...")
    # trace_args = list(trace_args)
    # trace_args[1] = [tx for tx in trace_args[1] if crypto_interface._get_tx_type(tx) in USER_KINDS]
    logger.info(f"Creating conflict graph {block_number}...")
    # G, txs, reads, writes, programs = lazy(lambda : crypto_interface.get_conflict_graph_rwsets([trace_args]), tuple_size=5)
    G, txs, reads, writes, programs = crypto_interface.get_conflict_graph_rwsets([trace_args])
    #  =  lazy_rwsets()
    # G: nx.Graph = G
    block = trace_args

    new_metrics = {}
    new_metrics_extra_file = {'block_number': block_number}
    logger.info(f"Getting graph metrics {block_number}...")

    if 'isolates' in newcols:
        new_metrics['isolates'] = nx.number_of_isolates(G)

    if 'edge-count' in newcols:
        new_metrics['edge_count'] = len(G())

    if 'instructions':
        totalProgramVector = [0 for _ in range(len(solana_std_program_idx))]
        total_programs = 0
        total_pure_reads = 0
        distinct_nonstd_programs = collections.defaultdict(int)
        using_nonstd_programs = 0
        for tx_id in txs:
            progs = programs[tx_id]
            totalProgramVector = [totalProgramVector[i] + int(solana_std_program_idx[i] in progs) for i in range(len(solana_std_program_idx))]
            total_programs += len(progs)
            nonstd_programs = progs - solana_std_program_ids.keys()
            for nonstd_program in nonstd_programs:
                distinct_nonstd_programs[nonstd_program] +=1
            if nonstd_programs:
                using_nonstd_programs += 1
            total_pure_reads += len(reads[tx_id] - progs)

        new_metrics[solana_std_program_csvheader] = totalProgramVector
        new_metrics['total_programs']= total_programs
        new_metrics['txs_using_nonstd_programs']= using_nonstd_programs
        new_metrics['no_distinct_nonstd_programs']= len(distinct_nonstd_programs)
        new_metrics['total_pure_reads']= total_pure_reads
        new_metrics_extra_file['distinct_nonstd_programs']= dict(distinct_nonstd_programs)

    # if 'no-ww-conflics' in newcols:
    #     G_W = crypto_interface.create_conflict_graph_from_writewrite_only(txs, writes)
    #     new_metrics['wwconflicts_count'] = len(G_W.edges)
    #     new_metrics['wwconflicts_exclusive'] = len(nx.difference(G_W, G).edges)

    field_stuff = {
        'fee': (lambda tx: tx['meta']['fee'],),
        'instructionsCount': (lambda tx: len(tx['transaction']['message']['instructions']),),
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
    # print(df, new_metrics, new_metrics_extra_file)
    return df, new_metrics, new_metrics_extra_file


def generate_csv_data_pairs(data_path, input_path, load_compressed_file=load_compressed_file):
    data_generator = load_compressed_file(data_path)
    dfs = pd.read_csv(input_path, dtype_backend='numpy_nullable').sort_values(by='block_number')
    df = iter(dfs.iterrows())

    # _, cols = next(df)
    df_i = dict(next(df)[1])
    for data_i in data_generator:
        if df_i['block_number'] == data_i[0]:
            yield *data_i, df_i
            try:
                df_i = dict(next(df)[1])
            except StopIteration:
                return
        elif "error" not in data_i:
            print(repr(data_i))
            raise Exception(repr(data_i)+"\nUnexpected error in fetching data" )


import concurrent.futures, threading
class DummyExecutor(concurrent.futures.Executor):

    def __init__(self,*args, **kwargs):
        self._shutdown = False
        self._shutdownLock = threading.Lock()

    def submit(self, fn, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')

            f = concurrent.futures.Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True
def generate_additional_data(data_path, input_path, output_path, newcols, load_compressed_file=load_compressed_file,
                             pool=None):
    datapair_generator = generate_csv_data_pairs(data_path, input_path, load_compressed_file=load_compressed_file)
    write_header = True

    max_pending = int(os.getenv("MAX_PENDING", 6))

    max_workers = int(os.getenv("MAX_METRIC_WORKERS", -1))
    if max_workers == -1:
        max_workers = None
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    # ThreadPoolExecutor = DummyExecutor
    with (open(output_path, mode="w", newline="") as file,
          open(output_path+"_addn", mode="w", newline="") as adden,
          (ProcessPoolExecutor(max_workers=max_workers) if pool is None else contextlib.nullcontext()) as _pool):
          # (DummyExecutor(max_workers=max_workers)) as _pool):
        if pool is None:
            pool = _pool
        futures = {pool.submit(process_newdata, newcols, *data): data for data in
                   islice(datapair_generator, max_pending)}
        all_submitted = len(futures) < max_pending
        writer = csv.writer(file)
        adden_writer = csv.writer(adden)
        i = 0
        while futures:
            for future in concurrent.futures.as_completed(futures):
                existingvalues, result, addendum = future.result()
                del futures[future]
                if result is not None:
                    result.update(existingvalues)
                    sorted_keys = sorted(result.keys())
                    adden_sorted_keys = sorted(addendum.keys())
                    sorted_values = [result[k] for k in sorted_keys]
                    adden_sorted_values = [addendum[k] for k in adden_sorted_keys]
                    if write_header:
                        write_header = False
                        writer.writerow(sorted_keys)
                        adden_writer.writerow(adden_sorted_keys)
                    writer.writerow(sorted_values)
                    adden_writer.writerow(adden_sorted_values)
                    file.flush()
                    adden.flush()
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
    filelist = (Seq(os.listdir(f"{dirpath}/chunks"))
                .map(lambda x: re.match(r"((\d+)_(\d+)).h5", x))
                .filter(lambda x: x is not None)
                .filter(lambda x: (MIN_BLOCK_EXCLUDE is None) or (int(x.group(2)) >= MIN_BLOCK_EXCLUDE))
                .filter(lambda x: (MAX_BLOCK_EXCLUDE is None) or (int(x.group(3)) < MAX_BLOCK_EXCLUDE))
                .filter(lambda x: (int(x.group(2)) in IGNORE_LIST) or (int(x.group(3)) in IGNORE_LIST)
                        or (int(x.group(2)) not in EXACT_LIST) and (int(x.group(3)) not in EXACT_LIST))
                .filter(lambda x: not os.path.exists(f"{dirpath}/metrics/{V_new}/{x.group(1)}.csv"))
                .map(lambda x: (f"{dirpath}/chunks/{x.group(0)}", x.group(1)))
                .sortby(lambda x: x[1])).tolist()
    load_maxpending = int(os.getenv("LOAD_MAXPENDING", 2))
    metric_max_workers = int(os.getenv("MAX_METRIC_WORKERS", -1))
    if metric_max_workers == -1:
        metric_max_workers = None
    with (#ThreadPoolExecutor(load_maxpending) as load_executor_pool,
          ProcessPoolExecutor(max_workers=metric_max_workers) as comp_executor_pool):
        # load_compressed_file_pross = lambda x: loaders.load_compressed_file_executor(x,
        #                                                load_executor_pool, load_maxpending)
        for datapath, range in filelist:
            generate_additional_data(datapath, f"{dirpath}/metrics/{V_Old}/{range}.csv", f"{dirpath}/metrics/{V_new}/temp_{range}.csv",
                                     newcols, pool=comp_executor_pool)
            os.rename(f"{dirpath}/metrics/{V_new}/temp_{range}.csv", f"{dirpath}/metrics/{V_new}/{range}.csv")


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
                    .filter(lambda x: (MAX_BLOCK_EXCLUDE is None) or (int(x.group(3)) < MAX_BLOCK_EXCLUDE))
                    .filter(lambda x: ((int(x.group(2)) not in IGNORE_LIST) and (int(x.group(3)) not in IGNORE_LIST)) or (int(x.group(2)) not in IGNORE_LIST) and (int(x.group(3)) not in IGNORE_LIST))
                    .filter(lambda x: not os.path.exists(f"{dirpath}/metrics/{V}/{x.group(1)}.csv"))
                    .map(lambda x: (f"{dirpath}/chunks/{x.group(0)}", x.group(1)))
                    .sortby(lambda x: x[1])):
        generate_data(datapath, f"{dirpath}/metrics/temp_{range}.csv")
        os.rename(f"{dirpath}/metrics/temp_{range}.csv", f"{dirpath}/metrics/{range}.csv")


@entrypoint
def do_plots():
    all_files = (Seq(glob.glob(os.path.join(f"./data/download/solana/metrics/{V}/", "*.csv")))
                 .map(lambda x: (x, re.match(r".+/((\d+)_(\d+)).csv", x)))
                 .filter(lambda x: x[1] is not None)
                 .filter(lambda x: (MIN_BLOCK_EXCLUDE is None) or (int(x[1].group(2)) >= MIN_BLOCK_EXCLUDE))
                 .filter(lambda x: (MAX_BLOCK_EXCLUDE is None) or (int(x[1].group(3)) < MAX_BLOCK_EXCLUDE))
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
    if os.getenv("EXACT_LIST") is not None:
        EXACT_LIST = {int(y) for x in os.getenv("EXACT_LIST").split(",") if (y := x.strip()) != ""}

    do_main_func(sys.argv[1])
