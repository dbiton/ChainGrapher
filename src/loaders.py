import glob
import itertools
import json
import logging

import h5pickle

logger = logging.getLogger(__name__)

import os
import sys
import zlib
import h5py
from concurrent.futures import ThreadPoolExecutor

def uncompress_chunk(dset, i):
    chunk = dset[i]
    if len(chunk) == 0:
        return []
    chunk = bytes(chunk)
    chunk = zlib.decompress(chunk)
    chunk = chunk.decode('ascii')
    chunk = json.loads(chunk)
    return chunk
def uncompress_chunk_given(arg):
    i, chunk = arg
    if len(chunk) == 0:
        return []
    chunk = zlib.decompress(chunk)
    chunk = chunk.decode('ascii')
    chunk = json.loads(chunk)
    return i, chunk
def read_chunk_given(arg):
    i, dset=arg
    return i, bytes(dset[i])

def load_compressed_file(filepath: str, limit=None):
    try:
        if os.path.exists(filepath+"_errors.txt"):
            os.remove(filepath+"_errors.txt")
        if os.path.exists(filepath):
            with h5py.File(filepath, 'r') as f:
                dset = f['dataset']
                i_entry = 0
                chunk_count = dset.shape[0]
                max_pending = 2
                with ThreadPoolExecutor(max_pending) as pool:
                    futures = [
                        pool.submit(uncompress_chunk, dset, i_chunk) 
                        for i_chunk in range(min(max_pending, chunk_count))
                    ]
                    i_chunk = len(futures)
                    while len(futures) > 0:
                        future = futures[0]
                        futures = futures[1:]
                        entries = future.result()
                        for (i,entry) in entries:
                            i_entry += 1
                            logger.info(f"loaded {i_entry} values from {filepath}")
                            if "error" in entry:
                                with open(filepath+"_errors.txt", 'a') as f:
                                    f.write(f"{i_entry}\t{entry['error']['message']}\n")
                                continue
                            yield [i, entry["result"]]
                            if i_entry == limit:
                                return
                        if i_chunk < chunk_count:
                            futures.append(pool.submit(uncompress_chunk, dset, i_chunk))
                            i_chunk += 1
        else:
            logger.error("No traces file found.")
    except Exception as e:
        logger.error(f"Failed loading {filepath} due to {e}")
        print(repr(e))
        os._exit(1)

def uncompress_chunk_loading(i, filepath):
    with h5py.File(filepath, 'r') as f_h5:
        dset = f_h5['dataset']
        chunk = dset[i]
        if len(chunk) == 0:
            return []
        chunk = bytes(chunk)
        chunk = zlib.decompress(chunk)
        chunk = chunk.decode('ascii')
        chunk = json.loads(chunk)
        return i, chunk
def load_compressed_file_executor(filepath: str, pool,max_pending):
    try:
        if os.path.exists(filepath+"_errors.txt"):
            os.remove(filepath+"_errors.txt")
        if os.path.exists(filepath):
            with (h5pickle.File(filepath, 'r') as f_h5,
                  ThreadPoolExecutor(max_workers=max_pending) as iopool,
                    # (h5py.File(filepath, 'r') as f_h5,
                  open(filepath+"_errors.txt", 'a') as f_err):
                dset = f_h5['dataset']
                i_entry = 0
                chunk_count = dset.shape[0]
                reads = iopool.map(read_chunk_given,map(lambda x: (x, dset),range(chunk_count)))
                futures = [
                    # pool.submit(uncompress_chunk_loading, filepath, i_chunk_submitted)
                    # pool.submit(uncompress_chunk, dset, i_chunk_submitted)
                    pool.submit(uncompress_chunk_given, chunk)
                    for chunk in itertools.islice(reads,min(max_pending, chunk_count))
                ]
                i_chunk_submitted = len(futures)
                while len(futures) > 0:
                    future = futures[0]
                    futures = futures[1:]
                    # entries = future.result()
                    i_chunk,entries = future.result()
                    if i_chunk_submitted < chunk_count:
                        # futures.append(pool.submit(uncompress_chunk_loading, filepath, i_chunk_submitted))
                        # futures.append(pool.submit(uncompress_chunk, dset, i_chunk_submitted))
                        pool.submit(uncompress_chunk_given, next(reads))
                        i_chunk_submitted += 1
                    for (i,entry) in entries:
                        i_entry += 1
                        logger.info(f"loaded {i_entry} values from {filepath}")
                        if "error" in entry:
                            f_err.write(f"{i_entry}\t{entry['error']['message']}\n")
                            continue
                        yield [i, entry["result"]]

        else:
            logger.error("No traces file found.")
    except Exception as e:
        logger.error(f"Failed loading {filepath} due to {e}")
        print(repr(e))
        os._exit(1)

def load_file(filepath: str, limit=None):
    if os.path.exists(filepath):
        with h5py.File(filepath, 'r') as f:
            dset = f['dataset']
            limit = limit if limit is not None else dset.shape[0]
            for i in range(limit):
                value = json.loads(dset[i])
                yield value  # Read one line at a time
    else:
        logger.error("No traces file found.")


def get_single_block(dir="./data/download/solana/chunks", no=0):
    filestart = ((int(os.getenv("START_BLOCK")) - no) // 1000) * 1000 + int(os.getenv("START_BLOCK"))
    i_chunk = ((int(os.getenv("START_BLOCK")) - no) % 1000)
    filepath = glob.glob(dir+ f"/{filestart}_*.h5")[0]
    with h5py.File(filepath, 'r') as f:
        dset = f['dataset']
        return uncompress_chunk(dset, i_chunk)[i_chunk]