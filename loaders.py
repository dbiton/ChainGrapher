import json
import os
import pickle
import zlib

import h5py

from parsers import apply_recursively, bytes_to_hex
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

def load_compressed_file(filepath: str, limit=None):
    try:
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
                        for entry in entries:
                            i_entry += 1
                            print(f"loaded {i_entry} values from {filepath}")
                            yield entry
                            if i_entry == limit:
                                return
                        if i_chunk < chunk_count:
                            futures.append(pool.submit(uncompress_chunk, dset, i_chunk))
                            i_chunk += 1
        else:
            print("No traces file found.")
    except Exception as e:
        print(f"Failed loading {filepath} due to {e}")

def load_file(filepath: str, limit=None):
    if os.path.exists(filepath):
        with h5py.File(filepath, 'r') as f:
            dset = f['dataset']
            limit = limit if limit is not None else dset.shape[0]
            for i in range(limit):
                value = json.loads(dset[i])
                yield value  # Read one line at a time
    else:
        print("No traces file found.")