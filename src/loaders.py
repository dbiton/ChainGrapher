import json
import logging
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
                            logging.info(f"loaded {i_entry} values from {filepath}")
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
            logging.error("No traces file found.")
    except Exception as e:
        logging.error(f"Failed loading {filepath} due to {e}")
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
        logging.error("No traces file found.")