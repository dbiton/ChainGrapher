import logging
from itertools import count
import json
import os
import h5py
import zlib
import numpy as np

CHUNK_SIZE = 100

def append_to_file(filename: str, generator, limit=None) -> None:
    if not os.path.exists(filename):
        raise Exception(f"{filename} doesn't exist!")
    chunk_size = CHUNK_SIZE # chuck compressed list of blocks. One file is 100 chunks
    is_done = False
    for i in count(0, chunk_size):
        i_chunk = i // chunk_size
        start = i
        end = start
        chunk = []
        for _ in range(chunk_size):
            try:
                chunk.append(next(generator))
                end += 1
                if end == limit:
                    is_done = True
                    break
            except StopIteration:
                is_done = True
                break
        if end - start == 0:
            return
        chunk = json.dumps(chunk)
        chunk = chunk.encode('ascii')
        chunk = zlib.compress(chunk, 7)
        chunk = np.frombuffer(chunk, dtype=np.uint8)
        with h5py.File(filename, 'a') as f:
            dset = f['dataset']
            dset.resize((i_chunk + 1,))
            dset[i_chunk] = chunk
        logging.info(f"Saved {end} values in total to {filename}")
        if is_done:
            break


def save_to_file(filename: str, generator, limit=None) -> None:
    if os.path.exists(filename):
        raise Exception(f"{filename} already exists!")
    dirname,fname= os.path.split(filename)
    tempname = os.path.join(dirname,"temp_"+fname)
    with h5py.File(tempname, 'w') as f:
        f.create_dataset(
            'dataset',
            maxshape=(None,),
            shape=(0,),
            dtype=h5py.vlen_dtype(np.dtype('uint8')),
        )
    append_to_file(tempname, generator, limit)
    os.rename(tempname,filename)
