import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Any, Iterable
from concurrent.futures.thread import ThreadPoolExecutor

def killProcessOnExecption(func):
    def func_wrapper(i):
        try:
            return func(i)
        except Exception as e:
            print(e,flush=True)
            # os.remove(str(e).split("::")[0])
            os._exit(1)
    return func_wrapper

def fetch_parallel(it: Iterable[int], fetcher: Callable[[int], Any]):
    fetcher = killProcessOnExecption(fetcher)
    max_workers = int(os.getenv("MAX_FETCH_WORKERS", 25))
    with ProcessPoolExecutor(max_workers=25) as executor:
        futures = [executor.submit(fetcher, i) for i in it]
        for future in futures:
            result = future.result()
            if result:
                yield result

def fetch_serial(it: Iterable[int], fetcher: Callable[[int], Any]):
    for i in it:
        result = fetcher(i)
        if result:
            yield result