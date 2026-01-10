import concurrent
import itertools
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

def fetch_parallel_thread(it: Iterable[int], fetcher: Callable[[int], Any]):
    # fetcher = killProcessOnExecption(fetcher)
    max_workers = int(os.getenv("MAX_FETCH_WORKERS", 25))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            futures = (executor.submit(fetcher, i) for i in it)
            # for future in futures:
            #     result = future.result()
            #     if result:
            #         yield result
            yield from (f.result() for f in futures if f.result())
        except KeyboardInterrupt as e:
            executor.shutdown(wait=True, cancel_futures=True)
            yield from (f.result() for f in futures if f.cancelled() == False)
def fetch_parallel_process(it: Iterable[int], fetcher: Callable[[int], Any]):
    # fetcher = killProcessOnExecption(fetcher)
    max_workers = int(os.getenv("MAX_FETCH_WORKERS", 25))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        try:
            futures = (executor.submit(fetcher, i) for i in it)
            for future in futures:
                result = future.result()
                if result:
                    yield result
        except KeyboardInterrupt as e:
            executor.shutdown(wait=True, cancel_futures=True)
            yield from (f.result() for f in futures if f.cancelled() == False)


def killProcessOnExecption_2(func, *args):
    try:
        return func(*args)
    except KeyboardInterrupt:
        pass



def fetch_parallel_2(it: Iterable[int], fetcher: Callable[[int], Any]):
    it = iter(it)
    max_workers = int(os.getenv("MAX_FETCH_WORKERS", 25))
    max_pending = int(os.getenv("MAX_PENDING",6))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(killProcessOnExecption_2, fetcher, data): data for data in itertools.islice(it, max_pending)}
        all_submitted = len(futures) < max_pending
        while futures:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                del futures[future]
                if result is not None:
                    yield result
                if not all_submitted:
                    try:
                        next_data = next(it)
                        new_future = pool.submit(fetcher, next_data)
                        futures[new_future] = next_data
                    except StopIteration:
                        all_submitted = True
                break  # Exit early to allow re-entering as_completed with updated futures

def fetch_serial(it: Iterable[int], fetcher: Callable[[int], Any]):
    for i in it:
        result = fetcher(i)
        if result:
            yield result