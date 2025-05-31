from typing import Callable, Any, Iterable
from concurrent.futures.thread import ThreadPoolExecutor

def fetch_parallel(it: Iterable[int], fetcher: Callable[[int], Any]): 
    with ThreadPoolExecutor() as executor:
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