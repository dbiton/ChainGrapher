import json
import os.path
from typing import Dict, Tuple, Set, Any, List
import httpx
import time
import networkx as nx
from httpx import TimeoutException, HTTPStatusError
from matplotlib.figure import Figure
import pandas as pd
import logging
logger = logging.getLogger(__name__)

import datetime, urllib.parse
class Interface:

    def __init__(self, fetch_parallel: bool, url_rpc: str):
        self.fetch_parallel = fetch_parallel
        self.url_rpc = url_rpc
        self.url_domain = urllib.parse.urlparse(url_rpc).netloc

    def get_timestamp(self):
        return datetime.datetime.now().astimezone().strftime(f"%Y-%m-%d_%H:%M:%S.%f/{self.url_domain}")
    def fetch(self, block_number: int):
        pass
    
    def get_conflict_graph(self, block_trace) -> Tuple[Dict[str, Set[str]],Dict[str, Set[str]]]:
        pass
    
    def get_additional_figures(self, df: pd.DataFrame) -> List[Tuple[str, Figure]]:
        return []
    
    def get_additional_metrics(self, block_number, trace) -> Dict[str, float]:
        return {"block_number": block_number}
    
    def _create_conflict_graph_from_readset_writeset(self, txs: List[str], reads: Dict[str, Set[str]], writes: Dict[str, Set[str]]) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(txs)
        for tx0_hash, tx0_writes in writes.items():
            for tx1_hash, tx1_reads in reads.items():
                if tx0_hash != tx1_hash and not tx0_writes.isdisjoint(tx1_reads):
                    G.add_edge(tx0_hash, tx1_hash)
            for tx1_hash, tx1_writes in writes.items():
                if tx0_hash != tx1_hash and not tx0_writes.isdisjoint(tx1_writes):
                    G.add_edge(tx0_hash, tx1_hash)
        return G
    def create_conflict_graph_from_writewrite_only(self, txs: List[str], writes: Dict[str, Set[str]]) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(txs)
        for tx0_hash, tx0_writes in writes.items():
            for tx1_hash, tx1_writes in writes.items():
                if tx0_hash != tx1_hash and not tx0_writes.isdisjoint(tx1_writes):
                    G.add_edge(tx0_hash, tx1_hash)
        return G
    
    def _post_with_retry(self, payload: Any,pathname:str=None, timeout: int = 600, max_retries: int=10, base_delay: float=2.0) -> httpx.Response:
        if pathname is not None:
            if os.path.exists(pathname):
                try:
                    with open(pathname, 'r') as f:
                        j = json.load(f)

                    logger.info(f'Found without fetch {pathname}')
                    return j
                except json.decoder.JSONDecodeError as e:
                    logger.error(f"error in {pathname} : {e}")
                    raise Exception(f"{pathname}:: error in {e}")
        delay = base_delay
        logger.info(f'Doing HTTP for {pathname}')
        for attempt in range(1, max_retries + 1):
            try:
                response = httpx.post(self.url_rpc, json=payload, timeout=timeout)
                if response.status_code != 200:
                    try:
                        j = response.json()
                    except json.JSONDecodeError:
                        j =None
                    if 'error' in j and j['error']['code'] == 429:
                        print(e, flush=True)
                        # os.remove(str(e).split("::")[0])
                        os._exit(429)
                    raise httpx.HTTPStatusError(
                        f"{response.status_code} {response.reason_phrase}",
                        request=response.request,
                        response=response,
                    )
                # print(repr(response))
                j = response.json()
                if pathname is not None:
                    with open(pathname+"_temp", "w") as f:
                        json.dump(j,f)
                    os.rename(pathname+"_temp",pathname)
                    logger.info(f'Server fetch stored locally {pathname}')
                return j
                # Option A: force UTF-8
                # response.encoding = "utf-8"
                # import json
                # data = json.loads(response.text)
                # return data
            except TimeoutException as timeout_exception:
                logger.warning(f"[Attempt {attempt}] Timeout Exception: {e}")
                if attempt == max_retries:
                    logger.error("Max retries reached. Giving up.")
                    raise
                time.sleep(delay)
                delay *= 2
            except httpx.HTTPStatusError as e:
                logger.warning(f"[Attempt {attempt}] HTTP Error: {e} \n\tRequest: {e.request}{e.request.content}\n\tResponse: {e.response.content}")
                if attempt == max_retries:
                    logger.error("Max retries reached. Giving up.")
                    raise
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                logger.warning(f"[Attempt {attempt}] Error: {e}")
                if attempt == max_retries:
                    logger.error("Max retries reached. Giving up.")
                    raise
                time.sleep(delay)
                delay *= 2

