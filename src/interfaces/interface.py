from typing import Dict, Tuple, Set, Any, List
import httpx
import time
import networkx as nx
from matplotlib.figure import Figure
import pandas as pd

class Interface:

    def __init__(self, fetch_parallel: bool, url_rpc: str):
        self.fetch_parallel = fetch_parallel
        self.url_rpc = url_rpc
    
    def fetch(self, block_number: int):
        pass
    
    def get_conflict_graph(self, block_trace) -> Tuple[Dict[str, Set[str]],Dict[str, Set[str]]]:
        pass
    
    def get_additional_figures(self, df: pd.DataFrame) -> List[Tuple[str, Figure]]:
        return []
    
    def get_additional_metrics(self, block_number, trace) -> Dict[str, float]:
        return {}
    
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
    
    def _post_with_retry(self, payload: Any, timeout: int = 600, max_retries: int=10, base_delay: float=2.0) -> httpx.Response:
        delay = base_delay
        for attempt in range(1, max_retries + 1):
            try:
                response = httpx.post(self.url_rpc, json=payload, timeout=timeout)
                if response.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"{response.status_code} {response.reason_phrase}",
                        request=response.request,
                        response=response,
                    )
                return response.json()["result"]
            except Exception as e:
                print(f"[Attempt {attempt}] Error: {e}")
                if attempt == max_retries:
                    print("Max retries reached. Giving up.")
                    raise
                time.sleep(delay)
                delay *= 2
