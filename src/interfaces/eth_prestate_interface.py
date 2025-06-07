from typing import Dict, Set
from interfaces.interface import Interface
import os


class EthPerstateInterface(Interface):

    def __init__(self):
        rpc_url = os.getenv("ETH_RPC_URL")
        super().__init__(True, rpc_url)

    def fetch(self, block_number: int):
        diffFalse = self._fetch_block_trace(block_number, "prestateTracer", {"diffMode": False})
        diffTrue = self._fetch_block_trace(block_number, "prestateTracer", {"diffMode": True})
        return block_number, diffFalse, diffTrue

    def _fetch_block_trace(self, block_number: str, tracer_name: str, tracer_config={}) -> dict:
        if tracer_name not in ["callTracer", "prestateTracer"]:
            raise Exception(f"unknown tracer type {tracer_name}")
        if tracer_config not in [{}, {"diffMode": True}, {"diffMode": False}]:
            raise Exception(f"unknown tracer config {tracer_config}")
        payload = {
            "jsonrpc": "2.0",
            "method": "debug_traceBlockByNumber",
            "params": [
                hex(block_number),
                {
                    "tracer": tracer_name,
                    "tracerConfig": tracer_config
                }
            ],
            "id": 1
        }
        return self._post_with_retry(payload)
    
    def get_conflict_graph(self, block_trace):
        block_trace_diffFalse, block_trace_diffTrue = block_trace
        
        writes: Dict[str, Set[str]] = {}
        reads: Dict[str, Set[str]] = {}
        
        for entry in block_trace_diffTrue:
            tx = entry["result"]
            tx_hash = entry["txHash"]
            tx_writes = set(tx['pre'])
            tx_writes.update(set(tx['post']))
            if len(tx_writes) > 0:
                writes[tx_hash] = tx_writes
        
        for entry in block_trace_diffFalse:
            tx = entry["result"]
            tx_hash = entry["txHash"]
            tx_reads = set(tx).difference(writes.get(tx_hash, set()))
            if len(tx_reads) > 0:
                reads[tx_hash] = set(tx).difference(writes.get(tx_hash, set()))
        
        txs = [tx_trace["txHash"] for tx_trace in block_trace_diffFalse]

        return self._create_conflict_graph_from_readset_writeset(txs, reads, writes)

    def get_additional_metrics(self, block_number, block_trace) -> Dict[str, float]:
        block_trace_diffFalse, _ = block_trace
        return {
            "block_number": block_number, 
            "txs": len(block_trace_diffFalse)
        }