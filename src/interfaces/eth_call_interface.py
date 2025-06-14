from typing import Dict, Set, List
from interfaces.interface import Interface
import os
import networkx as nx
import numpy as np


class EthCallInterface(Interface):

    def __init__(self):
        rpc_url = os.getenv("ETH_RPC_URL")
        super().__init__(True, rpc_url)

    def fetch(self, block_number: int):
        trace = self._fetch_block_trace(block_number, "callTracer")
        return block_number, trace

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

    def _create_call_graph_recursive(self, G: nx.Graph, call) -> None:
        node_id = G.number_of_nodes()
        G.add_node(node_id)
        if 'calls' in call:
            for call in call['calls']:
                child_id = self._create_call_graph_recursive(G, call)
                G.add_edge(node_id, child_id)
        return node_id

    def _create_call_graphs(self, trace) -> List[nx.Graph]:
        graphs = []
        for tx in trace:
            G = nx.Graph()
            self._create_call_graph_recursive(G, tx['result'])
            graphs.append(G)
        return graphs 

    def get_additional_metrics(self, block_number, trace) -> Dict[str, float]:
        call_graphs = self._create_call_graphs(trace)
        call_graphs_smart_contracts = [G for G in call_graphs if G.number_of_nodes() > 1]
        call_counts = [G.number_of_nodes() for G in call_graphs_smart_contracts]
        call_heights = [1 + max(nx.shortest_path_length(G, source=0).values()) for G in call_graphs_smart_contracts]
        call_degrees = [np.mean([val for (_, val) in G.degree()]) for G in call_graphs_smart_contracts]
        call_leaves = [np.sum([1 for (node, val) in G.degree() if node != 0]) for G in call_graphs_smart_contracts]
        count_txs_value_transfer = len(call_graphs) - len(call_graphs_smart_contracts)
        return {
            "block_number": block_number, 
            "txs": len(trace),
            "mean_call_count_smart_contract": np.mean(call_counts),
            "mean_call_height_smart_contract": np.mean(call_heights),
            "mean_call_degree_smart_contract": np.mean(call_degrees),
            "mean_call_count_leaves_smart_contract": np.mean(call_leaves),
            "count_txs_value_transfer": count_txs_value_transfer
        }
        
    def _parse_trace_calls(self, call, reads, writes, inherited_prems=[True, True, True, True]):
        # CALLTYPE, RF, WF, RT, WT
        calls_prems = {
            "CALL":[True, False, True, True],
            "DELEGATECALL":[True, True, True, False],
            "CALLCODE":[True, True, True, False],
            "CREATE":[True, True, False, True],
            "CREATE2":[True, True, False, True],
            "STATICCALL":[True, False, True, False],
            "SELFDESTRUCT":[True, False, False, True],
            "SUICIDE":[True, False, False, True],
            "INVALID":[False, False, False, False],
            "REVERT":[False, False, False, False],
        }
        
        call_type = call["type"]
        prems = [p0 and p1 for (p0, p1) in zip(calls_prems[call_type], inherited_prems)]
        from_addr = call["from"]
        to_addr = call["to"]
        
        if prems[0]:
            reads.add(from_addr)
        if prems[1]:
            writes.add(from_addr)
        if prems[2]:
            reads.add(to_addr)
        if prems[3]:
            writes.add(to_addr)

        if "calls" in call:
            for sub_call in call["calls"]:
                self._parse_trace_calls(sub_call, reads, writes)

    def get_conflict_graph(self, block_trace):
        writes: Dict[str, Set[str]] = {}
        reads: Dict[str, Set[str]] = {}

        for entry in block_trace:
            tx = entry["result"]
            tx_hash = entry["txHash"]
            if "calls" in tx:
                for call in tx["calls"]:
                    iter_reads = set()
                    iter_writes = set()
                    self._parse_trace_calls(call, iter_reads, iter_writes)
                    if tx_hash not in writes:
                        writes[tx_hash] = iter_writes
                    else:
                        writes[tx_hash].update(iter_writes)
                    if tx_hash not in reads:
                        reads[tx_hash] = iter_reads
                    else:
                        reads[tx_hash].update(iter_reads)
        
        txs = [tx_trace["txHash"] for tx_trace in block_trace]
        
        conflict_graph = self._create_conflict_graph_from_readset_writeset(txs, reads, writes)
        
        return conflict_graph
