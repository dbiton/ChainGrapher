from typing import Any, List, Dict, Set
import httpx
import time
from dotenv import load_dotenv
import os
import json
from matplotlib import pyplot as plt
import networkx as nx
import pickle

load_dotenv()

SUI_RPC_URL = os.getenv("SUI_RPC_URL")
PATH_OUTPUT = os.getenv("PATH_TRACE_OUTPUT")


def post_with_retry(payload: Any, max_retries: int=5, base_delay: float=1.0) -> httpx.Response:
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = httpx.post(SUI_RPC_URL, json=payload)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise httpx.HTTPStatusError(
                    f"{response.status_code} {response.reason_phrase}",
                    request=response.request,
                    response=response,
                )
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            print(f"[Attempt {attempt}] Error: {e}")
            if attempt == max_retries:
                print("Max retries reached. Giving up.")
                raise
            time.sleep(delay)
            delay *= 2  # exponential backoff


def get_latest_checkpoint():
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "sui_getLatestCheckpointSequenceNumber",
        "params": []
    }
    response = post_with_retry(payload)
    return response.json()["result"]


def get_checkpoints(checkpoints_indice: List[int]) -> list:
    checkpoints_ids = [str(i) for i in checkpoints_indice] 
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "sui_getCheckpoint",
        "params": [checkpoints_ids]
    }
    response = post_with_retry(payload)
    return response.json()["result"]


def get_checkpoint(checkpoint_index: int) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "sui_getCheckpoint",
        "params": [str(checkpoint_index)]
    }
    response = post_with_retry(payload)
    return response.json()["result"]


def get_transactions(txs_ids):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "sui_multiGetTransactionBlocks",
        "params": [txs_ids, {
            "showInput": True,
            "showRawInput": True,
            "showEffects": True,
            "showEvents": True,
            "showObjectChanges": True,
            "showBalanceChanges": True,
            "showRawEffects": True
        }]
    }
    response = post_with_retry(payload)
    return response.json()["result"]


def get_transaction(tx_id):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "sui_getTransactionBlock",
        "params": [tx_id, {
            "showInput": True,
            "showRawInput": True,
            "showEffects": True,
            "showEvents": True,
            "showObjectChanges": True,
            "showBalanceChanges": True,
            "showRawEffects": True
        }]
    }
    response = post_with_retry(payload)
    return response.json()["result"]


def trace_all_checkpoints(start=0, end=None, delay=1):
    checkpoint_id = start
    for checkpoint_id in range(start, end):
        try:
            checkpoint = get_checkpoint(checkpoint_id)
            if not checkpoint:
                print(f"Checkpoint {checkpoint_id} not found. Exiting.")
                return
            txs_ids = checkpoint.get("transactions", [])
            print(f"Checkpoint {checkpoint_id} | {len(txs_ids)} transactions")
            txs = get_transactions(txs_ids)
            if txs:
                print(f"Transactions traced.")
            else:
                print(f"Transactions not found. Exiting.")
                return
            with open(PATH_OUTPUT, "ab") as f:
                print(f"Checkpoint {checkpoint_id} trace written to file.")
                entry = (checkpoint, txs)
                pickle.dump(entry, f)
            time.sleep(delay)
            checkpoint_id += 1
            if end is not None and checkpoint_id > end:
                break
        except Exception as e:
            print(f"Error at checkpoint {checkpoint_id}: {e}")
            break


def load_checkpoints():
    with open(PATH_OUTPUT, "rb") as f:
        while True:
            try:
                [checkpoint, txs] = pickle.load(f)
                yield [checkpoint, txs]
            except Exception as e:
                print(e)
                return


def create_read_write_sets(tx):
    write_addrs = set()
    read_addrs = set()
    if 'objectChanges' in tx:
        for change in tx['objectChanges']:
            if change['type'] in {'created', 'mutated', 'unwrapped', 'wrapped', 'deleted', 'unwrappedThenDeleted'}:
                write_addrs.add(change['objectId'])
    if 'effects' in tx:
        effects = tx['effects']
        for mod in effects.get('modifiedAtVersions', []):
            read_addrs.add(mod['objectId'])
        for shared in effects.get('sharedObjects', []):
            read_addrs.add(shared['objectId'])
    read_addrs -= write_addrs
    return read_addrs, write_addrs

def process_trace(id, txs):
    print(f'Plotting {id}...')
    writes: Dict[str, Set[str]] = {}
    reads: Dict[str, Set[str]] = {}
    for tx in txs:
        tx_id = tx['digest']
        tx_reads, tx_writes = create_read_write_sets(tx)
        reads[tx_id] = tx_reads
        writes[tx_id] = tx_writes
    txs_ids = [tx['digest'] for tx in txs]
    G = create_conflict_graph(txs_ids, reads, writes)
    pos = nx.kamada_kawai_layout(G)  # positions for all nodes
    nx.draw(G, pos, edge_color='gray')
    plt.title("Sui Transaction Read/Write Object Access")
    plt.show()
    x = 3

def create_conflict_graph(txs: List[str], reads: Dict[str, Set[str]], writes: Dict[str, Set[str]]) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(txs)
    for tx0_hash, tx0_writes in writes.items():
        for tx1_hash, tx1_reads in reads.items():
            if tx0_hash != tx1_hash:
                if not tx0_writes.isdisjoint(tx1_reads):
                    G.add_edge(tx0_hash, tx1_hash)
        for tx1_hash, tx1_writes in reads.items():
            # checks both tx0_hash != tx1_hash and removes redundent checks with >
            if tx0_hash > tx1_hash:
                if not tx0_writes.isdisjoint(tx1_writes):
                    G.add_edge(tx0_hash, tx1_hash)
    return G


# last = 150497902
# trace_all_checkpoints(start=150000000, end=150000000 + 100)
total = []
agg_size = 100
for i, (chck, txs) in enumerate(load_checkpoints()):
    if i == agg_size:
        break
    total += txs
process_trace("id", total)
