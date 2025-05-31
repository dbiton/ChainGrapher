from typing import Any, List, Dict, Set
import httpx
import time
import json
from matplotlib import pyplot as plt
import networkx as nx
import pickle
from graph_metrics import get_graph_metrics
from plotters import plot_data
import csv


PATH_OUTPUT = 'sui_trace_150M[+100+1100].pkl'


def load_checkpoints():
    with open(PATH_OUTPUT, "rb") as f:
        while True:
            try:
                [checkpoint, txs] = pickle.load(f)
                yield [checkpoint, txs]
            except Exception as e:
                print(e)
                return





def process_trace(id, txs):
    print(f'Plotting {id}...')
    writes: Dict[str, Set[str]] = {}
    reads: Dict[str, Set[str]] = {}
    for _, tx in txs:
        tx_id = tx['digest']
        tx_reads, tx_writes = create_read_write_sets(tx)
        reads[tx_id] = tx_reads
        writes[tx_id] = tx_writes
    txs_ids = [tx['digest'] for (_, tx) in txs]
    txs_groups = [cid for (cid, _) in txs]
    G = create_conflict_graph(txs_ids, reads, writes)
    tx_id_to_group = {tx_id: group for tx_id, group in zip(txs_ids, txs_groups)}
    unique_groups = sorted(set(txs_groups))
    group_to_color = {group: i for i, group in enumerate(unique_groups)}
    node_colors = [group_to_color[tx_id_to_group[n]] for n in G.nodes()]
    pos = nx.kamada_kawai_layout(G)  # positions for all nodes
    nx.draw(
        G,
        pos,
        edge_color='gray',
        node_color=node_colors,
        cmap=plt.cm.tab20,  # Use a categorical colormap
    )
    #plt.title("Sui Transaction Read/Write Object Access")
    #plt.show()
    return G

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


def write_data(data, output_path):
    write_header = True
    with open(output_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        for result in data:
            if write_header:
                write_header = False
                writer.writerow(result.keys())
            writer.writerow(result.values())

# last = 150497902
# trace_all_checkpoints(start=150000000 + 100, end=150000000 + 1100)

output_path = "output.csv"
data = []
agg_txs = []
agg_size = 1
for i, (chck, txs) in enumerate(load_checkpoints()):
    print(i)
    chck_id = chck['sequenceNumber']
    if i % agg_size == 0 and i > 0:
        G = process_trace("id", agg_txs)
        metrics = get_graph_metrics(G, {"block_number": chck_id, "txs": len(agg_txs)})
        data.append(metrics)
        print(metrics)
        agg_txs = []
    agg_txs += [[chck_id, tx] for tx in txs]

write_data(data, output_path)
plot_data(output_path)