from typing import Dict, Set, List, Tuple

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from interfaces.interface import Interface
import os
from collections import Counter

class SuiInterface(Interface):

    def __init__(self):
        rpc_url = os.getenv("SUI_RPC_URL")
        super().__init__(True, rpc_url)

    def _fetch_checkpoint(self, checkpoint_id: int) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sui_getCheckpoint",
            "params": [str(checkpoint_id)]
        }
        response = self._post_with_retry(payload)
        return response

    def _fetch_txs(self, txs_ids: List[str]) -> List[dict]:
        all_results = []
        for i in range(0, len(txs_ids), 50):
            batch = txs_ids[i:i + 50]
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sui_multiGetTransactionBlocks",
                "params": [
                    batch,
                    {
                        "showInput": True,
                        "showRawInput": True,
                        "showEffects": True,
                        "showEvents": True,
                        "showObjectChanges": True,
                        "showBalanceChanges": True,
                        "showRawEffects": True
                    }
                ]
            }
            batch_results = self._post_with_retry(payload)
            all_results.extend(batch_results)
        return all_results
    
    def fetch(self, checkpoint_id):
        checkpoint = self._fetch_checkpoint(checkpoint_id)
        if not checkpoint:
            print(f"Checkpoint {checkpoint_id} | Checkpoint not found. Exiting.")
            exit()
        txs_ids = checkpoint.get("transactions", [])
        print(f"Checkpoint {checkpoint_id} | {len(txs_ids)} transactions")
        txs = self._fetch_txs(txs_ids)
        if not txs:
            print(f"Checkpoint {checkpoint_id} | Transactions not found. Exiting.")
            exit()
        return checkpoint_id, checkpoint, txs
    
    def get_conflict_graph(self, checkpoint_trace):
        checkpoint_trace, txs_traces = checkpoint_trace
        writes: Dict[str, Set[str]] = {}
        reads: Dict[str, Set[str]] = {}
        for tx_trace in txs_traces:
            tx_id = tx_trace['digest']
            tx_reads, tx_writes = self._parse_tx(tx_trace)
            reads[tx_id] = tx_reads
            writes[tx_id] = tx_writes
        txs = [tx_trace["digest"] for tx_trace in txs_traces]
        return self._create_conflict_graph_from_readset_writeset(txs, reads, writes)

    def _get_tx_type(self, tx):
        return tx['transaction']['data']['transaction']['kind']

    def get_tx_count_over_time_figure(self, df: pd.DataFrame, n_buckets: int) -> Figure:
        df['datetime'] = pd.to_datetime(df['timestampMs'], unit='ms', utc=True)
        df_sorted = df.sort_values('datetime')
        df_sorted['bucket'] = pd.qcut(df_sorted['datetime'], q=n_buckets, labels=False)
        txs_per_bucket = df_sorted.groupby('bucket').size()
        bucket_midpoints = df_sorted.groupby('bucket')['datetime'].apply(lambda x: x.iloc[len(x)//2])
        fig, ax = plt.subplots()
        ax.plot(bucket_midpoints, txs_per_bucket, marker='o')
        ax.set_title(f"Transaction Count Over Time ({n_buckets} Buckets)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Number of Transactions")
        fig.autofmt_xdate()
        return fig, "tx_count_over_time.png"

    def get_tx_count_by_hour_of_day_figure(self, df: pd.DataFrame) -> Figure:
        df['hour'] = df['datetime'].dt.hour
        txs_by_hour = df['hour'].value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.bar(txs_by_hour.index, txs_by_hour.values, width=0.8)
        ax.set_xticks(range(24))
        ax.set_xlabel("Hour of Day (UTC)")
        ax.set_ylabel("Number of Transactions")
        ax.set_title("Transaction Count by Hour of Day")
        return fig, "tx_count_by_hour_of_day.png"
    
    def get_additional_figures(self, df: pd.DataFrame) -> List[Tuple[str, Figure]]:
        txs_kinds_cols = [col for col in df.columns if col.startswith('kind_') and col.endswith('_count')]
        txs_kinds_total_counts = {col: df[col].sum() for col in txs_kinds_cols}
        
        sorted_items = sorted(txs_kinds_total_counts.items(), key=lambda x: x[1], reverse=True)
        labels, sizes = zip(*sorted_items)
        labels = [v.split('_')[1] for v in txs_kinds_total_counts.keys()]

        fig_pie, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title("Transaction Kind Distribution (Ordered by Count)")
        ax.axis('equal')  # Equal aspect ratio ensures the pie is circular

        df['datetime'] = pd.to_datetime(df['timestampMs'], unit='ms', utc=True)

        
        return [
            (fig_pie, "piechart.png"), 
            self.get_tx_count_by_hour_of_day_figure(df), 
            self.get_tx_count_over_time_figure(df, 12)
        ]
    
        
    def get_additional_metrics(self, block_number, trace) -> Dict[str, float]:
        checkpoint_trace, txs_traces = trace

        total_sui_transfered = 0
        for tx in txs_traces:
            tx_total_transfered = [abs(int(e['amount'])) for e in tx['balanceChanges'] if e['coinType'] == '0x2::sui::SUI']
            total_sui_transfered += sum(tx_total_transfered)
        
        txs_types = [self._get_tx_type(tx) for tx in txs_traces]
        txs_kind_counter = Counter(txs_types)
        
        user_kinds = {
            "ProgrammableTransaction", "TransferObject", "TransferSui",
            "Pay", "PaySui", "PayAllSui", "SplitCoin", "MergeCoin", "Publish"
        }

        system_kinds = {
            "ConsensusCommitPrologue", "ConsensusCommitPrologueV1", "ConsensusCommitPrologueV3",
            "ChangeEpoch", "Genesis", "RandomnessStateUpdate"
        }

        all_kinds = user_kinds | system_kinds
        unknown_kinds = set(txs_kind_counter.keys()).difference(all_kinds)
        if len(unknown_kinds) > 0:
            print(f"unknown kinds {unknown_kinds}")
            exit()
        
        timestamp = checkpoint_trace['timestampMs']
        epoch = checkpoint_trace['epoch']
        digest = checkpoint_trace['digest']
        additional_metrics = {
            "digest": digest,
            "epoch": epoch,
            "user_tx_count": sum(txs_kind_counter.get(k, 0) for k in user_kinds),
            "system_tx_count": sum(txs_kind_counter.get(k, 0) for k in system_kinds),
            "block_number": block_number,
            "timestampMs": timestamp,
            "total_sui_transfered": total_sui_transfered,
            "txs": len(txs_traces)
        }
        for kind in all_kinds:
            additional_metrics[f"kind_{kind}_count"] = txs_kind_counter.get(kind, 0)
        return additional_metrics
    
    def _parse_tx(self, tx):
        write_addrs = {
            change['objectId']
            for change in tx.get('objectChanges', [])
            if 'objectId' in change
        }
        shared_addrs = {
            obj['objectId']
            for obj in tx.get('effects', {}).get('sharedObjects', [])
        }
        inputs = (
            tx.get('transaction', {})
            .get('data', {})
            .get('transaction', {})
            .get('inputs', [])
        )
        inputs_addrs = {
            inp['objectId']
            for inp in inputs
            if inp.get('type') == 'object'
        }
        read_addrs = (inputs_addrs | shared_addrs) - write_addrs
        return read_addrs, write_addrs