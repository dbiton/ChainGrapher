from typing import Dict, Set, List, Tuple

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from interfaces.interface import Interface
import os
from collections import Counter

USER_KINDS = {
    "ProgrammableTransaction", "TransferObject", "TransferSui",
    "Pay", "PaySui", "PayAllSui", "SplitCoin", "MergeCoin", "Publish"
}

SYSTEM_KINDS = {
    "ConsensusCommitPrologue", "ConsensusCommitPrologueV1", "ConsensusCommitPrologueV3",
    "ChangeEpoch", "Genesis", "RandomnessStateUpdate", 'EndOfEpochTransaction', 'AuthenticatorStateUpdate'
}

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

    def get_ops_datetime_figure(self, df: pd.DataFrame, n_buckets: int) -> Figure:
        df_sorted = df.sort_values('datetime')

        # Define time range and bucket edges
        start = df_sorted['datetime'].min()
        end = df_sorted['datetime'].max()
        bucket_edges = pd.date_range(start=start, end=end, periods=n_buckets + 1)

        # Assign each row to a bucket based on datetime
        df_sorted['bucket'] = pd.cut(df_sorted['datetime'], bins=bucket_edges, labels=False, include_lowest=True)

        # Group by bucket
        grouped = df_sorted.groupby('bucket')

        # Total transactions per bucket
        txs_per_bucket = grouped['txs'].sum()
        cks_per_bucket = grouped['txs'].count()

        # Duration per bucket in seconds (should be equal for all, but compute for accuracy)
        durations = [(bucket_edges[i+1] - bucket_edges[i]).total_seconds() for i in range(n_buckets)]

        # Avoid division by zero
        durations = np.array(durations)
        durations[durations == 0] = 1

        # Operations per second
        txs_per_sec = txs_per_bucket.values / durations
        cks_per_sec = cks_per_bucket.values / durations

        # Midpoint of each bucket
        bucket_midpoints = [bucket_edges[i] + (bucket_edges[i+1] - bucket_edges[i]) / 2 for i in range(n_buckets)]

        # Plot
        fig, ax = plt.subplots()
        ax.plot(bucket_midpoints, txs_per_sec, marker='o', label='txs')
        ax.plot(bucket_midpoints, cks_per_sec, marker='x', label='checkpoints')
        ax.set_title(f"Operations per Second Over Time ({n_buckets} Buckets)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Operations per Second")
        fig.autofmt_xdate()

        return fig, "ops_vs_datetime.png"

    def get_ops_hour_figure(self, df: pd.DataFrame) -> Figure:
        df = df.copy()
        df['hour'] = df['datetime'].dt.hour
        df['date'] = df['datetime'].dt.date

        # Total transactions per hour of day
        txs_per_hour = df.groupby('hour')['txs'].sum()

        # Count how many times each hour appears (i.e., how many distinct days had data at that hour)
        active_days_per_hour = df.groupby('hour')['date'].nunique()

        # Calculate ops/sec: total txs / (number of occurrences * 3600 seconds)
        seconds_per_hour = 3600
        ops_per_sec = txs_per_hour / (active_days_per_hour * seconds_per_hour)

        # Fill in missing hours (0–23) with 0s
        ops_per_sec = ops_per_sec.reindex(range(24), fill_value=0)

        # Plot
        fig, ax = plt.subplots()
        ax.bar(ops_per_sec.index, ops_per_sec.values, width=0.8)
        ax.set_xticks(range(24))
        ax.set_xlabel("Hour of Day (UTC)")
        ax.set_ylabel("Operations per Second")
        ax.set_title("Average Operations per Second by Hour of Day")
        return fig, "ops_vs_hour_of_day.png"
    
    def get_txs_kinds_piechart_figure(self, df: pd.DataFrame) -> Figure:
        txs_kinds_cols = [col for col in df.columns if col.startswith('kind_') and col.endswith('_count')]
        txs_kinds_total_counts = {col: df[col].sum() for col in txs_kinds_cols}
        sorted_items = sorted(txs_kinds_total_counts.items(), key=lambda x: x[1], reverse=True)
        labels, sizes = zip(*sorted_items)
        labels = [label.replace('kind_', '').replace('_count', '') for label in labels]
        colors = []
        for label in labels:
            if label in USER_KINDS:
                colors.append(plt.cm.Reds(0.4 + 0.6 * labels.index(label) / len(labels)))
            elif label in SYSTEM_KINDS:
                colors.append(plt.cm.Blues(0.4 + 0.6 * labels.index(label) / len(labels)))
            else:
                colors.append("gray")
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        ax.set_title("Transaction Kind Distribution (Ordered by Count)")
        ax.axis('equal')
        return fig, "piechart_txs_kinds.png"
        
    
    def get_additional_figures(self, df: pd.DataFrame) -> List[Tuple[str, Figure]]:
        df['datetime'] = pd.to_datetime(df['timestampMs'], unit='ms', utc=True)

        return [
            self.get_txs_kinds_piechart_figure(df), 
            self.get_ops_hour_figure(df), 
            self.get_ops_datetime_figure(df, 32)
        ]
    
        
    def get_additional_metrics(self, block_number, trace) -> Dict[str, float]:
        checkpoint_trace, txs_traces = trace

        total_sui_transfered = 0
        for tx in txs_traces:
            tx_total_transfered = [abs(int(e['amount'])) for e in tx['balanceChanges'] if e['coinType'] == '0x2::sui::SUI']
            total_sui_transfered += sum(tx_total_transfered)
        
        txs_types = [self._get_tx_type(tx) for tx in txs_traces]
        txs_kind_counter = Counter(txs_types)
    
        all_kinds = USER_KINDS | SYSTEM_KINDS
        unknown_kinds = set(txs_kind_counter.keys()).difference(all_kinds)
        if len(unknown_kinds) > 0:
            with open("log.txt", "a") as f:
                f.writelines([f"unknown kinds {unknown_kinds}"])
        
        timestamp = checkpoint_trace['timestampMs']
        epoch = checkpoint_trace['epoch']
        digest = checkpoint_trace['digest']
        additional_metrics = {
            "digest": digest,
            "epoch": epoch,
            "user_tx_count": sum(txs_kind_counter.get(k, 0) for k in USER_KINDS),
            "system_tx_count": sum(txs_kind_counter.get(k, 0) for k in SYSTEM_KINDS),
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