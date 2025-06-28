from datetime import datetime
from typing import Dict, Set, List, Tuple
import matplotlib.patches as mpatches

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from interfaces.interface import Interface
import os
from collections import Counter, defaultdict

USER_KINDS = {
    "ProgrammableTransaction", "TransferObject", "TransferSui",
    "Pay", "PaySui", "PayAllSui", "SplitCoin", "MergeCoin", "Publish"
}

SYSTEM_KINDS = {
    "ConsensusCommitPrologue", "ConsensusCommitPrologueV1", "ConsensusCommitPrologueV3",
    "ChangeEpoch", "Genesis", "RandomnessStateUpdate", 'EndOfEpochTransaction', 'AuthenticatorStateUpdate'
}

KIND_COLORS = {
    "ProgrammableTransaction": "tab:blue",
    "TransferObject": "tab:orange",
    "TransferSui": "tab:cyan",
    "Pay": "tab:olive",
    "PaySui": "tab:purple",
    "PayAllSui": "tab:brown",
    "SplitCoin": "tab:pink",
    "MergeCoin": "tab:gray",
    "Publish": "gold",
    "ConsensusCommitPrologue": "#8b0000",
    "ConsensusCommitPrologueV1": "#d62728",
    "ConsensusCommitPrologueV3": "#ff9896",
    "ChangeEpoch": "navy",
    "Genesis": "black",
    "RandomnessStateUpdate": "tab:green",
    "EndOfEpochTransaction": "darkorange",
    "AuthenticatorStateUpdate": "slateblue",
}


class SuiInterface(Interface):

    def __init__(self):
        rpc_url = os.getenv("SUI_RPC_URL")
        super().__init__(True, rpc_url)

    def _find_checkpoint_by_date(self, target_dt: datetime, search_range=(0, 150000000)) -> int:
        target_ms = int(target_dt.timestamp() * 1000)
        low, high = search_range
        best_checkpoint = -1
        best_diff = float("inf")

        while low <= high:
            mid = (low + high) // 2
            checkpoint = self._fetch_checkpoint(mid)
            if checkpoint is None or "timestampMs" not in checkpoint:
                high = mid - 1
                continue

            ts = int(checkpoint["timestampMs"])
            counttx = len(checkpoint['transactions'])
            diff = abs(ts - target_ms)
            print("Checkpoint", mid, "|", counttx, "TX |", diff / 1000, "SEC OFFSET FROM", target_dt)
            if diff < best_diff:
                best_diff = diff
                best_checkpoint = mid

            if ts < target_ms:
                low = mid + 1
            elif ts > target_ms:
                high = mid - 1
            else:
                return mid

        return best_checkpoint

    def _fetch_checkpoints(
        self,
        cursor: int,
        limit=100,
        descending_order: bool=False
    ) -> dict:
        print(f"Fetching checkpoints {cursor}->{cursor+limit}...")
        params = [str(cursor), limit, descending_order]
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sui_getCheckpoints",
            "params": params
        }
        response = self._post_with_retry(payload)
        return response['data']

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
        durations = [(bucket_edges[i + 1] - bucket_edges[i]).total_seconds() for i in range(n_buckets)]

        # Avoid division by zero
        durations = np.array(durations)
        durations[durations == 0] = 1

        # Operations per second
        txs_per_sec = txs_per_bucket.values / durations
        cks_per_sec = cks_per_bucket.values / durations

        # Midpoint of each bucket
        bucket_midpoints = [bucket_edges[i] + (bucket_edges[i + 1] - bucket_edges[i]) / 2 for i in range(n_buckets)]

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
    
    def get_inputs_kinds_piechart_figure(self, df):
        """
        Build a pie chart of input-kind frequencies with labels shown on the pie.
        """
        # 1. collect raw counts -------------------------------------------------------
        cols = [c for c in df.columns if c.startswith("inputs_")]
        counts = {c: df[c].sum() for c in cols}

        # 2. map / merge categories ---------------------------------------------------
        rename_map = {
            "pure": "Pure",
            "shared_mut": "Shared Mutable",
            "shared_ro": "Shared Immutable",
            "shared_immutable":"Shared Immutable",
            "imm_or_owned": "Immutable/Owned",
        }
        # accumulate into four buckets
        grouped = defaultdict(int)
        for raw_col, cnt in counts.items():
            raw_label = raw_col.removeprefix("inputs_").removesuffix("_count")
            if raw_label == "receiving":  # drop the ‘receiving’ slice entirely
                continue
            label = rename_map.get(raw_label)
            if label:
                grouped[label] += cnt

        # 3. order slices (optional, keeps a stable order) ----------------------------
        order = ["Pure", "Shared Mutable", "Shared Immutable", "Immutable/Owned"]
        labels = [lbl for lbl in order if lbl in grouped]
        sizes = [grouped[lbl] for lbl in labels]

        # 4. plot ---------------------------------------------------------------------
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,  # names on slices
            autopct=lambda p: f"{p:.1f}%" if p >= 1 else "",  # % inside slices
            startangle=90,
            wedgeprops=dict(linewidth=0.8, edgecolor="white"),
            textprops={"size": 11},
        )
        ax.axis("equal")  # keep the pie circular
        plt.setp(autotexts, fontsize=11, weight="bold")
        fig.tight_layout()
        return fig, "piechart_inputs_kinds.png"
    
    def get_txs_kinds_piechart_figure(self, df: pd.DataFrame):
        # 1. aggregate counts ------------------------------------------------------
        cols = [c for c in df.columns if c.startswith('kind_') and c.endswith('_count')]
        counts = {c: df[c].sum() for c in cols}
        total = sum(counts.values())

        items = sorted(
            ((c.replace('kind_', '').replace('_count', ''), v) for c, v in counts.items()),
            key=lambda x: x[1],
            reverse=True,
        )

        # 2. collapse <1 % into "other" -------------------------------------------
        labels, sizes, other_sum = [], [], 0
        for lbl, cnt in items:
            if cnt / total < 0.01:
                other_sum += cnt
            else:
                labels.append(lbl)
                sizes.append(cnt)
        if other_sum:
            labels.append("other")
            sizes.append(other_sum)
            
        colors = [
            KIND_COLORS.get(lbl, "lightgrey")  # graceful fallback for “other” or unknown kinds
            for lbl in labels
        ]
        
        # 4. plot ------------------------------------------------------------------
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            sizes,
            colors=colors,
            startangle=140,
            autopct=lambda p: f"{p:.1f}%" if p >= 1 else "",
        )
        ax.axis("equal")
        plt.setp(autotexts, fontsize=12, weight="bold")  # ← NEW LINE
        return fig, "piechart_txs_kinds.png"
    
    def get_error_kinds_piechart_figure(
    self,
    df: pd.DataFrame,
    min_share: float=0.01,
    ):
        cols = [c for c in df.columns if c.startswith("failed_")]
        counts = {c: df[c].sum() for c in cols}
        total = sum(counts.values()) or 1  # avoid ZeroDivisionError on empty

        grouped = defaultdict(int)
        other_sum = 0
        for raw_col, cnt in counts.items():
            raw_label = raw_col.removeprefix("failed_")
            if cnt / total < min_share:
                other_sum += cnt
            else:
                grouped[raw_label] += cnt
        if other_sum:
            grouped["other"] += other_sum

        order = cols + ["other"]
        labels = [
            lbl.removeprefix("failed_")
            for lbl in order
            if lbl.removeprefix("failed_") in grouped
        ]
        sizes = [grouped[lbl] for lbl in labels]

        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct=lambda p: f"{p:.1f}%" if p >= 1 else "",
            startangle=90,
            wedgeprops=dict(linewidth=0.8, edgecolor="white"),
            textprops={"size": 11},
        )
        ax.axis("equal")
        plt.setp(autotexts, fontsize=11, weight="bold")
        fig.tight_layout()
        return fig, "piechart_error_kinds.png"
    
    def get_writes_kinds_piechart_figure(self, df, min_share: float=0.01):
        cols = [c for c in df.columns if c.startswith("writes_")]
        counts = {c: df[c].sum() for c in cols}

        grouped = defaultdict(int)
        total = sum(counts.values())

        # 2. collapse tiny slices and drop “receiving” --------------------------
        other_sum = 0
        for raw_col, cnt in counts.items():
            raw_label = raw_col.removeprefix("writes_")
            if raw_label == "receiving":  # skip this one entirely
                continue
            if cnt / total < min_share:
                other_sum += cnt  # fold into “other”
            else:
                grouped[raw_label] += cnt

        if other_sum:  # append the “other” slice last
            grouped["other"] += other_sum

        # 3. keep a stable ordering --------------------------------------------
        order = cols + ["other"]  # ensures “other” is last if present
        labels = [lbl.removeprefix("writes_") for lbl in order if lbl.removeprefix("writes_") in grouped]
        sizes = [grouped[lbl] for lbl in labels]

        # 4. plot ---------------------------------------------------------------
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct=lambda p: f"{p:.1f}%" if p >= 1 else "",
            startangle=90,
            wedgeprops=dict(linewidth=0.8, edgecolor="white"),
            textprops={"size": 11},
        )
        ax.axis("equal")
        plt.setp(autotexts, fontsize=11, weight="bold")
        fig.tight_layout()
        return fig, "piechart_writes_kinds.png"

    def get_additional_figures(self, df: pd.DataFrame) -> List[Tuple[str, Figure]]:
        df['datetime'] = pd.to_datetime(df['timestampMs'], unit='ms', utc=True)

        return [
            self.get_error_kinds_piechart_figure(df),
            self.get_writes_kinds_piechart_figure(df),
            self.get_inputs_kinds_piechart_figure(df),
            self.get_txs_kinds_piechart_figure(df),
            self.get_ops_hour_figure(df),
            self.get_ops_datetime_figure(df, 16),
        ]
    
    def _get_failed_txs(self, txs_traces: List[Dict]) -> Dict[str, int]:
        ERROR_KINDS = [
            "InsufficientGas",
            "InvalidGasObject",
            "InvariantViolation",
            "FeatureNotYetSupported",
            "MoveObjectTooBig",
            "MovePackageTooBig",
            "CircularObjectOwnership",
            "InsufficientCoinBalance",
            "CoinBalanceOverflow",
            "PublishErrorNonZeroAddress",
            "SuiMoveVerificationError",
            "MovePrimitiveRuntimeError",
            "MoveAbort",
            "VMVerificationOrDeserializationError",
            "VMInvariantViolation",
            "FunctionNotFound",
            "ArityMismatch",
            "TypeArityMismatch",
            "NonEntryFunctionInvoked",
            "CommandArgumentError",
            "TypeArgumentError",
            "UnusedValueWithoutDrop",
            "InvalidPublicFunctionReturnType",
            "InvalidTransferObject",
            "EffectsTooLarge",
            "PublishUpgradeMissingDependency",
            "PublishUpgradeDependencyDowngrade",
            "PackageUpgradeError",
            "WrittenObjectsTooLarge",
            "CertificateDenied",
            "SuiMoveVerificationTimedout",
            "SharedObjectOperationNotAllowed",
            "InputObjectDeleted",
            "ExecutionCancelledDueToSharedObjectCongestion",
            "AddressDeniedForCoin",
            "CoinTypeGlobalPause",
            "ExecutionCancelledDueToRandomnessUnavailable",
            "MoveVectorElemTooBig",
            "MoveRawValueTooBig",
            "InvalidLinkage",
        ]   
        stats = Counter({f"failed_{k}": 0 for k in ERROR_KINDS})
        stats["failed_unknown"] = 0
        for tx in txs_traces:
            status = tx.get("effects", {}).get("status", {})
            if status.get("status", "").lower() == "success":
                continue
            error_msg = status.get("error", "")
            matched = next(
                (k for k in ERROR_KINDS if k in error_msg),
                None,
            )
            if matched:
                stats[f"failed_{matched}"] += 1
            else:
                stats["failed_unknown"] += 1

        return stats
    
    def _get_writes(self, txs_traces: List[Dict]) -> Tuple[Counter, Dict[str, Set[str]]]:
        SUI_SINGLETONS: Set[int] = {
            0x5, 0x6, 0x7, 0x8, 0x9, 0x403, 0xacc,
        }
        
        stats = Counter(
            writes_owned=0,
            writes_mutable_shared=0,
            writes_system=0,
            writes_gas=0,
            writes_created=0,
            writes_published=0,
            writes_wrapped=0,
            writes_unwrapped=0,
            writes_deleted=0,
            writes_unknown=0,
        )
        sets = defaultdict(set)

        for tx in txs_traces:
            changes = tx.get("objectChanges", [])
            tx_data = tx.get("transaction", {}).get("data", {})
            inputs = tx_data.get("transaction", {}).get("inputs", [])
            gas_ids = {p["objectId"] for p in tx_data.get("gasData", {})
                                                .get("payment", [])}

            imm_or_owned_ids = {
                inp["objectId"]
                for inp in inputs
                if inp.get("type") == "object"
                and inp.get("objectType") == "immOrOwnedObject"
            }
            mut_shared_ids = {
                inp["objectId"]
                for inp in inputs
                if inp.get("type") == "object"
                and inp.get("objectType") == "sharedObject"
                and inp.get("mutable", False)
            }

            for ch in changes:
                oid = ch["objectId"]
                ctype = ch["type"]

                if ctype == "created":
                    cat = "writes_created"
                elif ctype == "published":
                    cat = "writes_published"
                elif ctype == "wrapped":
                    cat = "writes_wrapped"
                elif ctype == "unwrapped":
                    cat = "writes_unwrapped"
                elif ctype == "deleted":
                    cat = "writes_deleted"

                elif ctype == "mutated":
                    if oid in gas_ids:
                        cat = "writes_gas"
                    elif int(oid, 16) in SUI_SINGLETONS:
                        cat = "writes_system"
                    elif oid in mut_shared_ids:
                        cat = "writes_mutable_shared"
                    elif oid in imm_or_owned_ids:
                        cat = "writes_owned"
                    else:
                        cat = "writes_unknown"
                else:
                    cat = "writes_unknown"

                stats[cat] += 1
                sets[cat].add(oid)

        return stats
     
    def _get_inputs_types(self, txs_traces: List[Dict]) -> Dict[str, int]:
        stats = Counter(
            inputs_pure=0,
            inputs_imm_or_owned=0,
            inputs_shared_mut=0,
            inputs_shared_ro=0,
            inputs_receiving=0,
        )
        for tx in txs_traces:
            tx_data = tx.get("transaction", {}).get("data", {})
            inputs = tx_data.get("transaction", {}).get("inputs", [])
            for inp in inputs:
                inp_type = inp.get("type") 
                if inp_type == 'object':
                    object_type = inp.get("objectType")
                    if object_type == 'sharedObject':
                        is_mutable = inp.get('mutable', False)
                        if is_mutable:
                            stats['inputs_shared_mut'] += 1
                        else:
                            stats['inputs_shared_ro'] += 1
                    elif object_type == 'immOrOwnedObject':
                        stats['inputs_imm_or_owned'] += 1
                    elif object_type == 'receiving':
                        stats['inputs_receiving'] += 1
                    else:
                        print(f'Unknown object type {object_type}')
                        exit()
                elif inp_type == 'pure':
                    stats['inputs_pure'] += 1
                else:
                    print(f'Unknown input type {inp_type}')
                    exit()
        return stats
     
    '''def _get_inputs_types(self, txs_traces: List[Dict]) -> Dict[str, int]:
        stats = Counter(
            inputs_pure         = 0,
            inputs_imm_or_owned = 0,
            inputs_shared_mut   = 0,
            inputs_shared_ro    = 0,
            inputs_receiving    = 0,
        )
        for tx in txs_traces:
            tx_data = tx.get("transaction", {}).get("data", {})
            inputs = tx_data.get("transaction", {}).get("inputs", [])
            for inp in inputs:
                inp_type = inp.get("type") 
                if inp_type == 'object':
                    object_type = inp.get("objectType")
                    if object_type == 'sharedObject':
                        is_mutable = inp.get('mutable', False)
                        if is_mutable:
                            stats['inputs_shared_mut'] += 1
                        else:
                            stats['inputs_shared_ro'] += 1
                    elif object_type == 'immOrOwnedObject':
                        stats['inputs_imm_or_owned'] += 1
                    elif object_type == 'receiving':
                        stats['inputs_receiving'] += 1
                    else:
                        print(f'Unknown object type {object_type}')
                        exit()
                elif inp_type == 'pure':
                    stats['inputs_pure'] += 1
                else:
                    print(f'Unknown input type {inp_type}')
                    exit()
        return stats'''
        
    def get_additional_metrics(self, block_number, trace) -> Dict[str, float]:
        checkpoint_trace, txs_traces = trace

        total_sui_transfered = 0
        for tx in txs_traces:
            tx_total_transfered = [abs(int(e['amount'])) for e in tx['balanceChanges'] if e['coinType'] == '0x2::sui::SUI']
            total_sui_transfered += sum(tx_total_transfered)
        
        txs_types = [self._get_tx_type(tx) for tx in txs_traces]
        txs_kind_counter = Counter(txs_types)
    
        all_kinds = USER_KINDS | SYSTEM_KINDS
        
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
        additional_metrics.update(self._get_inputs_types(txs_traces))
        additional_metrics.update(self._get_writes(txs_traces))
        additional_metrics.update(self._get_failed_txs(txs_traces))
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
