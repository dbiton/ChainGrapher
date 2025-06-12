from typing import Dict, Set, List
from interfaces.interface import Interface
import os
from collections import Counter

# This is for IOTA Rebased (after switching from tangle to MoveVM)
class IotaInterface(Interface):

    def __init__(self):
        rpc_url = os.getenv("IOTA_RPC_URL")
        super().__init__(True, rpc_url)

    def _fetch_checkpoint(self, checkpoint_id: int) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "iota_getCheckpoint",
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
                "method": "iota_multiGetTransactionBlocks",
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
    
    def get_additional_metrics(self, block_number, checkpoint_trace) -> Dict[str, float]:
        checkpoint_trace, txs_traces = checkpoint_trace

        txs_types = [self._get_tx_type(tx) for tx in txs_traces]
        txs_type_counter = Counter(txs_types)

        user_kinds = {
            "ProgrammableTransaction", "TransferObject", "TransferSui",
            "Pay", "PaySui", "PayAllSui", "SplitCoin", "MergeCoin", "Publish"
        }

        system_kinds = {
            "ConsensusCommitPrologue", "ConsensusCommitPrologueV1",
            "ChangeEpoch", "Genesis", "RandomnessStateUpdate"
        }

        return {
            "user_tx_count": sum(txs_type_counter.get(k, 0) for k in user_kinds),
            "system_tx_count": sum(txs_type_counter.get(k, 0) for k in system_kinds),
            "block_number": block_number,
            "txs": len(txs_traces)
        }
    
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
