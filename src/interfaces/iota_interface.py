from typing import Dict, Set, List
from interfaces.interface import Interface
import os

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

    def get_additional_metrics(self, block_number, checkpoint_trace) -> Dict[str, float]:
        checkpoint_trace, txs_traces = checkpoint_trace
        return {
            "block_number": block_number, 
            "txs": len(txs_traces)
        }
    
    def _parse_tx(self, tx_trace):
        write_addrs = {
            change['objectId']
            for change in tx_trace.get('objectChanges', [])
            if 'objectId' in change
        }

        effects = tx_trace.get('effects', {})
        pure_read_addrs = {
            obj['objectId']
            for obj in effects.get('sharedObjects', [])
        }

        inputs = (
            tx_trace.get('transaction', {})
            .get('data', {})
            .get('transaction', {})
            .get('inputs', [])
        )

        potential_reads = {
            inp['objectId']
            for inp in inputs
            if inp.get('type') == 'object'
            and inp.get('mutable', False)
            and inp['objectId'] not in write_addrs
        }

        read_addrs = pure_read_addrs | potential_reads
        read_addrs = read_addrs.difference(write_addrs)
        return read_addrs, write_addrs
