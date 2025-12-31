import os
import random
from typing import Dict, Set, List, Tuple, Any

from interfaces.interface import Interface
REQ_ID = int(random.uniform(1, 100000))
DIR_PATH=None
class SolanaInterface(Interface):
    
    def __init__(self):
        rpc_url = os.getenv("SOL_RPC_URL")
        super().__init__(False, rpc_url)
        # Ignore common sysvars and program IDs that every tx touches
        self._ignore_accounts: Set[str] = {
            "SysvarC1ock11111111111111111111111111111111",
            "SysvarEpochSchedu1e111111111111111111111111",
            "SysvarFees111111111111111111111111111111111",
            "SysvarRent111111111111111111111111111111111",
            "SysvarRecentB1ockHashes11111111111111111111",
            "SysvarS1otHashes111111111111111111111111111",
            "SysvarS1otHistory11111111111111111111111111",
            "SysvarStakeHistory1111111111111111111111111",
            "Sysvar1nstructions1111111111111111111111111",
            "ComputeBudget111111111111111111111111111111",
            "AddressLookupTab1e1111111111111111111111111",
        }

    def get_additional_metrics(self, block_number, trace) -> Dict[str, float]:
        txs_count = len(trace[0].get("transactions", []))
        return {"block_number": block_number, "txs": txs_count}
        

    def fetch(self, slot: int) -> Tuple[int, dict]:
        global REQ_ID,DIR_PATH
        REQ_ID +=1
        payload = {
            "jsonrpc": "2.0",
            "id": REQ_ID,
            "method": "getBlock",
            "params": [
                slot,
                {
                    "encoding": "jsonParsed",
                    "transactionDetails": "full",
                    "rewards": False,
                    "maxSupportedTransactionVersion": 0
                },
            ],
        }
        resp = self._post_with_retry(payload,pathname=f"{DIR_PATH}/sol_{slot}.json")
        return slot, resp

    def _parse_tx(self, tx_entry: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
        """
        Returns (read_set, write_set) for one getBlock transaction entry.
        Safe and spec-accurate:
        - Legacy JSON: writability derived from header + account order.
        - Versioned JSON: append meta.loadedAddresses.{writable,readonly} to the static keys.
        - JSON Parsed: message.accountKeys already carries {pubkey, writable, source}; use as-is.
        No heuristics based on balances or token owners are used.
        """
        message: Dict[str, Any] = (tx_entry.get("transaction") or {}).get("message") or {}
        meta: Dict[str, Any] = tx_entry.get("meta") or {}
        acc_keys = message.get("accountKeys") or []

        # Optional: remove known-irrelevant accounts later (e.g., sysvars) via self._ignore_accounts
        ignore: Set[str] = getattr(self, "_ignore_accounts", set())

        # Branch 1: JSON PARSED (accountKeys is list of dicts with pubkey/writable)
        if acc_keys and isinstance(acc_keys[0], dict) and "pubkey" in acc_keys[0]:
            resolved_keys: List[str] = []
            writable_idx: Set[int] = set()
            for i, ak in enumerate(acc_keys):
                pk = ak.get("pubkey")
                if not pk:
                    continue
                resolved_keys.append(pk)
                if ak.get("writable"):
                    writable_idx.add(i)
            # In jsonParsed, loaded addresses are already represented in accountKeys with source="lookupTable"
            write_addrs = {resolved_keys[i] for i in writable_idx} - ignore
            read_addrs = set(resolved_keys) - write_addrs - ignore
            return read_addrs, write_addrs

        # Branch 2: RAW JSON (legacy or v0) – accountKeys is list[str]
        # Build resolved key list and writable set of indices
        static_keys: List[str] = [str(k) for k in acc_keys]
        hdr = message.get("header") or {}

        n_sign = int(hdr.get("numRequiredSignatures", 0))
        n_ro_sign = int(hdr.get("numReadonlySignedAccounts", 0))
        n_ro_unsign = int(hdr.get("numReadonlyUnsignedAccounts", 0))

        resolved_keys: List[str] = static_keys[:]          # start with static keys
        writable_idx: Set[int] = set()

        # Writable signed accounts come first among the signers
        # [0 .. n_sign-1] are signers; last n_ro_sign of those are readonly
        for i in range(max(0, n_sign - n_ro_sign)):
            if i < len(resolved_keys):
                writable_idx.add(i)

        # Unsigned accounts follow; last n_ro_unsign of those are readonly
        n_unsigned = max(0, len(static_keys) - n_sign)
        n_rw_unsign = max(0, n_unsigned - n_ro_unsign)
        for i in range(n_sign, min(len(resolved_keys), n_sign + n_rw_unsign)):
            writable_idx.add(i)

        # If this is a v0 transaction, extend with loaded addresses from Address Lookup Tables
        # Order is: loadedAddresses.writable then loadedAddresses.readonly
        loaded = meta.get("loadedAddresses") or {}
        loaded_w = loaded.get("writable") or []
        loaded_r = loaded.get("readonly") or []

        base = len(resolved_keys)
        resolved_keys.extend(loaded_w)
        for j in range(len(loaded_w)):
            writable_idx.add(base + j)

        resolved_keys.extend(loaded_r)
        # readonly loaded addresses: no additions to writable_idx

        write_addrs = {resolved_keys[i] for i in writable_idx} - ignore
        read_addrs = set(resolved_keys) - write_addrs - ignore
        return read_addrs, write_addrs

    def get_conflict_graph(self, data: dict) -> Any:
        block = data[0]
        tx_entries: List[dict] = block.get("transactions", [])

        writes: Dict[str, Set[str]] = {}
        reads: Dict[str, Set[str]] = {}
        txs: List[str] = []

        for tx_entry in tx_entries:
            sigs = tx_entry.get("transaction", {}).get("signatures", [])
            if not sigs:
                continue
            tx_id = sigs[0]
            read_addrs, write_addrs = self._parse_tx(tx_entry)

            if read_addrs:
                reads[tx_id] = read_addrs
            if write_addrs:
                writes[tx_id] = write_addrs
            txs.append(tx_id)

        return self._create_conflict_graph_from_readset_writeset(txs, reads, writes)