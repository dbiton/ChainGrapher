from typing import Any, List
import os
from dotenv import load_dotenv
import httpx
import time

load_dotenv()

SUI_RPC_URL = os.getenv("SUI_RPC_URL")

def fetcher_sui(checkpoint_number: int):
    delay = .33
    checkpoint = get_checkpoint(checkpoint_number)
    if not checkpoint:
        print(f"Checkpoint {checkpoint_number} | Checkpoint not found. Exiting.")
        exit()
    txs_ids = checkpoint.get("transactions", [])
    print(f"Checkpoint {checkpoint_number} | {len(txs_ids)} transactions")
    time.sleep(delay)
    txs = get_transactions(txs_ids)
    time.sleep(delay)
    if not txs:
        print(f"Checkpoint {checkpoint_number} | Transactions not found. Exiting.")
        exit()
    return checkpoint_number, checkpoint, txs

def post_with_retry(payload: Any, max_retries: int=5, base_delay: float=2.0) -> httpx.Response:
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
            return response.json()["result"]
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            print(f"[Attempt {attempt}] Error: {e}")
            if attempt == max_retries:
                print("Max retries reached. Giving up.")
                raise
            time.sleep(delay)
            delay *= 2


def get_latest_checkpoint():
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "sui_getLatestCheckpointSequenceNumber",
        "params": []
    }
    response = post_with_retry(payload)
    return response


def get_checkpoints(checkpoints_indice: List[int]) -> list:
    checkpoints_ids = [str(i) for i in checkpoints_indice] 
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "sui_getCheckpoint",
        "params": [checkpoints_ids]
    }
    response = post_with_retry(payload)
    return response


def get_checkpoint(checkpoint_index: int) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "sui_getCheckpoint",
        "params": [str(checkpoint_index)]
    }
    response = post_with_retry(payload)
    return response


def get_transactions(txs_ids: List[str]):
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
    return response


def get_transaction(tx_id: str) -> dict:
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
    return response