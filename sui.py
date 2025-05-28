from typing import Any, List
import httpx
import time
from dotenv import load_dotenv
import os
import json

load_dotenv()

SUI_RPC_URL = os.getenv("SUI_RPC_URL")
PATH_OUTPUT = os.getenv("PATH_OUTPUT")

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
    while True:
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
            with open(PATH_OUTPUT, "w") as f:
                print(f"Checkpoint {checkpoint_id} trace written to file.")
                json.dump([checkpoint, txs], f)
            time.sleep(delay)
            checkpoint_id += 1
            if end is not None and checkpoint_id > end:
                break
        except Exception as e:
            print(f"Error at checkpoint {checkpoint_id}: {e}")
            break


last = 150497902
trace_all_checkpoints(start=150497902 - 100, end=150497902 - 90)
