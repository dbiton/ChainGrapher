import queue
import requests
import os
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

ETH_RPC_URL = os.getenv("ETH_RPC_URL")


def fetcher_prestate(block_number: int):
    diffFalse = fetch_block_trace(block_number, "prestateTracer", {"diffMode": False})
    diffTrue = fetch_block_trace(block_number, "prestateTracer", {"diffMode": True})
    return block_number, diffFalse, diffTrue


def fetcher_call(block_number: int):
    trace = fetch_block_trace(block_number, "callTracer")
    return block_number, trace


def fetch_block_trace(block_number: str, tracer_name: str, tracer_config={}) -> dict:
    if tracer_name not in ["callTracer", "prestateTracer"]:
        raise Exception(f"unknown tracer type {tracer_name}")
    if tracer_config not in [{}, {"diffMode": True}, {"diffMode": False}]:
        raise Exception(f"unknown tracer config {tracer_config}")
    try:
        payload = {
            "jsonrpc": "2.0",
            "method": "debug_traceBlockByNumber",
            "params": [
                hex(block_number),
                {
                    "tracer": tracer_name,
                    "tracerConfig": tracer_config
                }
            ],
            "id": 1
        }
        response = requests.post(ETH_RPC_URL, json=payload, timeout=600)
        if response.status_code == 200:
            print(f"fetched block trace {block_number} with {tracer_name}, {tracer_config}")
            result = response.json()["result"]
            if result is None:
                return None
        else:
            print(f"Error tracing block: {response.text}")
    except Exception as e:
        print(f"Error tracing block: {e}")


NODES_URLS = [
    "https://cloudflare-eth.com",
    "https://rpc.ankr.com/eth",
    "https://eth-mainnet.public.blastapi.io",
    "https://mainnet.infura.io/v3/61c26b521ed84355864460361fc8ca52",
    "https://eth-mainnet.g.alchemy.com/v2/LXa59vi3WiXQzdnd469is-WafCkdDDss",
]
eth_clients = [Web3(Web3.HTTPProvider(url)) for url in NODES_URLS]
conn = [eth_client.is_connected() for eth_client in eth_clients]
eth_clients_queue = queue.Queue()
for eth_client in eth_clients:
    eth_clients_queue.put(eth_client)


def fetch_block(block_num):
    while True:
        try:
            web3 = eth_clients_queue.get()
            block_details = web3.eth.get_block(block_num, full_transactions=False)
            eth_clients_queue.put(web3)
            print(f"fetched block {block_num}")
            return block_details['hash'].hex()
        except Exception as e:
            print(f"Error fetching block {block_num}: {str(e)}")
