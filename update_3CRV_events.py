import os
import time
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================
# Config
# ============================
POOL_ADDRESS = "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7".lower()
LP_TOKEN_ADDRESS = "0x6c3f90f043a72fa612cbac8115ee7e52bde6e490".lower()
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
ZERO_TOPIC = "0x" + "00" * 32
COINS = {0: "DAI", 1: "USDC", 2: "USDT"}

LP_PATH = Path("./data/Curve/3CRV_lpevents.parquet")
SWAP_PATH = Path("./data/Curve/3CRV_swapevents.parquet")
HOURLY_BLOCKS_PATH = Path("./data/ETH_blocks/hourly_blocks.parquet")

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
ETHERSCAN_URL = "https://api.etherscan.io/v2/api"
CHAIN_ID = 1

CHUNK_SIZE = 50_000
PAGE_SIZE = 1000
SLEEP_BETWEEN_CALLS = 0.25


# ============================
# Logging
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# ============================
# Topic hashes
# ============================
def keccak_hex(text: str) -> str:
    try:
        from eth_utils import keccak
        return "0x" + keccak(text=text).hex()
    except Exception:
        try:
            from web3 import Web3
            return Web3.keccak(text=text).hex()
        except Exception as e:
            raise RuntimeError(
                "Need eth_utils or web3 installed to compute event signature hashes."
            ) from e


TRANSFER_TOPIC0 = keccak_hex("Transfer(address,address,uint256)")
TOKEN_EXCHANGE_TOPIC0 = keccak_hex("TokenExchange(address,int128,uint256,int128,uint256)")


# ============================
# Helpers
# ============================
def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def parse_int(x, default=None) -> Optional[int]:
    if x is None:
        return default
    if isinstance(x, float) and math.isnan(x):
        return default
    if isinstance(x, int):
        return x

    s = str(x).strip()
    if s in {"", "0x", "0X", "nan", "None"}:
        return default
    if s.startswith(("0x", "0X")):
        return int(s, 16)
    return int(s)


def topic_to_address(topic: str) -> str:
    return "0x" + topic[-40:].lower()


def split_words(data_hex: str) -> List[str]:
    s = data_hex[2:] if data_hex.startswith("0x") else data_hex
    if len(s) % 64 != 0:
        s = s.zfill((len(s) + 63) // 64 * 64)
    return [s[i:i + 64] for i in range(0, len(s), 64)]


def to_utc_datetime(ts: int):
    return pd.to_datetime(ts, unit="s", utc=True)


def atomic_write_parquet(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def empty_lp_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "block_number", "transaction_index", "log_index", "timestamp",
        "datetime", "transaction_hash", "block_hash", "category",
        "from_address", "to_address", "lp_amount"
    ])


def empty_swap_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "block_number", "transaction_index", "log_index", "timestamp",
        "datetime", "transaction_hash", "block_hash", "buyer", "sold_id",
        "sold_symbol", "tokens_sold", "bought_id", "bought_symbol",
        "tokens_bought"
    ])


def load_or_empty(path: Path, kind: str) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            log.warning("Failed to read %s: %s", path, e)
    log.warning("%s not found or unreadable, starting from empty.", path)
    return empty_lp_df() if kind == "lp" else empty_swap_df()


def dedupe_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # block_number + log_index uniquely identifies a log within Ethereum
    df = df.drop_duplicates(subset=["block_number", "log_index"])
    return df.sort_values(
        ["block_number", "transaction_index", "log_index"]
    ).reset_index(drop=True)


# ============================
# Etherscan API
# ============================
def etherscan_get(session: requests.Session, params: Dict, timeout: int = 30) -> Dict:
    full_params = {
        "chainid": CHAIN_ID,
        "apikey": ETHERSCAN_API_KEY,
    }
    full_params.update(params)

    r = session.get(ETHERSCAN_URL, params=full_params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    result = data.get("result")
    message = str(data.get("message", ""))

    if data.get("status") == "0":
        result_str = str(result)
        if "No records found" in result_str or "No records found" in message:
            return {"result": []}
        if "Query Timeout" in result_str or "Query Timeout" in message:
            raise TimeoutError(result_str)
        if "rate limit" in result_str.lower() or "rate limit" in message.lower():
            raise RuntimeError(result_str)
        raise RuntimeError(f"Etherscan API error: {data}")

    return data


def fetch_logs(
    session: requests.Session,
    address: str,
    topic0: str,
    from_block: int,
    to_block: int,
    extra_topics: Optional[Dict[str, str]] = None,
    chunk_size: int = CHUNK_SIZE,
) -> List[Dict]:
    if from_block > to_block:
        return []

    all_logs = []
    extra_topics = extra_topics or {}

    current = from_block
    current_chunk_size = chunk_size

    while current <= to_block:
        end = min(current + current_chunk_size - 1, to_block)

        try:
            page = 1
            while True:
                params = {
                    "module": "logs",
                    "action": "getLogs",
                    "fromBlock": current,
                    "toBlock": end,
                    "address": address,
                    "topic0": topic0,
                    "page": page,
                    "offset": PAGE_SIZE,
                }
                params.update(extra_topics)

                data = etherscan_get(session, params)
                batch = data.get("result", []) or []
                all_logs.extend(batch)

                if len(batch) < PAGE_SIZE:
                    break

                page += 1
                time.sleep(SLEEP_BETWEEN_CALLS)

            current = end + 1
            time.sleep(SLEEP_BETWEEN_CALLS)

        except TimeoutError:
            if current_chunk_size <= 1000:
                log.warning(
                    "Query timeout for %s [%s, %s] even at small chunk size. Skipping chunk.",
                    address, current, end
                )
                current = end + 1
            else:
                old = current_chunk_size
                current_chunk_size = max(1000, current_chunk_size // 2)
                log.warning(
                    "Query timeout for %s [%s, %s]. Reducing chunk size %s -> %s.",
                    address, current, end, old, current_chunk_size
                )
                time.sleep(1)

        except Exception as e:
            log.warning(
                "Failed fetching logs for %s [%s, %s]: %s",
                address, current, end, e
            )
            current = end + 1
            time.sleep(1)

    return all_logs


_block_ts_cache: Dict[int, int] = {}


def get_block_timestamp(session: requests.Session, block_number: int) -> Optional[int]:
    if block_number in _block_ts_cache:
        return _block_ts_cache[block_number]

    try:
        data = etherscan_get(session, {
            "module": "proxy",
            "action": "eth_getBlockByNumber",
            "tag": hex(block_number),
            "boolean": "false",
        })
        ts = parse_int(data["result"]["timestamp"])
        _block_ts_cache[block_number] = ts
        return ts
    except Exception as e:
        log.warning("Failed to fetch timestamp for block %s: %s", block_number, e)
        return None


# ============================
# Decoders
# ============================
def decode_lp_logs(logs: List[Dict], session: requests.Session) -> pd.DataFrame:
    rows = []

    for log_item in logs:
        try:
            topics = log_item.get("topics", [])
            if len(topics) < 3:
                continue

            from_address = topic_to_address(topics[1])
            to_address = topic_to_address(topics[2])

            if from_address == ZERO_ADDRESS:
                category = "mint"
            elif to_address == ZERO_ADDRESS:
                category = "burn"
            else:
                continue

            block_number = parse_int(log_item.get("blockNumber"))
            if block_number is None:
                log.warning("Skipping LP log with missing block number: %s", log_item)
                continue

            timestamp = parse_int(log_item.get("timeStamp"))
            if timestamp is None:
                timestamp = get_block_timestamp(session, block_number)
            if timestamp is None:
                log.warning(
                    "Skipping LP log with missing timestamp at block %s tx %s",
                    block_number, log_item.get("transactionHash")
                )
                continue

            raw_value = parse_int(log_item.get("data"))
            if raw_value is None:
                log.warning("Skipping LP log with missing amount: %s", log_item)
                continue

            rows.append({
                "block_number": block_number,
                "transaction_index": parse_int(log_item.get("transactionIndex"), default=-1),
                "log_index": parse_int(log_item.get("logIndex"), default=-1),
                "timestamp": timestamp,
                "datetime": to_utc_datetime(timestamp),
                "transaction_hash": str(log_item.get("transactionHash", "")).lower(),
                "block_hash": str(log_item.get("blockHash", "")).lower(),
                "category": category,
                "from_address": from_address,
                "to_address": to_address,
                "lp_amount": raw_value / 1e18,
            })

        except Exception as e:
            log.warning("Failed decoding LP log: %s | raw=%s", e, log_item)

    if not rows:
        return empty_lp_df()

    return pd.DataFrame(rows)


def decode_swap_logs(logs: List[Dict], session: requests.Session) -> pd.DataFrame:
    rows = []

    for log_item in logs:
        try:
            topics = log_item.get("topics", [])
            data_words = split_words(log_item.get("data", "0x"))

            # Handle both possible ABI/indexing layouts robustly
            if len(topics) >= 2:
                buyer = topic_to_address(topics[1])
                if len(data_words) < 4:
                    continue
                sold_id = int(data_words[0], 16)
                tokens_sold_raw = int(data_words[1], 16)
                bought_id = int(data_words[2], 16)
                tokens_bought_raw = int(data_words[3], 16)
            else:
                if len(data_words) < 5:
                    continue
                buyer = "0x" + data_words[0][-40:].lower()
                sold_id = int(data_words[1], 16)
                tokens_sold_raw = int(data_words[2], 16)
                bought_id = int(data_words[3], 16)
                tokens_bought_raw = int(data_words[4], 16)

            sold_symbol = COINS.get(sold_id)
            bought_symbol = COINS.get(bought_id)
            if sold_symbol is None or bought_symbol is None:
                log.warning(
                    "Unknown coin ids in swap log: sold_id=%s bought_id=%s tx=%s",
                    sold_id, bought_id, log_item.get("transactionHash")
                )
                continue

            block_number = parse_int(log_item.get("blockNumber"))
            if block_number is None:
                log.warning("Skipping swap log with missing block number: %s", log_item)
                continue

            timestamp = parse_int(log_item.get("timeStamp"))
            if timestamp is None:
                timestamp = get_block_timestamp(session, block_number)
            if timestamp is None:
                log.warning(
                    "Skipping swap log with missing timestamp at block %s tx %s",
                    block_number, log_item.get("transactionHash")
                )
                continue

            sold_scale = 1e18 if sold_symbol == "DAI" else 1e6
            bought_scale = 1e18 if bought_symbol == "DAI" else 1e6

            rows.append({
                "block_number": block_number,
                "transaction_index": parse_int(log_item.get("transactionIndex"), default=-1),
                "log_index": parse_int(log_item.get("logIndex"), default=-1),
                "timestamp": timestamp,
                "datetime": to_utc_datetime(timestamp),
                "transaction_hash": str(log_item.get("transactionHash", "")).lower(),
                "block_hash": str(log_item.get("blockHash", "")).lower(),
                "buyer": buyer,
                "sold_id": sold_id,
                "sold_symbol": sold_symbol,
                "tokens_sold": tokens_sold_raw / sold_scale,
                "bought_id": bought_id,
                "bought_symbol": bought_symbol,
                "tokens_bought": tokens_bought_raw / bought_scale,
            })

        except Exception as e:
            log.warning("Failed decoding swap log: %s | raw=%s", e, log_item)

    if not rows:
        return empty_swap_df()

    return pd.DataFrame(rows)


# ============================
# Update functions
# ============================
def update_lp_events(session: requests.Session, target_block: int, fallback_start_block: int):
    df = load_or_empty(LP_PATH, "lp")

    if not df.empty and "block_number" in df.columns:
        start_block = int(pd.to_numeric(df["block_number"], errors="coerce").max()) + 1
    else:
        start_block = fallback_start_block

    if start_block > target_block:
        log.info("LP events already up to date (last=%s, target=%s).", start_block - 1, target_block)
        return

    log.info("Updating LP events from block %s to %s", start_block, target_block)

    mint_logs = fetch_logs(
        session=session,
        address=LP_TOKEN_ADDRESS,
        topic0=TRANSFER_TOPIC0,
        from_block=start_block,
        to_block=target_block,
        extra_topics={"topic1": ZERO_TOPIC},
    )

    burn_logs = fetch_logs(
        session=session,
        address=LP_TOKEN_ADDRESS,
        topic0=TRANSFER_TOPIC0,
        from_block=start_block,
        to_block=target_block,
        extra_topics={"topic2": ZERO_TOPIC},
    )

    new_df = decode_lp_logs(mint_logs + burn_logs, session)
    if new_df.empty:
        log.info("No new LP events found.")
        return

    out = dedupe_and_sort(pd.concat([df, new_df], ignore_index=True))
    atomic_write_parquet(out, LP_PATH)
    log.info("LP events updated: +%s rows, last block=%s", len(new_df), int(out["block_number"].max()))


def update_swap_events(session: requests.Session, target_block: int, fallback_start_block: int):
    df = load_or_empty(SWAP_PATH, "swap")

    if not df.empty and "block_number" in df.columns:
        start_block = int(pd.to_numeric(df["block_number"], errors="coerce").max()) + 1
    else:
        start_block = fallback_start_block

    if start_block > target_block:
        log.info("Swap events already up to date (last=%s, target=%s).", start_block - 1, target_block)
        return

    log.info("Updating swap events from block %s to %s", start_block, target_block)

    logs = fetch_logs(
        session=session,
        address=POOL_ADDRESS,
        topic0=TOKEN_EXCHANGE_TOPIC0,
        from_block=start_block,
        to_block=target_block,
    )

    new_df = decode_swap_logs(logs, session)
    if new_df.empty:
        log.info("No new swap events found.")
        return

    out = dedupe_and_sort(pd.concat([df, new_df], ignore_index=True))
    atomic_write_parquet(out, SWAP_PATH)
    log.info("Swap events updated: +%s rows, last block=%s", len(new_df), int(out["block_number"].max()))


# ============================
# Main
# ============================
def main():
    if not ETHERSCAN_API_KEY:
        log.warning("Missing ETHERSCAN_API_KEY in environment. Exiting without update.")
        return

    if not HOURLY_BLOCKS_PATH.exists():
        log.warning("Missing hourly blocks file: %s", HOURLY_BLOCKS_PATH)
        return

    try:
        hourly = pd.read_parquet(HOURLY_BLOCKS_PATH)
    except Exception as e:
        log.warning("Failed to read hourly blocks parquet: %s", e)
        return

    if hourly.empty or "block_number" not in hourly.columns:
        log.warning("hourly_blocks parquet is empty or missing 'block_number'.")
        return

    try:
        block_numbers = pd.to_numeric(hourly["block_number"], errors="coerce").dropna().astype(int)
        if block_numbers.empty:
            log.warning("No valid block numbers found in hourly_blocks parquet.")
            return
        target_block = int(block_numbers.max())
        fallback_start_block = int(block_numbers.min())
    except Exception as e:
        log.warning("Failed to parse block numbers from hourly blocks: %s", e)
        return

    log.info("Target block from hourly indexer: %s", target_block)

    session = make_session()

    try:
        update_lp_events(session, target_block, fallback_start_block)
    except Exception as e:
        log.warning("LP update failed: %s", e)

    try:
        update_swap_events(session, target_block, fallback_start_block)
    except Exception as e:
        log.warning("Swap update failed: %s", e)


if __name__ == "__main__":
    main()