from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# Config
# ============================================================

API_KEY = os.environ.get("GRAPH_API_KEY")

SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
ENDPOINT = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}"

POOL = "0x3416cf6c708da44db2624d63ea0aaef7113527c6".lower()
NFPM_MAINNET = "0xc36442b4a4522e871399cd717abdd847ab11fe88".lower()

FIRST_BLOCK = 13609065

BLOCKS_PATH = "./data/ETH_blocks/hourly_blocks.parquet"
EVENTS_PATH = "./data/Uniswap/hourly_non_nfpm_mint_burn_events_full.parquet"

REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
PAGE_SIZE = 1000
SLEEP_BETWEEN_PAGES = 0.05


# ============================================================
# Safe parquet helpers
# ============================================================

def _safe_read_parquet(path: str | Path):
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def atomic_parquet_write(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


# ============================================================
# GraphQL helpers
# ============================================================

session = requests.Session()


def gql(endpoint: str, query: str, variables=None, timeout: int = REQUEST_TIMEOUT, retries: int = MAX_RETRIES):
    last_err = None
    for attempt in range(retries):
        try:
            r = session.post(
                endpoint,
                json={"query": query, "variables": variables or {}},
                timeout=timeout,
            )
            r.raise_for_status()
            j = r.json()
            if "errors" in j:
                raise RuntimeError(j["errors"])
            return j["data"]
        except Exception as e:
            last_err = e
            wait = min(30, 2 ** attempt)
            print(f"[retry {attempt + 1}/{retries}] request failed: {e}; sleeping {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"GraphQL request failed after {retries} retries: {last_err}")


def paginate_id_gt(
    endpoint: str,
    query: str,
    root: str,
    variables: dict,
    page_size: int = PAGE_SIZE,
    sleep_s: float = SLEEP_BETWEEN_PAGES,
) -> list[dict]:
    out = []
    last_id = ""

    while True:
        v = dict(variables)
        v["first"] = int(page_size)
        v["lastId"] = last_id

        data = gql(endpoint, query, v)
        batch = data[root]

        if not batch:
            break

        out.extend(batch)
        last_id = batch[-1]["id"]

        if len(batch) < page_size:
            break

        if sleep_s > 0:
            time.sleep(sleep_s)

    return out


# ============================================================
# Event queries
# ============================================================

MINTS_QUERY_GT = f"""
query($pool: String!, $first: Int!, $lastId: ID!, $b0: Int!, $b1: Int!) {{
  mints(
    block: {{ number: $b1 }}
    first: $first
    orderBy: id
    orderDirection: asc
    where: {{
      id_gt: $lastId
      pool: $pool
      owner_not: "{NFPM_MAINNET}"
      transaction_: {{ blockNumber_gt: $b0, blockNumber_lte: $b1 }}
    }}
  ) {{
    id
    timestamp
    logIndex
    owner
    sender
    origin
    tickLower
    tickUpper
    amount
    transaction {{ id blockNumber }}
  }}
}}
"""

BURNS_QUERY_GT = f"""
query($pool: String!, $first: Int!, $lastId: ID!, $b0: Int!, $b1: Int!) {{
  burns(
    block: {{ number: $b1 }}
    first: $first
    orderBy: id
    orderDirection: asc
    where: {{
      id_gt: $lastId
      pool: $pool
      owner_not: "{NFPM_MAINNET}"
      transaction_: {{ blockNumber_gt: $b0, blockNumber_lte: $b1 }}
    }}
  ) {{
    id
    timestamp
    logIndex
    owner
    origin
    tickLower
    tickUpper
    amount
    transaction {{ id blockNumber }}
  }}
}}
"""

MINTS_QUERY_GTE = f"""
query($pool: String!, $first: Int!, $lastId: ID!, $b0: Int!, $b1: Int!) {{
  mints(
    block: {{ number: $b1 }}
    first: $first
    orderBy: id
    orderDirection: asc
    where: {{
      id_gt: $lastId
      pool: $pool
      owner_not: "{NFPM_MAINNET}"
      transaction_: {{ blockNumber_gte: $b0, blockNumber_lte: $b1 }}
    }}
  ) {{
    id
    timestamp
    logIndex
    owner
    sender
    origin
    tickLower
    tickUpper
    amount
    transaction {{ id blockNumber }}
  }}
}}
"""

BURNS_QUERY_GTE = f"""
query($pool: String!, $first: Int!, $lastId: ID!, $b0: Int!, $b1: Int!) {{
  burns(
    block: {{ number: $b1 }}
    first: $first
    orderBy: id
    orderDirection: asc
    where: {{
      id_gt: $lastId
      pool: $pool
      owner_not: "{NFPM_MAINNET}"
      transaction_: {{ blockNumber_gte: $b0, blockNumber_lte: $b1 }}
    }}
  ) {{
    id
    timestamp
    logIndex
    owner
    origin
    tickLower
    tickUpper
    amount
    transaction {{ id blockNumber }}
  }}
}}
"""


# ============================================================
# Normalization
# ============================================================

def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "id",
        "timestamp",
        "logIndex",
        "owner",
        "sender",
        "origin",
        "tickLower",
        "tickUpper",
        "liquidityAmount",
        "type",
        "liquiditySign",
        "tx",
        "blockNumber",
        "event_key",
    ])


def _norm_events(df: pd.DataFrame, typ: str) -> pd.DataFrame:
    if df.empty:
        return _empty_events_df()

    df = df.copy()

    if "sender" not in df.columns:
        df["sender"] = pd.NA

    df["type"] = typ
    df["liquiditySign"] = 1 if typ == "mint" else -1

    df["tx"] = df["transaction"].apply(lambda x: x["id"])
    df["blockNumber"] = df["transaction"].apply(lambda x: int(x["blockNumber"]))
    df.drop(columns=["transaction"], inplace=True)

    df["timestamp"] = df["timestamp"].astype("int64")
    df["logIndex"] = df["logIndex"].astype("int64")
    df["tickLower"] = df["tickLower"].astype("int64")
    df["tickUpper"] = df["tickUpper"].astype("int64")

    # keep exact liquidity as string
    df["liquidityAmount"] = df["amount"].astype("string")
    df.drop(columns=["amount"], inplace=True)

    for c in ["owner", "sender", "origin", "tx"]:
        df[c] = df[c].astype("string").str.lower()

    df["event_key"] = (
        df["tx"].astype(str)
        + ":"
        + df["logIndex"].astype(str)
        + ":"
        + df["type"].astype(str)
    )

    return df[[
        "id",
        "timestamp",
        "logIndex",
        "owner",
        "sender",
        "origin",
        "tickLower",
        "tickUpper",
        "liquidityAmount",
        "type",
        "liquiditySign",
        "tx",
        "blockNumber",
        "event_key",
    ]]


def fetch_non_nfpm_events_frozen(
    endpoint: str,
    pool: str,
    block_start: int,
    block_end: int,
    inclusive_start: bool,
) -> pd.DataFrame:
    vars_ = {
        "pool": pool.lower(),
        "b0": int(block_start),
        "b1": int(block_end),
    }

    if inclusive_start:
        q_mints = MINTS_QUERY_GTE
        q_burns = BURNS_QUERY_GTE
    else:
        q_mints = MINTS_QUERY_GT
        q_burns = BURNS_QUERY_GT

    mints = paginate_id_gt(endpoint, q_mints, "mints", vars_)
    burns = paginate_id_gt(endpoint, q_burns, "burns", vars_)

    dfm = _norm_events(pd.DataFrame(mints), "mint")
    dfb = _norm_events(pd.DataFrame(burns), "burn")

    if dfm.empty and dfb.empty:
        return _empty_events_df()

    events = pd.concat([dfm, dfb], ignore_index=True)

    # deterministic dedup
    events = events.sort_values(["event_key", "id"]).drop_duplicates("event_key", keep="first")
    events = events.sort_values(["blockNumber", "tx", "logIndex", "type", "id"]).reset_index(drop=True)

    return events


# ============================================================
# Hour mapping
# ============================================================

def normalize_blocks_df(df: pd.DataFrame) -> pd.DataFrame:
    if "hour_utc" not in df.columns:
        raise ValueError("hourly_blocks must contain 'hour_utc'")
    if "block_number" not in df.columns:
        raise ValueError("hourly_blocks must contain 'block_number'")

    out = df.copy().reset_index().rename(columns={"index": "hour_idx"})
    out["hour_utc"] = pd.to_datetime(out["hour_utc"], utc=True)
    out["block_number"] = out["block_number"].astype("int64")

    out = out.sort_values(["hour_utc", "block_number"]).reset_index(drop=True)

    out["interval_start_hour_utc"] = out["hour_utc"].shift(1)
    out["block_start"] = out["block_number"].shift(1)
    out["block_end"] = out["block_number"]

    return out[[
        "hour_idx",
        "interval_start_hour_utc",
        "hour_utc",
        "block_start",
        "block_end",
    ]]


def assign_events_to_hours(events: pd.DataFrame, hourly_blocks: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    hour_map = hourly_blocks[[
        "hour_idx",
        "interval_start_hour_utc",
        "hour_utc",
        "block_start",
        "block_end",
    ]].sort_values("block_end").reset_index(drop=True)

    events = events.sort_values("blockNumber").reset_index(drop=True)

    # Assign each event to the first hourly boundary with block_end >= event.blockNumber
    out = pd.merge_asof(
        events,
        hour_map,
        left_on="blockNumber",
        right_on="block_end",
        direction="forward",
        allow_exact_matches=True,
    )

    # Drop anything beyond the latest available hourly block boundary
    out = out[out["block_end"].notna()].copy()

    # Safety filter: event should belong to (block_start, block_end]
    # For first row block_start is NaN, so keep if blockNumber <= block_end
    ok = out["block_start"].isna() | (
        (out["blockNumber"] > out["block_start"]) & (out["blockNumber"] <= out["block_end"])
    )
    out = out[ok].copy()

    out["hour_utc"] = pd.to_datetime(out["hour_utc"], utc=True)
    out["interval_start_hour_utc"] = pd.to_datetime(out["interval_start_hour_utc"], utc=True)

    cols = [
        "hour_idx",
        "interval_start_hour_utc",
        "hour_utc",
        "block_start",
        "block_end",
        "id",
        "timestamp",
        "logIndex",
        "owner",
        "sender",
        "origin",
        "tickLower",
        "tickUpper",
        "liquidityAmount",
        "type",
        "liquiditySign",
        "tx",
        "blockNumber",
        "event_key",
    ]
    return out[cols].sort_values(["blockNumber", "tx", "logIndex", "type", "id"]).reset_index(drop=True)


# ============================================================
# Main update logic
# ============================================================

if __name__ == "__main__":
    if not API_KEY:
        print("[WARN] GRAPH_API_KEY missing. Exiting gracefully.")
        raise SystemExit(0)

    blocks_df = _safe_read_parquet(BLOCKS_PATH)
    if blocks_df is None or blocks_df.empty:
        print("[WARN] hourly_blocks missing/empty. Exiting gracefully.")
        raise SystemExit(0)

    try:
        hourly_blocks = normalize_blocks_df(blocks_df)
    except Exception as e:
        print(f"[WARN] Failed to normalize hourly_blocks: {e}")
        raise SystemExit(0)

    target_block = int(hourly_blocks["block_end"].max())
    print(f"[INFO] Latest block in hourly_blocks: {target_block}")

    existing = _safe_read_parquet(EVENTS_PATH)

    if existing is None or existing.empty:
        start_block = FIRST_BLOCK
        inclusive_start = True
        existing = pd.DataFrame()
        print(f"[INFO] No existing events file found. Full history run from block {FIRST_BLOCK}.")
    else:
        if "block_end" not in existing.columns:
            print("[WARN] Existing events parquet must contain 'block_end'. Exiting gracefully.")
            raise SystemExit(0)

        existing = existing.copy()
        existing["hour_utc"] = pd.to_datetime(existing["hour_utc"], utc=True)
        existing["interval_start_hour_utc"] = pd.to_datetime(existing["interval_start_hour_utc"], utc=True)
        existing["block_end"] = existing["block_end"].astype("int64")
        existing["blockNumber"] = existing["blockNumber"].astype("int64")

        start_block = int(existing["block_end"].max())
        inclusive_start = False
        print(f"[INFO] Existing events file found. Updating from block > {start_block}.")

    if target_block <= start_block:
        print(f"[INFO] Nothing to do: target_block={target_block} <= start_block={start_block}")
        raise SystemExit(0)

    print(f"[INFO] Fetching non-NFPM mint/burn events in block range "
          f"{'[' if inclusive_start else '('}{start_block}, {target_block}]")

    try:
        new_events_raw = fetch_non_nfpm_events_frozen(
            endpoint=ENDPOINT,
            pool=POOL,
            block_start=start_block,
            block_end=target_block,
            inclusive_start=inclusive_start,
        )
    except Exception as e:
        print(f"[WARN] Event fetch failed: {e}")
        raise SystemExit(0)

    if new_events_raw.empty:
        print("[INFO] No new events found in the requested block range.")
        raise SystemExit(0)

    try:
        new_events = assign_events_to_hours(new_events_raw, hourly_blocks)
    except Exception as e:
        print(f"[WARN] Failed to assign events to hours: {e}")
        raise SystemExit(0)

    if new_events.empty:
        print("[INFO] No events remained after hour assignment. Exiting.")
        raise SystemExit(0)

    if existing.empty:
        full = new_events.copy()
    else:
        full = pd.concat([existing, new_events], ignore_index=True)

    full["hour_utc"] = pd.to_datetime(full["hour_utc"], utc=True)
    full["interval_start_hour_utc"] = pd.to_datetime(full["interval_start_hour_utc"], utc=True)

    full = (
        full
        .sort_values(["blockNumber", "tx", "logIndex", "type", "id"])
        .drop_duplicates(subset=["event_key"], keep="last")
        .reset_index(drop=True)
    )

    try:
        atomic_parquet_write(full, EVENTS_PATH)
        print(f"[OK] Updated {EVENTS_PATH}")
        print(f"[OK] Added {len(new_events):,} new event rows")
        print(f"[OK] Total rows now: {len(full):,}")
    except Exception as e:
        print(f"[WARN] Failed to write output parquet: {e}")
        raise SystemExit(0)