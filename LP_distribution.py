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

BLOCKS_PATH = "./data/ETH_blocks/hourly_blocks.parquet"

# Existing full-history files to update
POSITIONS_PATH = "./data/Uniswap/hourly_positions_full.parquet"
HOURS_PATH = "./data/Uniswap/hourly_positions_hours_full.parquet"

PAGE_SIZE = 1000
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60


# ============================================================
# Safe parquet helpers
# ============================================================

def _safe_read_parquet(path: str):
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        print(f"[WARN] Missing file: {path}")
        return None
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def atomic_parquet_write(df: pd.DataFrame, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


# ============================================================
# GraphQL helpers
# ============================================================

session = requests.Session()


def gql_post(endpoint: str, query: str, variables: dict, timeout: int = REQUEST_TIMEOUT, retries: int = MAX_RETRIES) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            r = session.post(
                endpoint,
                json={"query": query, "variables": variables},
                timeout=timeout,
            )
            r.raise_for_status()
            out = r.json()
            if "errors" in out:
                raise RuntimeError(out["errors"])
            return out["data"]
        except Exception as e:
            last_err = e
            wait = min(30, 2 ** attempt)
            time.sleep(wait)
    raise RuntimeError(f"GraphQL request failed after {retries} retries: {last_err}")


POSITIONS_AT_BLOCK = """
query($pool: String!, $block: Int!, $first: Int!, $lastID: String!) {
  positions(
    block: { number: $block }
    first: $first
    where: {
      pool: $pool,
      liquidity_gt: "0",
      id_gt: $lastID
    }
    orderBy: id
    orderDirection: asc
  ) {
    id
    owner
    pool { id }
    tickLower { tickIdx }
    tickUpper { tickIdx }
    liquidity
    depositedToken0
    depositedToken1
    withdrawnToken0
    withdrawnToken1
    collectedFeesToken0
    collectedFeesToken1
    transaction { id }
  }
}
"""


def paginate_by_id(endpoint: str, query: str, root_field: str, variables: dict, page_size: int = 1000):
    rows_all = []
    last_id = ""

    while True:
        v = dict(variables)
        v.update({"first": page_size, "lastID": last_id})

        data = gql_post(endpoint, query, v)
        rows = data[root_field]

        if not rows:
            break

        rows_all.extend(rows)
        last_id = rows[-1]["id"]

        if len(rows) < page_size:
            break

    return rows_all


def fetch_positions_at_block(block_number: int) -> pd.DataFrame:
    rows = paginate_by_id(
        ENDPOINT,
        POSITIONS_AT_BLOCK,
        "positions",
        {"pool": POOL, "block": int(block_number)},
        page_size=PAGE_SIZE,
    )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "id", "owner", "pool", "tickLower", "tickUpper", "liquidity",
            "depositedToken0", "depositedToken1", "withdrawnToken0", "withdrawnToken1",
            "collectedFeesToken0", "collectedFeesToken1", "transaction"
        ])

    df["tickLower"] = df["tickLower"].apply(lambda x: int(x["tickIdx"]) if isinstance(x, dict) else pd.NA)
    df["tickUpper"] = df["tickUpper"].apply(lambda x: int(x["tickIdx"]) if isinstance(x, dict) else pd.NA)
    df["pool"] = df["pool"].apply(lambda x: x["id"] if isinstance(x, dict) else x)
    df["transaction"] = df["transaction"].apply(lambda x: x["id"] if isinstance(x, dict) else x)

    # Keep big numeric values as strings to avoid precision loss
    raw_numeric_cols = [
        "liquidity",
        "depositedToken0",
        "depositedToken1",
        "withdrawnToken0",
        "withdrawnToken1",
        "collectedFeesToken0",
        "collectedFeesToken1",
    ]
    for col in raw_numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def fetch_positions_at_block_with_retry(block_number: int, max_retries: int = MAX_RETRIES) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fetch_positions_at_block(block_number)
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                break
            wait = min(60, 2 ** attempt)
            print(f"[retry {attempt}/{max_retries}] block={block_number} failed: {e}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch positions at block={block_number}: {last_err}")


# ============================================================
# Normalization / collector
# ============================================================

def normalize_blocks_df(df: pd.DataFrame) -> pd.DataFrame:
    if "hour_utc" not in df.columns:
        raise ValueError("hourly_blocks must contain 'hour_utc'")
    if "block_number" not in df.columns:
        raise ValueError("hourly_blocks must contain 'block_number'")

    out = df.copy()
    out["hour_utc"] = pd.to_datetime(out["hour_utc"], utc=True)
    out["block_number"] = out["block_number"].astype("int64")
    out["hour_idx"] = out.index.astype("int64")
    out = out.sort_values("hour_utc").reset_index(drop=True)
    return out[["hour_idx", "hour_utc", "block_number"]]


def collect_positions_for_hours(hours_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    position_frames = []
    hour_rows = []

    for _, row in hours_df.iterrows():
        hour_idx = int(row["hour_idx"])
        hour_utc = pd.Timestamp(row["hour_utc"])
        block_number = int(row["block_number"])

        print(f"[INFO] Collecting hour={hour_utc} block={block_number}")
        pos_df = fetch_positions_at_block_with_retry(block_number)
        n_positions = len(pos_df)

        if not pos_df.empty:
            pos_df.insert(0, "hour_idx", hour_idx)
            pos_df.insert(1, "hour_utc", hour_utc)
            pos_df.insert(2, "block_number", block_number)
            position_frames.append(pos_df)

        hour_rows.append({
            "hour_idx": hour_idx,
            "hour_utc": hour_utc,
            "block_number": block_number,
            "n_open_positions": n_positions,
        })

    if position_frames:
        positions_new = pd.concat(position_frames, ignore_index=True)
    else:
        positions_new = pd.DataFrame(columns=[
            "hour_idx", "hour_utc", "block_number",
            "id", "owner", "pool", "tickLower", "tickUpper", "liquidity",
            "depositedToken0", "depositedToken1", "withdrawnToken0", "withdrawnToken1",
            "collectedFeesToken0", "collectedFeesToken1", "transaction"
        ])

    hours_new = pd.DataFrame(hour_rows)
    return positions_new, hours_new


# ============================================================
# Main updater
# ============================================================

if __name__ == "__main__":
    if not API_KEY:
        print("[WARN] GRAPH_API_KEY is missing. Exiting gracefully.")
        raise SystemExit(0)

    blocks_df = _safe_read_parquet(BLOCKS_PATH)
    if blocks_df is None or blocks_df.empty:
        print("[WARN] hourly_blocks missing/empty; cannot update positions. Exiting gracefully.")
        raise SystemExit(0)

    try:
        blocks_df = normalize_blocks_df(blocks_df)
    except Exception as e:
        print(f"[WARN] Failed to normalize blocks dataframe: {e}")
        raise SystemExit(0)

    positions_old = _safe_read_parquet(POSITIONS_PATH)
    hours_old = _safe_read_parquet(HOURS_PATH)

    # Do not backfill from scratch in the nightly updater
    if positions_old is None:
        print(f"[WARN] Existing positions file not found: {POSITIONS_PATH}. Skipping update.")
        raise SystemExit(0)

    if positions_old.empty and (hours_old is None or hours_old.empty):
        print("[WARN] Existing history file(s) empty; nightly updater will not backfill from scratch. Skipping.")
        raise SystemExit(0)

    # Normalize existing files
    if positions_old is not None and not positions_old.empty:
        if "hour_utc" not in positions_old.columns:
            print("[WARN] Existing positions parquet does not contain 'hour_utc'. Skipping update.")
            raise SystemExit(0)
        positions_old = positions_old.copy()
        positions_old["hour_utc"] = pd.to_datetime(positions_old["hour_utc"], utc=True)

    if hours_old is not None and not hours_old.empty:
        if "hour_utc" not in hours_old.columns:
            print("[WARN] Existing hours parquet does not contain 'hour_utc'. Ignoring hours file.")
            hours_old = None
        else:
            hours_old = hours_old.copy()
            hours_old["hour_utc"] = pd.to_datetime(hours_old["hour_utc"], utc=True)

    # Prefer the hours coverage file because it correctly captures zero-position hours
    if hours_old is not None and not hours_old.empty:
        last_processed_hour = pd.Timestamp(hours_old["hour_utc"].max())
        print(f"[INFO] Last processed hour from hours coverage file: {last_processed_hour}")
    else:
        last_processed_hour = pd.Timestamp(positions_old["hour_utc"].max())
        print(f"[WARN] Hours coverage file missing/empty; falling back to positions file.")
        print(f"[WARN] Last processed hour inferred from positions file: {last_processed_hour}")
        print(f"[WARN] This can miss trailing hours with zero positions.")

    new_hours_df = blocks_df.loc[blocks_df["hour_utc"] > last_processed_hour].copy()

    if new_hours_df.empty:
        print("[INFO] No new hours to collect. Exiting.")
        raise SystemExit(0)

    print(f"[INFO] Found {len(new_hours_df)} new hour(s) to collect.")
    print(f"[INFO] Range: {new_hours_df['hour_utc'].min()} -> {new_hours_df['hour_utc'].max()}")

    try:
        positions_new, hours_new = collect_positions_for_hours(new_hours_df)
    except Exception as e:
        print(f"[WARN] Collector failed: {e}. Exiting without writing.")
        raise SystemExit(0)

    if hours_new.empty:
        print("[INFO] No new hours returned by collector. Exiting without writing.")
        raise SystemExit(0)

    # ------------------------------------------------------------
    # Append + dedupe
    # ------------------------------------------------------------

    # Positions
    if positions_old is None or positions_old.empty:
        positions_full = positions_new.copy()
    else:
        positions_full = pd.concat([positions_old, positions_new], ignore_index=True)

    if not positions_full.empty:
        positions_full["hour_utc"] = pd.to_datetime(positions_full["hour_utc"], utc=True)
        positions_full = (
            positions_full
            .sort_values(["hour_utc", "id"])
            .drop_duplicates(subset=["hour_utc", "id"], keep="last")
            .reset_index(drop=True)
        )

    # Hours coverage
    if hours_old is None or hours_old.empty:
        hours_full = hours_new.copy()
    else:
        hours_full = pd.concat([hours_old, hours_new], ignore_index=True)

    hours_full["hour_utc"] = pd.to_datetime(hours_full["hour_utc"], utc=True)
    hours_full = (
        hours_full
        .sort_values(["hour_utc", "hour_idx"])
        .drop_duplicates(subset=["hour_utc"], keep="last")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Write safely
    #
    # IMPORTANT:
    # Write positions first, then hours coverage last.
    # The hours file acts like the "progress marker".
    # ------------------------------------------------------------
    try:
        atomic_parquet_write(positions_full, POSITIONS_PATH)
        atomic_parquet_write(hours_full, HOURS_PATH)
        print(f"[OK] Updated {POSITIONS_PATH}")
        print(f"[OK] Updated {HOURS_PATH}")
        print(f"[OK] Added {len(hours_new)} hour(s) and {len(positions_new)} position row(s).")
    except Exception as e:
        print(f"[WARN] Failed to write updated parquet(s): {e}")
        raise SystemExit(0)