# aave_liquidations_update.py
import os
import time
import json
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("GRAPH_API_KEY")

# Subgraph IDs (Ethereum)
AAVE_SUBGRAPHS = {
    "v3": "JCNWRypm7FYwV8fx5HhzZPSFaMxgkPuw4TnR3Gpi81zk",
    "v2": "C2zniPn45RnLDGzVeGZCx2Sw3GXrbc9gL4ZfL8B8Em2j",
}


LIQ_QUERY = """
query($first:Int!, $skip:Int!, $ts:Int!){
  liquidates(
    where: { timestamp_gte: $ts }
    orderBy: timestamp
    orderDirection: asc
    first: $first
    skip:  $skip
  ){
    id
    hash
    nonce
    logIndex
    gasPrice
    gasUsed
    gasLimit
    blockNumber
    timestamp
    liquidator
    liquidatee
    market { id }
    positions { id }
    asset
    amount
    amountUSD
    profitUSD
  }
}
"""


def gql(query: str, endpoint: str, variables: dict | None = None, timeout: int = 60) -> dict:
    while True:
        try:
            r = requests.post(endpoint, json={"query": query, "variables": variables or {}}, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            # transient network issue -> sleep & retry
            time.sleep(2)
            continue

        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", "2")))
            continue

        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            raise RuntimeError(json.dumps(data["errors"], indent=2))
        return data["data"]


def _normalize_liquidates_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    # flatten nested objects
    for r in rows:
        r["market"] = r["market"]["id"] if r.get("market") else None
        r["positions"] = ",".join(p["id"] for p in (r.get("positions") or []))

    df = pd.DataFrame(rows)

    # types
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    df["blockNumber"] = pd.to_numeric(df["blockNumber"], errors="coerce").astype("Int64")
    df["amountUSD"] = pd.to_numeric(df["amountUSD"], errors="coerce")
    df["profitUSD"] = pd.to_numeric(df["profitUSD"], errors="coerce")

    # datetime index
    df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    return df


def fetch_liquidation_events(endpoint: str, start_ts: int = 0, batch: int = 1000) -> pd.DataFrame:
    """
    Fetch liquidation events from `start_ts` onwards (inclusive).
    For incremental updates, we deliberately overlap (caller can set start_ts earlier)
    and then de-duplicate by event `id`.
    """
    rows, skip = [], 0
    while True:
        chunk = gql(LIQ_QUERY, endpoint, {"first": batch, "skip": skip, "ts": int(start_ts)})["liquidates"]
        if not chunk:
            break
        rows.extend(chunk)
        skip += len(chunk)

    df = _normalize_liquidates_rows(rows)
    if df.empty:
        return df

    # de-dupe in case of overlap re-fetch
    df = df[~df["id"].duplicated(keep="last")]
    return df

def to_hourly_usd(events: pd.DataFrame, end_time: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Aggregate events to hourly totals and ensure the index is continuous up to end_time,
    filling missing hours with 0. Uses UTC.
    """
    end_time = (end_time or pd.Timestamp.utcnow()).floor("h")

    if events is None or events.empty:
        # Caller must reindex/extend based on existing hourly data if no events exist.
        return pd.DataFrame()

    hourly = pd.DataFrame(
        {
            "liquidation_usd": events["amountUSD"].resample("1h").sum(min_count=1),
            "profit_usd": events["profitUSD"].resample("1h").sum(min_count=1),
            "liquidation_count": events["id"].resample("1h").count(),
        }
    ).fillna(0.0)

    start_time = hourly.index.min().floor("h")
    full_idx = pd.date_range(start=start_time, end=end_time, freq="1h", tz="UTC")
    hourly = hourly.reindex(full_idx, fill_value=0.0)

    return hourly


def update_liquidations_parquet(version: str, endpoint: str, events_path: str, hourly_path: str):
    events_path = Path(events_path)
    hourly_path = Path(hourly_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    hourly_path.parent.mkdir(parents=True, exist_ok=True)

    now_hour = pd.Timestamp.utcnow().floor("h")

    # ---- update events ----
    try:
        prev_events = pd.read_parquet(events_path)
        prev_events.index = pd.to_datetime(prev_events.index, utc=True)

        last_ts = int(prev_events["timestamp"].dropna().iloc[-1])
        start_ts = max(0, last_ts - 2 * 3600)

        new_events = fetch_liquidation_events(endpoint, start_ts=start_ts, batch=1000)

        if not new_events.empty:
            all_events = pd.concat([prev_events, new_events], axis=0)
            all_events = all_events[~all_events["id"].duplicated(keep="last")]
            all_events.sort_index(inplace=True)
            all_events.to_parquet(events_path)
            print(f"----- Updated AAVE {version} liquidation events: +{len(new_events)} -----")
        else:
            all_events = prev_events
            print(f"----- No new AAVE {version} liquidation events -----")

    except FileNotFoundError:
        all_events = fetch_liquidation_events(endpoint, start_ts=0, batch=1000)
        all_events.to_parquet(events_path)
        print(f"----- Collected FULL AAVE {version} liquidation events: {len(all_events)} -----")

    # ---- update hourly aggregates (ALWAYS extend to now) ----
    try:
        prev_hourly = pd.read_parquet(hourly_path)
        prev_hourly.index = pd.to_datetime(prev_hourly.index, utc=True)

        if all_events.empty:
            # No events at all: just extend existing hourly index to now with zeros
            start_time = prev_hourly.index.min()
            full_idx = pd.date_range(start=start_time, end=now_hour, freq="1h", tz="UTC")
            full_hourly = prev_hourly.reindex(full_idx, fill_value=0.0)
        else:
            # Rebuild tail window then extend to now (filled with zeros)
            tail_start = prev_hourly.index.max() - pd.Timedelta(days=7) if len(prev_hourly) else all_events.index.min()
            tail_events = all_events.loc[all_events.index >= tail_start]
            new_hourly_tail = to_hourly_usd(tail_events, end_time=now_hour)

            if not new_hourly_tail.empty:
                head = prev_hourly.loc[prev_hourly.index < new_hourly_tail.index.min()]
                full_hourly = pd.concat([head, new_hourly_tail]).sort_index()
            else:
                full_hourly = prev_hourly

            # final extend-to-now safety
            start_time = full_hourly.index.min()
            full_idx = pd.date_range(start=start_time, end=now_hour, freq="1h", tz="UTC")
            full_hourly = full_hourly.reindex(full_idx, fill_value=0.0)

        full_hourly = full_hourly[~full_hourly.index.duplicated(keep="last")]
        full_hourly.to_parquet(hourly_path)
        print(f"----- Updated AAVE {version} hourly liquidation series (extended to now) -----")

    except FileNotFoundError:
        if all_events.empty:
            # If absolutely no events on first run, create a minimal series (just current hour)
            full_hourly = pd.DataFrame(
                {"liquidation_usd": [0.0], "profit_usd": [0.0], "liquidation_count": [0.0]},
                index=pd.DatetimeIndex([now_hour], tz="UTC"),
            )
        else:
            full_hourly = to_hourly_usd(all_events, end_time=now_hour)

        full_hourly.to_parquet(hourly_path)
        print(f"----- Built FULL AAVE {version} hourly liquidation series (extended to now) -----")

if __name__ == "__main__":
    if not API_KEY:
        raise RuntimeError("GRAPH_API_KEY is missing (set it in environment or .env).")

    for version, subgraph_id in AAVE_SUBGRAPHS.items():
        try:
            endpoint = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{subgraph_id}"

            update_liquidations_parquet(
                version=version,
                endpoint=endpoint,
                events_path=f"./data/AAVE/liquidations/aave_{version}_eth_liquidation_events.parquet",
                hourly_path=f"./data/AAVE/liquidations/aave_{version}_eth_liquidations_hourly.parquet",
            )

        except Exception as e:
            print(f"[ERROR] version={version} -> {e}")