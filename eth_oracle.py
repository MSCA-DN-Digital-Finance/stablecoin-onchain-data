import os
import time
import json
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("GRAPH_API_KEY")

SUBGRAPH_ID = "39QtNcs7YrvJUvh2sNVGFTKLeahMBpB3BKWypPrSNYLc"
ENDPOINT = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}"

ASSET_PAIR = "ETH/USD"
CHAINLINK_DECIMALS = 8


def gql(query: str, endpoint: str, variables: dict | None = None, timeout: int = 60) -> dict:
    """GraphQL helper with 429 handling + basic retry on transient network errors."""
    while True:
        try:
            r = requests.post(endpoint, json={"query": query, "variables": variables or {}}, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
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


PRICES_QUERY = """
query($assetPair:String!, $first:Int!, $ts:Int!){
  prices(
    where: { assetPair: $assetPair, timestamp_gt: $ts }
    orderBy: timestamp
    orderDirection: asc
    first: $first
  ){
    id
    timestamp
    price
    assetPair { id }
  }
}
"""


def fetch_price_updates(endpoint: str, asset_pair: str, start_ts: int = 0, batch: int = 1000) -> pd.DataFrame:
    """
    Fetch price updates strictly after start_ts using timestamp-based pagination
    (avoids expensive `skip`).
    """
    rows = []
    ts = int(start_ts)

    while True:
        chunk = gql(
            PRICES_QUERY,
            endpoint=endpoint,
            variables={"assetPair": asset_pair, "first": batch, "ts": ts},
        )["prices"]

        if not chunk:
            break

        rows.extend(chunk)
        ts = int(chunk[-1]["timestamp"])  # advance cursor

        # safety: if subgraph returns same timestamp repeatedly (shouldn't), break
        if len(chunk) > 0 and int(chunk[0]["timestamp"]) == ts and len(chunk) == 1:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # normalize
    df["asset_pair"] = df["assetPair"].apply(lambda x: x.get("id") if isinstance(x, dict) else x)
    df.drop(columns=["assetPair"], inplace=True, errors="ignore")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    df["price_raw"] = pd.to_numeric(df["price"], errors="coerce")
    df["price_usd"] = df["price_raw"] / (10 ** CHAINLINK_DECIMALS)

    df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # de-dupe (overlap-safe)
    df = df[~df["id"].duplicated(keep="last")]

    return df


def to_hourly_price(df_events: pd.DataFrame, end_time: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Hourly time series filled through end_time (current hour by default).
    Uses last known price and forward-fills missing hours (incl. end).
    """
    if df_events is None or df_events.empty:
        return pd.DataFrame()

    end_time = (end_time or pd.Timestamp.utcnow()).floor("h")

    # Use last update observed within the hour, then ffill across hours
    hourly = df_events[["price_usd", "price_raw"]].resample("1h").last()

    start_time = hourly.index.min().floor("h")
    full_idx = pd.date_range(start=start_time, end=end_time, freq="1h", tz="UTC")

    hourly = hourly.reindex(full_idx).ffill()

    return hourly


def update_chainlink_ethusd_parquet(
    endpoint: str,
    events_path: str = "./data/ETH_blocks/Chainlink/ethusd_oracle_events.parquet",
    hourly_path: str = "./data/ETH_blocks/Chainlink/ethusd_oracle_hourly.parquet",
):
    events_path = Path(events_path)
    hourly_path = Path(hourly_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    hourly_path.parent.mkdir(parents=True, exist_ok=True)

    now_hour = pd.Timestamp.utcnow().floor("h")

    # -------- update raw events --------
    try:
        prev_events = pd.read_parquet(events_path)
        prev_events.index = pd.to_datetime(prev_events.index, utc=True)

        # overlap by 1 hour to avoid boundary misses
        last_ts = int(prev_events["timestamp"].dropna().iloc[-1])
        start_ts = max(0, last_ts - 3600)

        new_events = fetch_price_updates(endpoint, ASSET_PAIR, start_ts=start_ts, batch=1000)
        if not new_events.empty:
            all_events = pd.concat([prev_events, new_events], axis=0)
            all_events = all_events[~all_events["id"].duplicated(keep="last")]
            all_events.sort_index(inplace=True)
            all_events.to_parquet(events_path)
            print(f"----- Updated Chainlink {ASSET_PAIR} events: +{len(new_events)} -----")
        else:
            all_events = prev_events
            print(f"----- No new Chainlink {ASSET_PAIR} events -----")

    except FileNotFoundError:
        all_events = fetch_price_updates(endpoint, ASSET_PAIR, start_ts=0, batch=1000)
        all_events.to_parquet(events_path)
        print(f"----- Collected FULL Chainlink {ASSET_PAIR} events: {len(all_events)} -----")

    # -------- update hourly series (ALWAYS extend to current hour) --------
    try:
        prev_hourly = pd.read_parquet(hourly_path)
        prev_hourly.index = pd.to_datetime(prev_hourly.index, utc=True)

        if all_events.empty:
            # No events at all: just extend existing hourly index (ffill if any)
            start_time = prev_hourly.index.min()
            full_idx = pd.date_range(start=start_time, end=now_hour, freq="1h", tz="UTC")
            hourly = prev_hourly.reindex(full_idx).ffill()
        else:
            # rebuild tail window then extend to now
            tail_start = (prev_hourly.index.max() - pd.Timedelta(days=7)) if len(prev_hourly) else all_events.index.min()
            tail_events = all_events.loc[all_events.index >= tail_start]
            new_tail = to_hourly_price(tail_events, end_time=now_hour)

            if not new_tail.empty:
                head = prev_hourly.loc[prev_hourly.index < new_tail.index.min()]
                hourly = pd.concat([head, new_tail]).sort_index()
            else:
                hourly = prev_hourly

            # final extend-to-now safety
            full_idx = pd.date_range(start=hourly.index.min(), end=now_hour, freq="1h", tz="UTC")
            hourly = hourly.reindex(full_idx).ffill()

        hourly = hourly[~hourly.index.duplicated(keep="last")]
        hourly.to_parquet(hourly_path)
        print(f"----- Updated Chainlink {ASSET_PAIR} hourly series (extended to now) -----")

    except FileNotFoundError:
        if all_events.empty:
            # Nothing exists and no events fetched: create a minimal row
            hourly = pd.DataFrame(
                {"price_usd": [pd.NA], "price_raw": [pd.NA]},
                index=pd.DatetimeIndex([now_hour], tz="UTC"),
            )
        else:
            hourly = to_hourly_price(all_events, end_time=now_hour)

        hourly.to_parquet(hourly_path)
        print(f"----- Built FULL Chainlink {ASSET_PAIR} hourly series (extended to now) -----")


if __name__ == "__main__":
    try:
        update_chainlink_ethusd_parquet(ENDPOINT)
    except Exception as e:
        print(f"[ERROR] chainlink_ethusd -> {e}")