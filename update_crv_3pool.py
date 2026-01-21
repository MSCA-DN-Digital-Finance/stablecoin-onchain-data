import pandas as pd 
from dotenv import load_dotenv
import os
import time
import json
import requests
from pathlib import Path

load_dotenv()

API_KEY = os.environ.get("GRAPH_API_KEY") 

SUBGRAPH_ID = "3fy93eAT56UJsRCEht8iFhfi6wjHWXtZ9dnnbQmvFopF" 
ENDPOINT = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}"

# 3pool address
POOL_ID = "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7"

def run_query(query: str, variables: dict | None = None):
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(ENDPOINT, json=payload)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data["data"]

def gql(q: str, v: dict | None = None):
    r = requests.post(ENDPOINT, json={"query": q, "variables": v or {}})
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(json.dumps(data["errors"], indent=2))
    return data["data"]

def format_curve_3pool_hourly(df_raw: pd.DataFrame) -> pd.DataFrame:

    df = df_raw.copy()

    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="s", utc=True)

    df = df.set_index("timestamp").sort_index()
    df.index = df.index.ceil('h')
    full_idx = pd.date_range(df.index[0].ceil("H"),
                            df.index[-1].ceil("H"),
                            freq="1H")
    df = df.reindex(full_idx,method = 'ffill')
    df = df.iloc[:-1,:]

    bal = pd.DataFrame(
        df["inputTokenBalances"].tolist(),
        index=df.index,
        columns=["bal_DAI_raw", "bal_USDC_raw", "bal_USDT_raw"],
    )

    vol_amt = pd.DataFrame(
        df["hourlyVolumeByTokenAmount"].tolist(),
        index=df.index,
        columns=["volAmt_DAI_raw", "volAmt_USDC_raw", "volAmt_USDT_raw"],
    )


    vol_usd = pd.DataFrame(
        df["hourlyVolumeByTokenUSD"].tolist(),
        index=df.index,
        columns=["volUSD_DAI", "volUSD_USDC", "volUSD_USDT"],
    ).astype(float)

    w = pd.DataFrame(
        df["inputTokenWeights"].tolist(),
        index=df.index,
        columns=["w_DAI_pct", "w_USDC_pct", "w_USDT_pct"],
    )

    bal["bal_DAI"] = bal["bal_DAI_raw"].astype(float) / 1e18
    bal["bal_USDC"] = bal["bal_USDC_raw"].astype(float) / 1e6
    bal["bal_USDT"] = bal["bal_USDT_raw"].astype(float) / 1e6

    vol_amt["volAmt_DAI"] = vol_amt["volAmt_DAI_raw"].astype(float) / 1e18
    vol_amt["volAmt_USDC"] = vol_amt["volAmt_USDC_raw"].astype(float) / 1e6
    vol_amt["volAmt_USDT"] = vol_amt["volAmt_USDT_raw"].astype(float) / 1e6

    for c in ["w_DAI_pct", "w_USDC_pct", "w_USDT_pct"]:
        w[c] = pd.to_numeric(w[c], errors="coerce")

    w["w_DAI"] = w["w_DAI_pct"] / 100.0
    w["w_USDC"] = w["w_USDC_pct"] / 100.0
    w["w_USDT"] = w["w_USDT_pct"] / 100.0

    for c in ["totalValueLockedUSD", "hourlyVolumeUSD"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = pd.concat(
        [
            df[["totalValueLockedUSD", "hourlyVolumeUSD"]],
            bal[["bal_DAI", "bal_USDC", "bal_USDT"]],
            w[["w_DAI", "w_USDC", "w_USDT"]],
            vol_amt[["volAmt_DAI", "volAmt_USDC", "volAmt_USDT"]],
            vol_usd[["volUSD_DAI", "volUSD_USDC", "volUSD_USDT"]],
        ],
        axis=1,
    )

    return out


def update_curve_3pool(start_ts: int):

    pool_hour_query = """
    query ($poolId: String!, $skip: Int!, $ts: Int!) {
        liquidityPoolHourlySnapshots(
        first: 1000
        skip: $skip
        orderBy: timestamp
        orderDirection: asc
        where: {pool: $poolId, timestamp_gt: $ts}
    ) {
        timestamp
        totalValueLockedUSD
        inputTokenBalances
        inputTokenWeights
        hourlyVolumeUSD
        hourlyVolumeByTokenAmount
        hourlyVolumeByTokenUSD
        outputTokenPriceUSD
    }
    }
    """

    all_rows = []
    skip = 0
    while True:
        batch = run_query(pool_hour_query, {"poolId": POOL_ID, "skip": skip, "ts": start_ts})["liquidityPoolHourlySnapshots"]
        if not batch:
            break
        all_rows.extend(batch)
        skip += len(batch)

    df = pd.DataFrame(all_rows)
    df = format_curve_3pool_hourly(df)
    return df

if __name__ == "__main__":
    try:
        old = pd.read_parquet('./data/Curve/curve_3pool_hourly.parquet')
    except:
        full = update_curve_3pool(start_ts=0)
        full.to_parquet('./data/Curve/curve_3pool_hourly.parquet')
        print(f"----- Collected Full Historical Curve 3pool data-----")
    else:
        try:
            new = update_curve_3pool(start_ts=int(old.index[-1].timestamp()))
        except Exception as e:
            print(e)
        else:
            full = pd.concat([old,new], axis=0)
            full.to_parquet('./data/Curve/curve_3pool_hourly.parquet')
            print(f"----- Updated Historical Curve 3pool data-----")