"""
Classify 3CRV-related addresses as EOA or contract via Etherscan eth_getCode.

Respects Etherscan rate limits (free tier: 5 calls/sec) with exponential backoff.
Retries on rate-limit errors and network failures instead of misclassifying.

Run:  python classify_addresses.py
"""

import json
import os
import time
import logging
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
URL = "https://api.etherscan.io/v2/api"
CACHE_PATH = "./data/Curve/address_book.json"
PARQUET_PATH = "./data/Curve/3CRV_lpevents.parquet"
ZERO = "0x0000000000000000000000000000000000000000"

# Rate-limit: stay under 5 calls/sec for free tier
MIN_DELAY = 0.35          # seconds between requests
MAX_RETRIES = 8           # per address
BASE_BACKOFF = 1.0        # seconds; doubles each retry
MAX_BACKOFF = 120.0       # cap per retry wait
SAVE_EVERY = 50           # persist cache every N new results


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE_PATH)


def classify_address(session: requests.Session, addr: str) -> str:
    """Return 'contract' or 'eoa'. Retries on rate-limit / transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(
                URL,
                params={
                    "chainid": 1,
                    "module": "proxy",
                    "action": "eth_getCode",
                    "address": addr,
                    "tag": "latest",
                    "apikey": ETHERSCAN_API_KEY,
                },
                timeout=30,
            )
            r.raise_for_status()
            payload = r.json()

            result = payload.get("result", "")
            message = str(payload.get("message", ""))
            combined = message + " " + str(result)

            # Detect rate-limit response
            if "Max rate limit reached" in combined or "rate limit" in combined.lower():
                wait = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
                log.warning("Rate-limited on %s, waiting %.1fs (attempt %d/%d)",
                            addr, wait, attempt, MAX_RETRIES)
                time.sleep(wait)
                continue

            # Detect other API-level errors (status=0 but not rate-limit)
            if payload.get("status") == "0" and result:
                wait = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
                log.warning("API error on %s: %s — retrying in %.1fs (attempt %d/%d)",
                            addr, combined.strip(), wait, attempt, MAX_RETRIES)
                time.sleep(wait)
                continue

            # Successful response
            if result and result != "0x":
                return "contract"
            return "eoa"

        except (requests.RequestException, ValueError, KeyError) as exc:
            wait = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
            log.warning("Request error on %s: %s — retrying in %.1fs (attempt %d/%d)",
                        addr, exc, wait, attempt, MAX_RETRIES)
            time.sleep(wait)

    raise RuntimeError(f"Failed to classify {addr} after {MAX_RETRIES} attempts")


def main():
    if not ETHERSCAN_API_KEY:
        log.error("ETHERSCAN_API_KEY not set in environment. Exiting.")
        return

    log.info("Loading events from %s", PARQUET_PATH)
    df = pd.read_parquet(PARQUET_PATH)
    try:
        all_in = df.groupby("to_address")["lp_amount"].sum()
        all_out = df.groupby("from_address")["lp_amount"].sum()
        max_flow = pd.concat([all_in, all_out]).groupby(level=0).max()
        max_flow = max_flow.drop(ZERO, errors="ignore")
        significant = max_flow[max_flow > 100].index.tolist()
        log.info("Significant addresses (flow > 100 LP): %d", len(significant))

        cache = load_cache()
        to_query = [a for a in significant if a not in cache]
        log.info("Cached: %d | To query: %d", len(cache), len(to_query))

        if not to_query:
            log.info("Nothing to do. All addresses already classified.")
            return

        session = requests.Session()
        new_count = 0
        n_eoa = 0
        n_contract = 0

        try:
            for i, addr in enumerate(to_query, 1):
                addr_type = classify_address(session, addr)
                cache[addr] = addr_type

                if addr_type == "eoa":
                    n_eoa += 1
                else:
                    n_contract += 1

                new_count += 1
                if new_count % SAVE_EVERY == 0:
                    save_cache(cache)
                    log.info("[%d/%d] Saved checkpoint — %d EOAs, %d contracts so far",
                            i, len(to_query), n_eoa, n_contract)

                if i % 100 == 0:
                    log.info("[%d/%d] progress — %d EOAs, %d contracts",
                            i, len(to_query), n_eoa, n_contract)

                # Respect rate limit between calls
                time.sleep(MIN_DELAY)

        except KeyboardInterrupt:
            log.info("Interrupted. Saving progress...")
        finally:
            save_cache(cache)
            session.close()

        total_eoa = sum(1 for v in cache.values() if v == "eoa")
        total_sc = sum(1 for v in cache.values() if v == "contract")
        log.info("Done. Total classified: %d EOAs, %d contracts (%d total)",
                total_eoa, total_sc, len(cache))
    except Exception as e:
        log.error("Error processing events: %s. Exiting.", e)

if __name__ == "__main__":
    main()
