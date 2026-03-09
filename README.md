# Stablecoin Onchain data
[![DOI](.zenodo.18339068.svg)](https://doi.org/10.5281/zenodo.18339067)

This daily release contains hourly data regarding stablecoins on major DeFi venues. All the data is collected from subgraphs and aggregated into ready-to-use parquet files. 


# Data release structure
```
.
├── AAVE/
│   ├── aave_v2_*_eth                   # Hourly market data for Aave markets (v2, ETH)
│   ├── aave_v3_*_eth                   # Hourly market data for Aave markets (v2, ETH)
│   └── liquidations/
│       ├── hourly                      # Hourly liquidation aggregates (v2 & v3, ETH)
│       └── events                      # Liquidation events (v2 & v3, ETH)
│
├── Curve/
│   └── curve_3pool_hourly              # Hourly metrics for Curve 3pool (ETH)
│
├── ETH_blocks/
│   ├── hourly_blocks                   # Closest Ethereum block per hour (UTC) since 2022-01-01
│   └── Chainlink/
│       ├── *_oracle_events             # ETH/USD and BTC/USD event data
│       ├── *_oracle_hourly             # ETH/USD and BTC/USD hourly data
│       ├── usd_index_hourly            # Hourly reconstructed USD index
│       └── fx/
│           ├── eurusd_*                # Hourly + event data
│           ├── gbpusd_*                # Hourly + event data
│           └── jpyusd_*                # Hourly + event data
│
└── Uniswap/
    ├── USDC_USDT_hourly_metrics        # Hourly metrics: swaps, TVL, net flows, etc.
    ├── DAI_USDC_hourly_metrics         # Hourly metrics: swaps, TVL, net flows, etc.
    ├── hourly_pool_state               # helper dataset for the liquidity curve collection
    ├── hourly_positions_full           # Hourly snapshot of opened liquidity positions
    ├── hourly_liquidity_full           # ±50 ticks centered on the peg tick (hourly snapshot)
    └── hourly_liquidity_pricecentered  # ±50 ticks centered on current price tick (hourly snapshot)
```

# Data conventions

- **Timezone**: all hourly series are aligned to UTC hours
- **Granularity**: most datasets are hourly, with some event-level tables and some daily series
- **Chain**: Ethereum mainnet
- **Source**: The Graph subgraphs (protocol-specific)

# Dataset details

## Aave (Ethereum: v2 & v3) — Lending market + liquidations

Aave is one of the largest decentralized, crypto-collateralized lending protocols. Users supply assets to earn yield, and borrowers post collateral to borrow assets (including stablecoins). Because stablecoins are widely used as borrowed assets and as units of account, Aave market data is central to studying onchain credit conditions.

This repository provides:

- Hourly market series for stablecoin-relevant markets (e.g., USDC, USDT) on Aave v2 and Aave v3 (Ethereum) Typical contents include utilization/borrow dynamics, rates, and market-level balances (exact fields depend on the subgraph entities used).
- Liquidation data for Aave v2 and v3:
    - Event-level liquidations: each liquidation transaction/event (useful for microstructure and stress-event studies)
    - Hourly liquidation aggregates: counts and/or volumes aggregated to the hour (useful for panel regressions and monitoring)


## Curve (Ethereum) — Stable swap liquidity benchmark (3pool)

Curve is a DEX optimized for low-slippage swaps between correlated assets, especially stablecoins. The 3pool (historically a major pool for stablecoin liquidity) is often treated as a benchmark for stablecoin swap liquidity and imbalance dynamics.

This repository provides:

- Hourly metrics for the Curve 3pool, suitable for tracking:
    - pool liquidity conditions
    - imbalances (when one stablecoin is being heavily swapped in/out)

## Uniswap (Ethereum) — Concentrated liquidity stablecoin pools + liquidity distribution

Uniswap is a leading DEX. In Uniswap v3, liquidity providers concentrate liquidity over price ranges (“ticks”), making liquidity distribution itself a key state variable—especially for stablecoin pairs that tend to trade tightly around a peg.

This repository provides:

- Hourly pool metrics for stablecoin pairs, including: USDC/USDT, DAI/USDC 
    - Typical contents include swaps/flow metrics and TVL measures.
- Hourly liquidity curve snapshots for USDC/USDT:
    - ±50 ticks around the peg tick
    - ±50 ticks around the current price tick
- Swap size impact curves around the current active tick
- Full hourly snaphots of active positions, including owner adresses, liquidity size and range. 
- Non NFPM positions (incl. MEV bots) are also tracked via direct Mints/Burns on the pool

These snapshots help analyze how liquidity migrates during normal times vs. depeg events.

## Chainlink (Ethereum) — Oracle data (prices + FX)

Chainlink is a leading decentralized oracle network that publishes reference prices used by many DeFi protocols. Because smart contracts cannot directly access offchain market data, oracles act as the bridge that brings prices (and sometimes interest rates or FX rates) onchain.

This repo provides from Chainlink:

- ETH/USD and BTC/USD oracle data:
    - event-level updates (each onchain oracle update)
    - hourly series aligned to a consistent UTC hourly grid
- FX feeds: USD/EUR, USD/GBP, USD/JPY (hourly + event-level)
- An hourly reconstructed USD index (as produced by this pipeline)


# Typical use cases

- Stablecoin peg analysis and microstructure (DEX liquidity distribution vs. price)
- Liquidation dynamics vs. oracle movements
- Cross-protocol liquidity / stress indicators (Aave ↔ Uniswap ↔ Curve)
- Building unified hourly panels for econometrics / ML

# How to use

- Browse to the protocol folder of interest (e.g., AAVE/, Uniswap/, ETH_blocks/Chainlink/)
- Load the Parquet files into your tool of choice
- For cross-dataset joins, use ETH_blocks/hourly_blocks.parquet (hourly UTC) as the common alignment key

