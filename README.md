# Stablecoin Onchain data

[![DOI](https://zenodo.org/badge/1138117387.svg)](https://doi.org/10.5281/zenodo.18338843)

This daily release contains hourly data regarding stablecoins on major DeFi venues. All the data is collected from subgraphs and aggregated into ready-to-use parquet files. 

## AAVE

Hourly metrics for main AAVE stablecoins including interest rates, utilisation rates, TVL and volume. 


## Curve

Hourly metrics for Curve's main stablecoin liquidity reserve, the 3-pool

## Uniswap 

Hourly metrics for Uniswap's 2 largest stablecoin-stablecoin pools (USDC-USDT and DAI-USDC). This dataset includes price, volume, and aggregated hourly net swaps at each feeTier.

Hourly liquidity of the USDC-USDT pool : this dataset contains hourly active liquidity curves +- 50 bps around the peg or current price. 


## Additional data (block timestamps)

This dataset is needed to update the active liquidity curves. It catalogues the closest block number to each hour since 2022. 


