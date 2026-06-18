# Topological Compression of Prediction Market Belief Dynamics

This repository starts with a reproducible Polymarket data pipeline. The current scope is only data collection and panel construction; PCA, persistent homology, logistic regression, and neural nets are intentionally left for later work.

## What The Pipeline Builds

- `data/raw/gamma_markets_*.json`: raw Gamma API market metadata responses.
- `data/raw/clob_price_history_*.jsonl`: raw CLOB price-history responses, one market per line.
- `data/raw/data_api_trades_*.jsonl`: raw Data API trade responses for longer-horizon reconstruction.
- `data/processed/markets.parquet`: cleaned closed/resolved binary market metadata.
- `data/processed/prices_long.parquet`: cleaned YES-token history with columns:
  - `timestamp`
  - `market_id`
  - `token_id`
  - `yes_price`
  - `category`
  - `event_id`
  - `resolved_outcome`
- `data/processed/price_panel.parquet`: hourly timestamp-by-market matrix, ready for later feature construction.

## Setup

```bash
cd polymarket_topology
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts use public Polymarket endpoints:

- Gamma metadata API: `https://gamma-api.polymarket.com`
- CLOB price-history API: `https://clob.polymarket.com`

No API key is required for the endpoints used here.

## Run The Pipeline

Start with a manageable pull of high-volume closed markets:

```bash
python src/fetch_markets.py --limit 1000 --closed true
```

Fetch historical YES-token prices. The CLOB API returns empty histories for some older or inactive markets, and the script logs those while continuing.

```bash
python src/fetch_price_history.py --input data/processed/markets.parquet --limit 250
```

For research-scale historical coverage, use the Data API trade fallback. The CLOB `prices-history` endpoint is useful for recent chart history, but its public interval filters are short horizon (`1h`, `6h`, `1d`, `1w`, `1m`). The trade fallback reconstructs YES probabilities from executed trades:

```bash
python src/fetch_trade_history.py \
  --input data/processed/markets.parquet \
  --output data/processed/prices_long.parquet \
  --limit 1000 \
  --max-trades-per-market 5000
```

Build an hourly market-state panel:

```bash
python src/build_market_panel.py --freq 1h
```

To focus the metadata pull on one Gamma category when useful:

```bash
python src/fetch_markets.py --limit 500 --closed true --category Sports
```

Useful alternatives:

```bash
python src/fetch_markets.py --limit 1000 --closed true --order endDate --ascending false
python src/fetch_price_history.py --input data/processed/markets.parquet --limit 500 --min-volume-clob 10000
python src/fetch_trade_history.py --input data/processed/markets.parquet --limit 500 --max-trades-per-market 10000
```

## Validate The Data

Open the notebook:

```bash
jupyter notebook notebooks/01_data_check.ipynb
```

It checks:

- number of markets collected,
- number of markets with usable YES price history,
- timestamp range,
- missingness in the panel,
- distribution of categories,
- sample plots of several market price histories.

## Notes On API Fields

Polymarket Gamma fields are not fully consistent across market vintages. The parser handles string-encoded JSON fields such as `outcomes`, `outcomePrices`, `clobTokenIds`, and `umaResolutionStatuses`, while preserving raw responses in `data/raw` for auditability.

The initial cleaned metadata keeps only binary Yes/No markets with a YES CLOB token ID. Resolved outcome is inferred from final `outcomePrices` when one outcome is at least `0.99`; the raw resolution metadata is retained where available.

For statistically meaningful topology experiments, do not rely on the small committed smoke-test sample alone. Pull a longer trade-derived panel across many closed markets and report coverage before modeling: number of markets, number of markets with usable histories, timestamp span, category/event mix, panel missingness, and per-market observation counts.
