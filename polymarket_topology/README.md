# Topological Compression of Prediction Market Belief Dynamics

This repository starts with a reproducible Polymarket data pipeline. The current scope is only data collection and panel construction; PCA, persistent homology, logistic regression, and neural nets are intentionally left for later work.

## What The Pipeline Builds

- `data/raw/gamma_markets_*.json`: raw Gamma API market metadata responses.
- `data/raw/clob_price_history_*.jsonl`: raw CLOB price-history responses, one market per line.
- `data/raw/data_api_trades_*.jsonl`: raw Data API trade responses for longer-horizon reconstruction.
- `data/processed/markets.parquet`: cleaned closed/resolved binary market metadata.
- `data/processed/market_universe.parquet`: crypto taxonomy over the metadata universe, including core BTC/ETH price-threshold markets and satellite policy/ETF/MicroStrategy markets.
- `data/processed/selected_markets.csv`: selected core and satellite markets with taxonomy fields.
- `data/processed/excluded_markets.csv`: excluded crypto candidates with exclusion reasons.
- `data/processed/prices_long.parquet`: cleaned YES-token history with columns:
  - `timestamp`
  - `market_id`
  - `token_id`
  - `yes_price`
  - `category`
  - `event_id`
  - `resolved_outcome`
- `data/processed/panel_hourly_raw.parquet`: continuous hourly timestamp-by-market matrix with observed prices only.
- `data/processed/panel_hourly_active_ffill.parquet`: active-window forward-filled hourly matrix.
- `data/processed/panel_hourly_core.parquet`: strict primary BTC/ETH core panel.
- `data/processed/panel_hourly_core_plus_satellites.parquet`: strict core plus crypto policy/ETF/MicroStrategy satellite panel.
- `data/processed/validation_report.json`: quality-gate report.
- `data/processed/dataset_manifest.json`: dataset parameters, selected market IDs, and validation status.
- `data/processed/stress_tests/`: coverage summaries under alternate minimum-trade and fill policies.

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

## Run The Current Crypto Pipeline

The current research target is BTC/ETH price-threshold markets as the core panel, plus crypto policy, ETF, and MicroStrategy markets as satellite panels.

```bash
python src/fetch_markets.py --limit 5000 --closed true --order volumeNum --ascending false
python src/select_markets.py
python src/fetch_trade_history.py \
  --input data/processed/market_universe.parquet \
  --output data/processed/prices_long.parquet \
  --page-size 500 \
  --max-trades-per-market 3500
python src/build_market_panel.py --freq 1h
python src/validate_dataset.py
python src/stress_test_dataset.py
```

The public Data API currently returns enough data to build and audit the selected universe, but high-volume markets hit a practical pagination depth ceiling around 3,500 trades per market. The script default remains higher for future API improvements, but this committed dataset was pulled with `--max-trades-per-market 3500` after deeper offsets returned 400 responses on many high-volume markets.

Large raw trade JSONL files are written under `data/raw/` locally for auditability. Full raw trade payloads can exceed GitHub's normal file-size limit, so the processed Parquet files and validation artifacts are the committed reproducible dataset unless Git LFS or external object storage is added.

## Legacy/Smoke-Test Commands

Fetch a smaller high-volume metadata sample:

```bash
python src/fetch_markets.py --limit 1000 --closed true
```

Fetch historical YES-token prices from the CLOB price-history endpoint. This endpoint is useful for recent chart history, but its public interval filters are short horizon (`1h`, `6h`, `1d`, `1w`, `1m`) and often empty for older or inactive markets:

```bash
python src/fetch_price_history.py --input data/processed/markets.parquet --limit 250
```

Fetch trade-derived probabilities for a smaller exploratory subset:

```bash
python src/fetch_trade_history.py \
  --input data/processed/markets.parquet \
  --output data/processed/prices_long.parquet \
  --limit 1000 \
  --max-trades-per-market 5000
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

Run:

```bash
python src/validate_dataset.py
python src/stress_test_dataset.py
```

Validation outputs:

- `validation_report.json`: pass/fail quality gates, counts, missingness, timestamp range, and limitations.
- `coverage_by_market.csv`: usable points and observed date ranges by market.
- `coverage_by_timestamp.csv`: active market counts and missingness by timestamp.
- `dataset_manifest.json`: API sources, pull parameters, selected market IDs, and validation status.
- `stress_tests/stress_test_summary.csv`: alternate min-trade and fill-policy coverage summaries.

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

## Current Dataset Status

As of the latest committed validation run, the dataset is useful for pipeline development and market-universe inspection, but it is **not yet analysis-ready** for the full paper. It passes schema, binary/resolved-market, price-bound, duplicate, minimum-market-count, minimum-points-per-market, median-active-core, and missingness gates. It fails the 180-day strict hourly coverage gate.

Current key counts:

- Selected markets: 252.
- Core BTC/ETH price-threshold markets: 141.
- Satellite crypto policy/ETF/MicroStrategy markets: 111.
- Trade-derived price rows: 862,924.
- Raw observed trade span: 2024-01-05 to 2026-06-17 UTC.
- Strict primary core panel: 263 hourly rows by 23 markets, about 11 usable days.
- Strict core-plus-satellite panel: 263 hourly rows by 37 markets.
- Primary panel missingness: 12.56%.
- Median active core markets in the primary panel: 22.

The main blocker is data depth, not code structure: 239 of 252 markets hit the observed 3,500-trade cap in the current public pull, so older lifetime histories are truncated for many high-volume markets. To reach a statistically strong 180+ day market-state panel, the next step is a deeper archival source, Git LFS/object storage for raw pulls, or a Polymarket endpoint/export that supports deeper historical pagination.

## Notes On API Fields

Polymarket Gamma fields are not fully consistent across market vintages. The parser handles string-encoded JSON fields such as `outcomes`, `outcomePrices`, `clobTokenIds`, and `umaResolutionStatuses`, while preserving raw responses in `data/raw` for auditability.

The initial cleaned metadata keeps only binary Yes/No markets with a YES CLOB token ID. Resolved outcome is inferred from final `outcomePrices` when one outcome is at least `0.99`; the raw resolution metadata is retained where available.

For statistically meaningful topology experiments, do not rely on the small committed smoke-test sample alone. Pull a longer trade-derived panel across many closed markets and report coverage before modeling: number of markets, number of markets with usable histories, timestamp span, category/event mix, panel missingness, and per-market observation counts.
