from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from utils import ensure_dirs, project_root, setup_logging


OUTPUT_COLUMNS = [
    "timestamp",
    "market_id",
    "token_id",
    "yes_price",
    "category",
    "event_id",
    "resolved_outcome",
    "market_family",
    "asset",
    "is_core",
    "is_satellite",
    "source",
    "trade_asset",
    "trade_outcome",
    "trade_side",
    "trade_size",
    "transaction_hash",
]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_source(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        logging.warning("Source %s does not exist: %s", source_name, path)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["market_id"] = df["market_id"].astype(str)
    df["token_id"] = df["token_id"].astype(str)
    if "source" not in df.columns:
        df["source"] = source_name
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[OUTPUT_COLUMNS]


def merge_sources(trades: pd.DataFrame, clob: pd.DataFrame) -> pd.DataFrame:
    prices = pd.concat([trades, clob], ignore_index=True)
    if prices.empty:
        return prices
    prices = prices[(prices["yes_price"] >= 0) & (prices["yes_price"] <= 1)].copy()
    prices["_source_priority"] = prices["source"].map({"data_api_trades": 0, "clob_prices_history": 1}).fillna(2)
    prices = prices.sort_values(["market_id", "timestamp", "_source_priority"])
    prices = prices.drop_duplicates(
        subset=["market_id", "timestamp", "token_id", "source", "transaction_hash"],
        keep="last",
    )
    prices = prices.drop(columns=["_source_priority"])
    return prices.sort_values(["market_id", "timestamp", "source"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge trade-derived and CLOB sampled price histories.")
    parser.add_argument("--trade-input", default="data/processed/trade_prices_long.parquet")
    parser.add_argument("--clob-input", default="data/processed/clob_price_history_long.parquet")
    parser.add_argument("--output", default="data/processed/prices_long.parquet")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    root = project_root()
    trade_path = resolve_path(root, args.trade_input)
    clob_path = resolve_path(root, args.clob_input)
    output_path = resolve_path(root, args.output)
    ensure_dirs([output_path.parent])

    trades = read_source(trade_path, "data_api_trades")
    clob = read_source(clob_path, "clob_prices_history")
    merged = merge_sources(trades, clob)
    merged.to_parquet(output_path, index=False)
    logging.info(
        "Merged %s trade rows and %s CLOB rows into %s rows across %s markets at %s",
        len(trades),
        len(clob),
        len(merged),
        merged["market_id"].nunique() if not merged.empty else 0,
        output_path,
    )


if __name__ == "__main__":
    main()
