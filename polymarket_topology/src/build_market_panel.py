from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from utils import ensure_dirs, project_root, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build timestamp-by-market probability panel.")
    parser.add_argument("--input", default="data/processed/prices_long.parquet")
    parser.add_argument("--freq", default="1h")
    parser.add_argument("--output", default="data/processed/price_panel.parquet")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    root = project_root()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = root / input_path
    if not output_path.is_absolute():
        output_path = root / output_path
    ensure_dirs([output_path.parent])

    prices = pd.read_parquet(input_path)
    if prices.empty:
        panel = pd.DataFrame()
    else:
        prices = prices.copy()
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
        prices = prices.sort_values(["market_id", "timestamp"])
        prices["bucket"] = prices["timestamp"].dt.floor(args.freq)
        panel = (
            prices.dropna(subset=["bucket", "market_id", "yes_price"])
            .groupby(["bucket", "market_id"], as_index=False)["yes_price"]
            .last()
            .pivot(index="bucket", columns="market_id", values="yes_price")
            .sort_index()
            .ffill()
        )
        panel.index.name = "timestamp"

    panel.to_parquet(output_path)
    logging.info("Saved panel with shape=%s to %s", panel.shape, output_path)


if __name__ == "__main__":
    main()
