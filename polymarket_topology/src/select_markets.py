from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from market_selection import classify_markets
from utils import ensure_dirs, project_root, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and classify crypto Polymarket markets.")
    parser.add_argument("--input", default="data/processed/markets.parquet")
    parser.add_argument("--output", default="data/processed/market_universe.parquet")
    parser.add_argument("--selected-csv", default="data/processed/selected_markets.csv")
    parser.add_argument("--excluded-csv", default="data/processed/excluded_markets.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    root = project_root()
    input_path = Path(args.input)
    output_path = Path(args.output)
    selected_path = Path(args.selected_csv)
    excluded_path = Path(args.excluded_csv)
    if not input_path.is_absolute():
        input_path = root / input_path
    if not output_path.is_absolute():
        output_path = root / output_path
    if not selected_path.is_absolute():
        selected_path = root / selected_path
    if not excluded_path.is_absolute():
        excluded_path = root / excluded_path

    ensure_dirs([output_path.parent, selected_path.parent, excluded_path.parent])

    markets = pd.read_parquet(input_path)
    universe = classify_markets(markets)
    selected = universe[universe["is_core"] | universe["is_satellite"]].copy()
    excluded = universe[~(universe["is_core"] | universe["is_satellite"])].copy()

    universe.to_parquet(output_path, index=False)
    selected.to_csv(selected_path, index=False)
    excluded.to_csv(excluded_path, index=False)

    logging.info("Classified %s markets", len(universe))
    logging.info("Selected %s markets: %s", len(selected), selected["market_family"].value_counts().to_dict())
    logging.info("Excluded %s markets", len(excluded))


if __name__ == "__main__":
    main()
