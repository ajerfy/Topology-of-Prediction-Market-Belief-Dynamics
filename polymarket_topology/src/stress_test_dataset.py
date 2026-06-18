from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_market_panel import active_ffill, panel_metadata, raw_panel
from utils import ensure_dirs, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test dataset coverage under alternate gates.")
    parser.add_argument("--prices", default="data/processed/prices_long.parquet")
    parser.add_argument("--markets", default="data/processed/market_universe.parquet")
    parser.add_argument("--output-dir", default="data/processed/stress_tests")
    parser.add_argument("--freq", default="1h")
    args = parser.parse_args()

    root = project_root()
    prices_path = Path(args.prices)
    markets_path = Path(args.markets)
    output_dir = Path(args.output_dir)
    if not prices_path.is_absolute():
        prices_path = root / prices_path
    if not markets_path.is_absolute():
        markets_path = root / markets_path
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    ensure_dirs([output_dir])

    prices = pd.read_parquet(prices_path)
    prices["market_id"] = prices["market_id"].astype(str)
    markets = pd.read_parquet(markets_path)
    markets["market_id"] = markets["market_id"].astype(str)
    for col in ["start_date", "close_date", "end_date"]:
        if col in markets.columns:
            markets[col] = pd.to_datetime(markets[col], utc=True, errors="coerce")
    selected = markets[markets["is_core"].fillna(False) | markets["is_satellite"].fillna(False)].copy()
    core_ids = set(selected[selected["is_core"].fillna(False)]["market_id"].astype(str))
    core_sat_ids = set(selected["market_id"].astype(str))

    summaries = []
    for min_trades in [500, 1000, 2500]:
        counts = prices.groupby("market_id").size()
        keep = set(counts[counts >= min_trades].index.astype(str))
        trial_prices = prices[prices["market_id"].isin(keep)].copy()
        raw = raw_panel(trial_prices, args.freq) if len(trial_prices) else pd.DataFrame()
        for fill_name, fill_limit in [("6h", 6), ("24h", 24), ("active_unlimited", None)]:
            filled = active_ffill(raw, selected, fill_limit=fill_limit) if len(raw) else raw
            for panel_name, ids in [("core", core_ids), ("core_plus_satellites", core_sat_ids)]:
                cols = [col for col in filled.columns if col in ids]
                panel = filled[cols] if cols else pd.DataFrame(index=filled.index)
                meta = panel_metadata(panel, args.freq, f"{fill_name}_{panel_name}")
                meta.update(
                    {
                        "min_trades": min_trades,
                        "fill_limit": fill_name,
                        "panel": panel_name,
                    }
                )
                summaries.append(meta)
    out = pd.DataFrame(summaries)
    out.to_csv(output_dir / "stress_test_summary.csv", index=False)
    with (output_dir / "stress_test_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
