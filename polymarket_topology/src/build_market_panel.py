from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from utils import ensure_dirs, project_root, setup_logging


def load_market_universe(root: Path, path: str) -> pd.DataFrame:
    market_path = Path(path)
    if not market_path.is_absolute():
        market_path = root / market_path
    markets = pd.read_parquet(market_path)
    markets["market_id"] = markets["market_id"].astype(str)
    for col in ["start_date", "close_date", "end_date"]:
        if col in markets.columns:
            markets[col] = pd.to_datetime(markets[col], utc=True, errors="coerce")
    return markets


def raw_panel(prices: pd.DataFrame, freq: str, continuous: bool = True) -> pd.DataFrame:
    prices = prices.copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    prices["bucket"] = prices["timestamp"].dt.floor(freq)
    panel = (
        prices.dropna(subset=["bucket", "market_id", "yes_price"])
        .groupby(["bucket", "market_id"], as_index=False)["yes_price"]
        .last()
        .pivot(index="bucket", columns="market_id", values="yes_price")
        .sort_index()
    )
    panel.index.name = "timestamp"
    if continuous and len(panel):
        full_index = pd.date_range(panel.index.min(), panel.index.max(), freq=freq)
        panel = panel.reindex(full_index)
        panel.index.name = "timestamp"
    return panel


def active_mask(panel: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=panel.index, columns=panel.columns)
    market_index = markets.set_index("market_id")
    for market_id in panel.columns:
        if market_id not in market_index.index:
            continue
        row = market_index.loc[market_id]
        first_observed = panel[market_id].first_valid_index()
        start = row.get("start_date")
        close = row.get("close_date")
        end = row.get("end_date")
        start_candidates = [ts for ts in [first_observed, start] if pd.notna(ts)]
        if not start_candidates:
            continue
        active_start = max(start_candidates)
        active_end = close if pd.notna(close) else end
        if pd.isna(active_start) or pd.isna(active_end):
            continue
        mask[market_id] = (panel.index >= active_start.floor("h")) & (panel.index <= active_end.ceil("h"))
    return mask


def active_ffill(panel: pd.DataFrame, markets: pd.DataFrame, fill_limit: int | None = None) -> pd.DataFrame:
    filled = panel.ffill(limit=fill_limit)
    mask = active_mask(panel, markets)
    return filled.where(mask)


def panel_metadata(panel: pd.DataFrame, freq: str, fill_policy: str) -> dict[str, object]:
    active_counts = panel.notna().sum(axis=1) if len(panel) else pd.Series(dtype=float)
    return {
        "frequency": freq,
        "fill_policy": fill_policy,
        "timestamp_count": int(panel.shape[0]),
        "market_count": int(panel.shape[1]),
        "missingness": float(panel.isna().mean().mean()) if panel.size else None,
        "timestamp_min": str(panel.index.min()) if len(panel) else None,
        "timestamp_max": str(panel.index.max()) if len(panel) else None,
        "active_market_min": int(active_counts.min()) if len(active_counts) else 0,
        "active_market_median": float(active_counts.median()) if len(active_counts) else 0,
        "active_market_max": int(active_counts.max()) if len(active_counts) else 0,
    }


def strict_panel(
    panel: pd.DataFrame,
    *,
    min_active: int,
    min_markets: int,
    max_missingness: float,
) -> pd.DataFrame:
    if panel.empty:
        return panel
    rows = panel.notna().sum(axis=1) >= min_active
    subset = panel.loc[rows].copy()
    if subset.empty:
        return subset
    ordered_cols = subset.notna().mean(axis=0).sort_values(ascending=False).index.tolist()
    best = pd.DataFrame(index=subset.index)
    for market_count in range(min_markets, len(ordered_cols) + 1):
        cols = ordered_cols[:market_count]
        candidate = subset[cols]
        candidate = candidate.loc[candidate.notna().sum(axis=1) >= min_active]
        if candidate.empty:
            continue
        if (
            candidate.isna().mean().mean() <= max_missingness
            and candidate.notna().sum(axis=1).median() >= min_active
        ):
            best = candidate
    return best if not best.empty else subset[ordered_cols[:min_markets]]


def add_optional_columns(base: pd.DataFrame, optional: pd.DataFrame, *, max_missingness: float) -> pd.DataFrame:
    if base.empty or optional.empty:
        return base
    optional = optional.reindex(base.index)
    ordered_cols = optional.notna().mean(axis=0).sort_values(ascending=False).index.tolist()
    best = base.copy()
    for col in ordered_cols:
        candidate = best.join(optional[[col]])
        if candidate.isna().mean().mean() <= max_missingness:
            best = candidate
    return best


def write_panel(panel: pd.DataFrame, path: Path, freq: str, fill_policy: str) -> dict[str, object]:
    ensure_dirs([path.parent])
    panel.to_parquet(path)
    metadata = panel_metadata(panel, freq, fill_policy)
    meta_path = path.with_suffix(".metadata.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logging.info("Saved %s shape=%s missingness=%s", path, panel.shape, metadata["missingness"])
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build analysis-ready timestamp-by-market probability panels.")
    parser.add_argument("--input", default="data/processed/prices_long.parquet")
    parser.add_argument("--markets", default="data/processed/market_universe.parquet")
    parser.add_argument("--freq", default="1h")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--fill-limit-hours", type=int, default=None)
    parser.add_argument("--min-active-core", type=int, default=15)
    parser.add_argument("--min-core-markets", type=int, default=20)
    parser.add_argument("--max-primary-missingness", type=float, default=0.20)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    root = project_root()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.is_absolute():
        input_path = root / input_path
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    ensure_dirs([output_dir])

    prices = pd.read_parquet(input_path)
    prices["market_id"] = prices["market_id"].astype(str)
    markets = load_market_universe(root, args.markets)
    selected = markets[markets["is_core"].fillna(False) | markets["is_satellite"].fillna(False)].copy()

    raw = raw_panel(prices, args.freq, continuous=True)
    filled = active_ffill(raw, selected, fill_limit=args.fill_limit_hours)

    core_ids = selected[selected["is_core"].fillna(False)]["market_id"].astype(str).tolist()
    sat_ids = selected[selected["is_satellite"].fillna(False)]["market_id"].astype(str).tolist()
    core_cols = [col for col in raw.columns if col in set(core_ids)]
    core_sat_cols = [col for col in raw.columns if col in set(core_ids + sat_ids)]

    core_panel = filled[core_cols]
    core_plus_panel = filled[core_sat_cols]
    strict_core = strict_panel(
        core_panel,
        min_active=args.min_active_core,
        min_markets=args.min_core_markets,
        max_missingness=args.max_primary_missingness,
    )
    satellite_cols = [col for col in core_plus_panel.columns if col not in strict_core.columns]
    strict_core_plus = add_optional_columns(
        strict_core,
        core_plus_panel[satellite_cols] if satellite_cols else pd.DataFrame(index=strict_core.index),
        max_missingness=args.max_primary_missingness,
    )

    all_meta = {
        "panel_hourly_raw": write_panel(raw, output_dir / "panel_hourly_raw.parquet", args.freq, "none"),
        "panel_hourly_active_ffill": write_panel(
            filled,
            output_dir / "panel_hourly_active_ffill.parquet",
            args.freq,
            f"active_ffill_limit_{args.fill_limit_hours or 'unlimited'}",
        ),
        "panel_hourly_core": write_panel(
            strict_core,
            output_dir / "panel_hourly_core.parquet",
            args.freq,
            f"strict_active_ffill_core_limit_{args.fill_limit_hours or 'unlimited'}",
        ),
        "panel_hourly_core_plus_satellites": write_panel(
            strict_core_plus,
            output_dir / "panel_hourly_core_plus_satellites.parquet",
            args.freq,
            f"strict_active_ffill_core_plus_satellites_limit_{args.fill_limit_hours or 'unlimited'}",
        ),
    }
    with (output_dir / "panel_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(all_meta, handle, indent=2)


if __name__ == "__main__":
    main()
