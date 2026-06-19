from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils import ensure_dirs, project_root


QUALITY_GATES = {
    "min_core_markets": 20,
    "min_points_per_market": 500,
    "min_calendar_days": 180,
    "min_median_active_core": 15,
    "max_primary_missingness": 0.20,
}

REQUIRED_PRICE_COLUMNS = [
    "timestamp",
    "market_id",
    "token_id",
    "yes_price",
    "category",
    "event_id",
    "resolved_outcome",
]

REQUIRED_UNIVERSE_COLUMNS = [
    "market_id",
    "condition_id",
    "question",
    "is_binary",
    "yes_token_id",
    "market_family",
    "asset",
    "threshold",
    "direction",
    "target_date",
    "is_core",
    "is_satellite",
    "selection_reason",
    "exclusion_reason",
]


def read_parquet(root: Path, path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return pd.read_parquet(p)


def market_coverage(prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    grouped = prices.groupby("market_id")
    coverage = grouped.agg(
        usable_points=("yes_price", "size"),
        timestamp_min=("timestamp", "min"),
        timestamp_max=("timestamp", "max"),
        min_price=("yes_price", "min"),
        max_price=("yes_price", "max"),
        unique_transactions=("transaction_hash", "nunique"),
    ).reset_index()
    coverage["observed_days"] = (
        pd.to_datetime(coverage["timestamp_max"], utc=True) - pd.to_datetime(coverage["timestamp_min"], utc=True)
    ).dt.total_seconds() / 86400
    cols = [
        "market_id",
        "question",
        "market_family",
        "asset",
        "is_core",
        "is_satellite",
        "resolved_outcome",
        "start_date",
        "close_date",
        "volume_clob",
    ]
    return coverage.merge(universe[[col for col in cols if col in universe.columns]], on="market_id", how="left")


def timestamp_coverage(panel: pd.DataFrame, core_panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": panel.index,
            "active_markets": panel.notna().sum(axis=1).to_numpy(),
            "missing_fraction": panel.isna().mean(axis=1).to_numpy(),
            "active_core_markets": core_panel.notna().sum(axis=1).reindex(panel.index, fill_value=0).to_numpy(),
        }
    )


def duplicate_count(prices: pd.DataFrame) -> int:
    cols = ["transaction_hash", "market_id", "timestamp", "trade_asset", "trade_size", "yes_price"]
    present = [col for col in cols if col in prices.columns]
    return int(prices.duplicated(subset=present).sum()) if present else 0


def is_utc_timestamp(series: pd.Series) -> bool:
    try:
        converted = pd.to_datetime(series, utc=False)
    except Exception:
        return False
    tz = getattr(converted.dt, "tz", None)
    return str(tz) == "UTC"


def raw_payload_inventory(raw_dir: Path) -> list[dict[str, object]]:
    files = sorted(raw_dir.glob("data_api_trades_*.jsonl"))
    inventory = []
    for path in files:
        stat = path.stat()
        inventory.append(
            {
                "file": str(path.relative_to(path.parents[1])),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return inventory


def active_window_violations(active_panel: pd.DataFrame, prices: pd.DataFrame, universe: pd.DataFrame) -> list[dict[str, object]]:
    if active_panel.empty:
        return []
    first_trade = prices.groupby("market_id")["timestamp"].min()
    market_index = universe.set_index("market_id")
    violations: list[dict[str, object]] = []
    for market_id in active_panel.columns:
        observed = active_panel.index[active_panel[market_id].notna()]
        if len(observed) == 0 or market_id not in market_index.index:
            continue
        row = market_index.loc[market_id]
        start_candidates = []
        if market_id in first_trade.index and pd.notna(first_trade.loc[market_id]):
            start_candidates.append(first_trade.loc[market_id].floor("h"))
        for col in ["start_date"]:
            value = row.get(col)
            if pd.notna(value):
                start_candidates.append(pd.to_datetime(value, utc=True).floor("h"))
        end_candidates = []
        for col in ["close_date", "end_date"]:
            value = row.get(col)
            if pd.notna(value):
                end_candidates.append(pd.to_datetime(value, utc=True).ceil("h"))
                break
        active_start = max(start_candidates) if start_candidates else None
        active_end = end_candidates[0] if end_candidates else None
        if active_start is not None and observed.min() < active_start:
            violations.append({"market_id": market_id, "violation": "value_before_active_start", "timestamp": str(observed.min())})
        if active_end is not None and observed.max() > active_end:
            violations.append({"market_id": market_id, "violation": "value_after_active_end", "timestamp": str(observed.max())})
    return violations


def max_staleness_hours(active_panel: pd.DataFrame, raw_panel: pd.DataFrame) -> float | None:
    if active_panel.empty or raw_panel.empty:
        return None
    max_gap = 0.0
    for market_id in active_panel.columns:
        if market_id not in raw_panel.columns:
            continue
        raw_observed = raw_panel.index[raw_panel[market_id].notna()]
        active_observed = active_panel.index[active_panel[market_id].notna()]
        if len(raw_observed) == 0 or len(active_observed) == 0:
            continue
        raw_series = pd.Series(raw_observed, index=raw_observed)
        last_raw = raw_series.reindex(active_observed, method="ffill")
        gaps = (active_observed.to_series(index=active_observed) - last_raw).dt.total_seconds() / 3600
        if len(gaps):
            max_gap = max(max_gap, float(gaps.max()))
    return max_gap


def validate(root: Path, processed_dir: Path) -> dict[str, object]:
    universe = read_parquet(root, str(processed_dir / "market_universe.parquet"))
    prices = read_parquet(root, str(processed_dir / "prices_long.parquet"))
    raw = read_parquet(root, str(processed_dir / "panel_hourly_raw.parquet"))
    active = read_parquet(root, str(processed_dir / "panel_hourly_active_ffill.parquet"))
    primary = read_parquet(root, str(processed_dir / "panel_hourly_core.parquet"))
    core_plus = read_parquet(root, str(processed_dir / "panel_hourly_core_plus_satellites.parquet"))

    universe["market_id"] = universe["market_id"].astype(str)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    prices["market_id"] = prices["market_id"].astype(str)
    for panel in [raw, active, primary, core_plus]:
        panel.index = pd.to_datetime(panel.index, utc=True)
        panel.columns = panel.columns.astype(str)
    for col in ["start_date", "close_date", "end_date"]:
        if col in universe.columns:
            universe[col] = pd.to_datetime(universe[col], utc=True, errors="coerce")
    selected = universe[universe["is_core"].fillna(False) | universe["is_satellite"].fillna(False)].copy()
    core = universe[universe["is_core"].fillna(False)].copy()

    coverage_market = market_coverage(prices, universe)
    coverage_time = timestamp_coverage(core_plus, primary)
    coverage_market.to_csv(processed_dir / "coverage_by_market.csv", index=False)
    coverage_time.to_csv(processed_dir / "coverage_by_timestamp.csv", index=False)

    per_market_points = prices.groupby("market_id").size()
    raw_timestamp_span_days = (prices["timestamp"].max() - prices["timestamp"].min()).total_seconds() / 86400 if len(prices) else 0
    primary_usable_days = len(primary) / 24 if len(primary) else 0
    median_active_core = float(primary.notna().sum(axis=1).median()) if len(primary) else 0
    primary_missingness = float(primary.isna().mean().mean()) if primary.size else 1.0
    max_points_per_market = int(per_market_points.max()) if len(per_market_points) else 0
    markets_at_max_points = int((per_market_points == max_points_per_market).sum()) if len(per_market_points) else 0
    unresolved_selected = int(selected["resolved_outcome"].isna().sum()) if "resolved_outcome" in selected.columns else len(selected)
    non_binary_selected = int((~selected["is_binary"].fillna(False)).sum()) if "is_binary" in selected.columns else len(selected)
    missing_price_columns = [col for col in REQUIRED_PRICE_COLUMNS if col not in prices.columns]
    missing_universe_columns = [col for col in REQUIRED_UNIVERSE_COLUMNS if col not in universe.columns]
    required_price_nulls = {
        col: int(prices[col].isna().sum())
        for col in ["timestamp", "market_id", "token_id", "yes_price"]
        if col in prices.columns
    }
    active_violations = active_window_violations(active, prices, universe)
    max_stale_hours = max_staleness_hours(active, raw)
    excluded_without_reason = int(
        universe[
            ~(universe["is_core"].fillna(False) | universe["is_satellite"].fillna(False))
            & universe["exclusion_reason"].isna()
        ].shape[0]
    )
    gates = {
        "min_core_markets": int(core["market_id"].nunique()) >= QUALITY_GATES["min_core_markets"],
        "min_points_per_market": bool((per_market_points >= QUALITY_GATES["min_points_per_market"]).all()) if len(per_market_points) else False,
        "min_calendar_days": primary_usable_days >= QUALITY_GATES["min_calendar_days"],
        "min_median_active_core": median_active_core >= QUALITY_GATES["min_median_active_core"],
        "max_primary_missingness": primary_missingness <= QUALITY_GATES["max_primary_missingness"],
        "no_unresolved_selected": unresolved_selected == 0,
        "no_non_binary_selected": non_binary_selected == 0,
        "required_columns_present": not missing_price_columns and not missing_universe_columns,
        "required_nulls_absent": all(count == 0 for count in required_price_nulls.values()),
        "timestamps_utc": is_utc_timestamp(prices["timestamp"]),
        "price_bounds": bool(prices["yes_price"].between(0, 1).all()) if len(prices) else False,
        "active_window_respected": len(active_violations) == 0,
        "excluded_reasons_complete": excluded_without_reason == 0,
        "core_and_satellite_panels_exist": (processed_dir / "panel_hourly_core.parquet").exists()
        and (processed_dir / "panel_hourly_core_plus_satellites.parquet").exists(),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_ready": bool(all(gates.values())),
        "quality_gates": gates,
        "thresholds": QUALITY_GATES,
        "counts": {
            "metadata_markets": int(len(universe)),
            "selected_markets": int(len(selected)),
            "core_markets": int(core["market_id"].nunique()),
            "satellite_markets": int(selected["is_satellite"].fillna(False).sum()),
            "price_rows": int(len(prices)),
            "price_markets": int(prices["market_id"].nunique()) if len(prices) else 0,
            "duplicate_rows": duplicate_count(prices),
            "unresolved_selected": unresolved_selected,
            "non_binary_selected": non_binary_selected,
            "excluded_without_reason": excluded_without_reason,
            "active_window_violations": len(active_violations),
        },
        "coverage": {
            "timestamp_min": str(prices["timestamp"].min()) if len(prices) else None,
            "timestamp_max": str(prices["timestamp"].max()) if len(prices) else None,
            "raw_trade_calendar_days": raw_timestamp_span_days,
            "primary_usable_days": primary_usable_days,
            "primary_panel_shape": list(primary.shape),
            "core_plus_panel_shape": list(core_plus.shape),
            "primary_missingness": primary_missingness,
            "median_active_core_markets": median_active_core,
            "max_points_per_market": max_points_per_market,
            "markets_at_max_points": markets_at_max_points,
            "max_active_ffill_staleness_hours": max_stale_hours,
        },
        "schema": {
            "missing_price_columns": missing_price_columns,
            "missing_universe_columns": missing_universe_columns,
            "required_price_nulls": required_price_nulls,
            "prices_timestamp_dtype": str(prices["timestamp"].dtype),
            "panel_timestamp_dtype": str(active.index.dtype),
        },
        "active_window_sample_violations": active_violations[:20],
        "raw_payloads": {
            "data_api_trades": raw_payload_inventory(root / "data" / "raw"),
        },
        "market_counts_by_family": selected["market_family"].fillna("unknown").value_counts().to_dict(),
        "limitations": [],
    }
    if not report["analysis_ready"]:
        report["limitations"].append("One or more strict quality gates failed; inspect quality_gates and coverage outputs.")
    if not gates["min_calendar_days"]:
        report["limitations"].append(
            "The raw trades span more calendar time than the strict hourly panel because simultaneous active BTC/ETH coverage is sparse under the current public trade-history pull."
        )
    if markets_at_max_points >= max(1, int(0.5 * len(per_market_points))):
        report["limitations"].append(
            f"{markets_at_max_points} markets hit the observed per-market trade fetch cap of {max_points_per_market}; deeper historical pagination or an archival source is needed for lifetime histories."
        )
    with (processed_dir / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    manifest = {
        "generated_at": report["generated_at"],
        "api_sources": ["gamma-api.polymarket.com", "data-api.polymarket.com"],
        "parameters": {
            "frequency": "1h",
            "fill_policy": "active_window_forward_fill",
            "trade_fetch": {
                "endpoint": "data-api.polymarket.com/trades",
                "observed_per_market_cap": max_points_per_market,
                "markets_at_observed_cap": markets_at_max_points,
                "note": "Public pagination returned a max historical activity offset error beyond offset 3000 for high-volume markets; page_size=1000 yields up to 4000 trades per market.",
            },
            "quality_gates": QUALITY_GATES,
        },
        "selected_market_ids": selected["market_id"].astype(str).tolist(),
        "raw_payloads": report["raw_payloads"],
        "validation_status": report["analysis_ready"],
    }
    with (processed_dir / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate analysis-ready Polymarket crypto dataset.")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()
    root = project_root()
    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    ensure_dirs([processed_dir])
    report = validate(root, processed_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
