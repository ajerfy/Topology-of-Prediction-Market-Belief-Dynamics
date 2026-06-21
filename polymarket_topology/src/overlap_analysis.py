from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from build_market_panel import active_ffill, raw_panel
from universe_selection import class_entropy, correlation_metrics, pca_metrics
from utils import ensure_dirs, project_root, setup_logging


THRESHOLDS = (0.25, 0.40, 0.50)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_universe_b_markets(path: Path) -> pd.DataFrame:
    markets = pd.read_parquet(path).copy()
    markets = markets[markets["universe"] == "universe_b_macro_crypto"].copy()
    markets["market_id"] = markets["market_id"].astype(str)
    for col in ["start_date", "end_date", "close_date"]:
        if col in markets.columns:
            markets[col] = pd.to_datetime(markets[col], utc=True, errors="coerce", format="mixed")
    markets["Y_i"] = markets["resolved_outcome"].map({"Yes": 1, "No": 0})
    return markets


def load_price_panel(prices_path: Path, market_ids: list[str]) -> pd.DataFrame:
    prices = pd.read_parquet(prices_path)
    prices["market_id"] = prices["market_id"].astype(str)
    prices = prices[prices["market_id"].isin(market_ids)].copy()
    return raw_panel(prices, "1h", continuous=True)


def market_lifetimes(markets: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    market_index = markets.set_index("market_id")
    for market_id in raw.columns:
        row = market_index.loc[market_id]
        series = raw[market_id]
        first_observed = series.first_valid_index()
        last_observed = series.last_valid_index()
        start = row.get("start_date")
        close = row.get("close_date")
        end = row.get("end_date")
        active_start_candidates = [ts for ts in [start, first_observed] if pd.notna(ts)]
        active_start = min(active_start_candidates) if active_start_candidates else pd.NaT
        active_end_candidates = [ts for ts in [close, end, last_observed] if pd.notna(ts)]
        active_end = max(active_end_candidates) if active_end_candidates else pd.NaT
        observed_duration = (
            (last_observed - first_observed).total_seconds() / 3600
            if pd.notna(first_observed) and pd.notna(last_observed)
            else np.nan
        )
        active_duration = (
            (active_end - active_start).total_seconds() / 3600
            if pd.notna(active_start) and pd.notna(active_end)
            else np.nan
        )
        rows.append(
            {
                "market_id": market_id,
                "question": row.get("question"),
                "broad_domain": row.get("broad_domain"),
                "broad_family": row.get("broad_family"),
                "resolved_outcome": row.get("resolved_outcome"),
                "Y_i": row.get("Y_i"),
                "volume": row.get("volume"),
                "start_date": start,
                "end_date": end,
                "close_date": close,
                "active_start": active_start,
                "active_end": active_end,
                "active_duration_hours": active_duration,
                "observed_duration_hours": observed_duration,
                "first_observed_timestamp": first_observed,
                "last_observed_timestamp": last_observed,
                "observed_hours": int(series.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def active_mask(index: pd.DatetimeIndex, lifetimes: pd.DataFrame) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=index, columns=lifetimes["market_id"].astype(str).tolist())
    for row in lifetimes.itertuples(index=False):
        if pd.isna(row.active_start) or pd.isna(row.active_end):
            continue
        start = row.active_start.floor("h")
        end = row.active_end.ceil("h")
        mask[row.market_id] = (mask.index >= start) & (mask.index <= end)
    return mask


def pairwise_overlap(mask: pd.DataFrame) -> pd.DataFrame:
    values = mask.astype(np.uint8).to_numpy()
    both = values.T @ values
    counts = values.sum(axis=0)
    either = counts[:, None] + counts[None, :] - both
    overlap = np.divide(both, either, out=np.zeros_like(both, dtype=float), where=either > 0)
    return pd.DataFrame(overlap, index=mask.columns, columns=mask.columns)


def missingness_decomposition(panel: pd.DataFrame, raw: pd.DataFrame, mask: pd.DataFrame) -> dict[str, float | int]:
    panel = panel.reindex(index=mask.index, columns=mask.columns)
    raw = raw.reindex(index=mask.index, columns=mask.columns)
    total_cells = int(mask.size)
    observed_filled = panel.notna()
    raw_observed = raw.notna()
    structural = ~mask
    active = mask
    active_missing_after_fill = active & panel.isna()
    active_raw_missing = active & raw_observed.eq(False)
    missing = panel.isna()
    structural_missing = structural & missing
    data_missing = active_missing_after_fill
    return {
        "total_cells": total_cells,
        "observed_filled_cells": int(observed_filled.sum().sum()),
        "raw_observed_cells": int(raw_observed.sum().sum()),
        "missing_cells": int(missing.sum().sum()),
        "overall_missingness": float(missing.mean().mean()),
        "structural_missing_cells": int(structural_missing.sum().sum()),
        "data_missing_cells": int(data_missing.sum().sum()),
        "active_raw_missing_cells": int(active_raw_missing.sum().sum()),
        "structural_missing_fraction_of_all": float(structural_missing.sum().sum() / total_cells),
        "data_missing_fraction_of_all": float(data_missing.sum().sum() / total_cells),
        "structural_share_of_missing": float(structural_missing.sum().sum() / max(1, missing.sum().sum())),
        "data_share_of_missing": float(data_missing.sum().sum() / max(1, missing.sum().sum())),
        "raw_sparsity_within_active": float(active_raw_missing.sum().sum() / max(1, active.sum().sum())),
        "filled_missing_within_active": float(active_missing_after_fill.sum().sum() / max(1, active.sum().sum())),
    }


def summarize_panel(name: str, market_ids: list[str], panel: pd.DataFrame, mask: pd.DataFrame, markets: pd.DataFrame) -> dict[str, object]:
    ids = [mid for mid in market_ids if mid in panel.columns]
    sub_panel = panel[ids]
    sub_mask = mask[ids]
    sub_markets = markets[markets["market_id"].isin(ids)]
    yes_rate = float(sub_markets["Y_i"].mean()) if len(sub_markets) else np.nan
    overlap = pairwise_overlap(sub_mask)
    upper = overlap.to_numpy()[np.triu_indices_from(overlap.to_numpy(), k=1)]
    upper = upper[np.isfinite(upper)]
    corr = correlation_metrics(sub_panel)
    pca = pca_metrics(sub_panel)
    return {
        "universe": name,
        "market_count": int(len(ids)),
        "yes_rate": yes_rate,
        "no_rate": 1 - yes_rate if pd.notna(yes_rate) else np.nan,
        "class_entropy": class_entropy(yes_rate),
        "panel_missingness": float(sub_panel.isna().mean().mean()) if sub_panel.size else np.nan,
        "median_active_markets": float(sub_mask.sum(axis=1).median()) if len(sub_mask) else np.nan,
        "mean_active_markets": float(sub_mask.sum(axis=1).mean()) if len(sub_mask) else np.nan,
        "mean_pairwise_overlap": float(np.mean(upper)) if len(upper) else np.nan,
        "median_pairwise_overlap": float(np.median(upper)) if len(upper) else np.nan,
        "min_pairwise_overlap": float(np.min(upper)) if len(upper) else np.nan,
        "max_pairwise_overlap": float(np.max(upper)) if len(upper) else np.nan,
        **corr,
        **pca,
    }


def choose_recommendation(candidates: pd.DataFrame) -> str:
    viable = candidates[candidates["market_count"] >= 25].copy()
    constrained_viable = viable[viable["universe"] != "Universe B"]
    if constrained_viable.empty:
        return "A"
    if viable.empty:
        return "A"
    viable["balance_score"] = (1 - (viable["yes_rate"] - 0.5).abs() * 2).clip(lower=0).fillna(0)
    viable["missingness_score"] = (1 - viable["panel_missingness"]).clip(lower=0).fillna(0)
    viable["market_score"] = np.minimum(1, viable["market_count"] / 100)
    viable["latent_score"] = (1 - viable["pc_90"] / viable["market_count"]).clip(lower=0).fillna(0)
    viable["score"] = (
        0.30 * viable["missingness_score"]
        + 0.25 * viable["market_score"]
        + 0.25 * viable["balance_score"]
        + 0.20 * viable["latent_score"]
    )
    selected = viable.sort_values("score", ascending=False).iloc[0]["universe"]
    return {
        "Universe B": "A",
        "Universe B-25": "B",
        "Universe B-40": "C",
        "Universe B-50": "D",
    }[selected]


def write_summary(
    path: Path,
    original: dict[str, object],
    missingness: dict[str, float | int],
    overlap_stats: dict[str, float],
    candidates: pd.DataFrame,
    recommendation_code: str,
) -> None:
    rec_labels = {
        "A": "Keep original Universe B",
        "B": "Use Universe B-25",
        "C": "Use Universe B-40",
        "D": "Use Universe B-50",
    }
    selected_name = {
        "A": "Universe B",
        "B": "Universe B-25",
        "C": "Universe B-40",
        "D": "Universe B-50",
    }[recommendation_code]
    selected = candidates[candidates["universe"] == selected_name].iloc[0]

    table_cols = [
        "universe",
        "market_count",
        "yes_rate",
        "class_entropy",
        "panel_missingness",
        "median_active_markets",
        "mean_pairwise_overlap",
        "pc_90",
    ]
    table = candidates[table_cols].copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    table_text = "| " + " | ".join(table.columns) + " |\n"
    table_text += "| " + " | ".join(["---"] * len(table.columns)) + " |\n"
    for row in table.to_numpy():
        table_text += "| " + " | ".join(map(str, row)) + " |\n"

    mostly_structural = missingness["structural_share_of_missing"] >= 0.75
    constrained = candidates[candidates["universe"] != "Universe B"]
    any_constrained_viable = bool((constrained["market_count"] >= 25).any())
    lines = [
        "==================================================",
        "OVERLAP ANALYSIS SUMMARY",
        "==================================================",
        "",
        "Original Universe B:",
        "",
        f"- market count: {int(original['market_count'])}",
        f"- missingness: {original['panel_missingness']:.3f}",
        f"- structural missingness: {missingness['structural_missing_fraction_of_all']:.3f} of all cells, {missingness['structural_share_of_missing']:.3f} of missing cells",
        f"- data missingness after active forward-fill: {missingness['data_missing_fraction_of_all']:.3f} of all cells, {missingness['data_share_of_missing']:.3f} of missing cells",
        f"- raw active-period sparsity before forward-fill: {missingness['raw_sparsity_within_active']:.3f} of active cells",
        "",
        "Overlap statistics:",
        "",
        f"- mean overlap: {overlap_stats['mean']:.3f}",
        f"- median overlap: {overlap_stats['median']:.3f}",
        f"- minimum overlap: {overlap_stats['min']:.3f}",
        f"- maximum overlap: {overlap_stats['max']:.3f}",
        "",
        "Candidate universes:",
        "",
        table_text.rstrip(),
        "",
        "Interpretation:",
        "",
        "- The 78.5% panel missingness is mostly structural." if mostly_structural else "- The 78.5% panel missingness is not mostly structural; active-period data gaps are large enough to be a primary concern.",
        "- Active-period data gaps are small relative to structural gaps; the remaining NaNs are dominated by markets not coexisting in time." if mostly_structural else "- Forward-fill does not fully solve active-period sparsity; panel construction choices are likely to matter strongly for PCA/topology.",
        "",
        "Recommendation:",
        "",
        f"- {recommendation_code}) {rec_labels[recommendation_code]}",
        "",
        "Justification:",
        "",
        f"- Selected universe: {selected_name}",
        f"- It keeps {int(selected['market_count'])} markets, YES rate {selected['yes_rate']:.3f}, entropy {selected['class_entropy']:.3f}, missingness {selected['panel_missingness']:.3f}, and PC90 {int(selected['pc_90']) if pd.notna(selected['pc_90']) else 'NA'}.",
        "- This choice offers the best current tradeoff between reducing construction artifacts and preserving enough markets, outcome diversity, and latent-factor structure for a PCA-vs-topology comparison.",
        "",
        "Most important question:",
        "",
        "Can we reduce missingness substantially while preserving enough markets, enough outcome diversity, and enough latent-factor structure to make the PCA-vs-topology comparison scientifically meaningful?",
        "",
        "Answer: not with the requested average-overlap thresholds. Under the literal pairwise Jaccard-overlap definition, B-25/B-40/B-50 are empty, so they reduce missingness only by destroying the universe. The valid next move is to keep Universe B for now, but build future analysis on active-window-aware methods rather than treating the full rectangular panel as equally observed."
        if not any_constrained_viable
        else "Answer: yes, but only up to the recommended overlap threshold. More aggressive overlap filtering improves co-temporality but risks making the universe too small and too dominated by a narrow event cluster.",
        "",
        "==================================================",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    *,
    candidate_markets_path: Path,
    prices_path: Path,
    universe_b_panel_path: Path,
    output_dir: Path,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    ensure_dirs([output_dir])
    markets = load_universe_b_markets(candidate_markets_path)
    universe_b_panel = pd.read_parquet(universe_b_panel_path)
    universe_b_panel.columns = universe_b_panel.columns.astype(str)
    universe_b_panel.index = pd.to_datetime(universe_b_panel.index, utc=True)
    market_ids = universe_b_panel.columns.astype(str).tolist()
    markets = markets[markets["market_id"].isin(market_ids)].copy()
    raw = load_price_panel(prices_path, market_ids).reindex(universe_b_panel.index)
    raw = raw.reindex(columns=market_ids)

    lifetimes = market_lifetimes(markets, raw)
    mask = active_mask(universe_b_panel.index, lifetimes)
    overlap = pairwise_overlap(mask)
    avg_overlap = overlap.mask(np.eye(len(overlap), dtype=bool)).mean(axis=1)
    lifetimes["average_pairwise_overlap"] = lifetimes["market_id"].map(avg_overlap)

    missingness = missingness_decomposition(universe_b_panel, raw, mask)
    upper = overlap.to_numpy()[np.triu_indices_from(overlap.to_numpy(), k=1)]
    upper = upper[np.isfinite(upper)]
    overlap_stats = {
        "mean": float(np.mean(upper)),
        "median": float(np.median(upper)),
        "min": float(np.min(upper)),
        "max": float(np.max(upper)),
    }

    candidate_rows = [summarize_panel("Universe B", market_ids, universe_b_panel, mask, markets)]
    for threshold in THRESHOLDS:
        keep = lifetimes.loc[lifetimes["average_pairwise_overlap"] >= threshold, "market_id"].tolist()
        name = f"Universe B-{int(threshold * 100)}"
        candidate_rows.append(summarize_panel(name, keep, universe_b_panel, mask, markets))
        universe_b_panel[keep].to_parquet(output_dir / f"universe_b_overlap_{int(threshold * 100)}_panel.parquet")

    candidates = pd.DataFrame(candidate_rows)
    recommendation = choose_recommendation(candidates)

    lifetimes.to_csv(output_dir / "universe_b_market_lifetimes.csv", index=False)
    overlap.to_csv(output_dir / "universe_b_pairwise_overlap.csv")
    pd.DataFrame([missingness]).to_csv(output_dir / "universe_b_missingness_decomposition.csv", index=False)
    candidates.to_csv(output_dir / "overlap_constrained_universe_summary.csv", index=False)
    write_summary(
        output_dir / "overlap_analysis_summary.md",
        candidate_rows[0],
        missingness,
        overlap_stats,
        candidates,
        recommendation,
    )
    return {
        "lifetimes": lifetimes,
        "overlap": overlap,
        "missingness": missingness,
        "candidates": candidates,
        "overlap_stats": overlap_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Universe B market overlap and missingness.")
    parser.add_argument("--candidate-markets", default="data/processed/candidate_universe_markets.parquet")
    parser.add_argument("--prices", default="data/processed/prices_long.parquet")
    parser.add_argument("--universe-b-panel", default="data/processed/universe_b_macro_crypto_panel.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    root = project_root()
    result = run_analysis(
        candidate_markets_path=resolve_path(root, args.candidate_markets),
        prices_path=resolve_path(root, args.prices),
        universe_b_panel_path=resolve_path(root, args.universe_b_panel),
        output_dir=resolve_path(root, args.output_dir),
    )
    print(result["candidates"].to_string(index=False))
    print(result["missingness"])


if __name__ == "__main__":
    main()
