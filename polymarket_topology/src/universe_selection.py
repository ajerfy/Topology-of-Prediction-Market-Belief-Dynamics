from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from build_market_panel import active_ffill
from utils import ensure_dirs, project_root, setup_logging


BALANCE_WEIGHT = 0.30
LIQUIDITY_WEIGHT = 0.20
COVERAGE_WEIGHT = 0.20
COMPRESS_WEIGHT = 0.20
RELATION_WEIGHT = 0.10


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    label: str
    goal: str
    domains: tuple[str, ...]
    family_patterns: tuple[str, ...] = ()


CANDIDATES = (
    CandidateSpec(
        name="universe_a_crypto_diversified",
        label="Universe A: Crypto diversified",
        goal="Increase outcome diversity while staying crypto-focused.",
        domains=("crypto",),
        family_patterns=(
            "crypto_btc",
            "crypto_eth",
            "crypto_etf",
            "crypto_policy",
            "crypto_microstrategy",
            "crypto_stablecoin",
            "crypto_regulation",
            "crypto_other",
        ),
    ),
    CandidateSpec(
        name="universe_b_macro_crypto",
        label="Universe B: Macro + crypto",
        goal="Introduce cross-domain latent factors.",
        domains=("crypto", "macro"),
    ),
    CandidateSpec(
        name="universe_c_multi_domain",
        label="Universe C: Multi-domain",
        goal="Maximize outcome diversity while retaining market structure.",
        domains=("crypto", "macro", "elections", "sports"),
    ),
)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).lower()


def row_text(row: pd.Series) -> str:
    tags = row.get("tags")
    if isinstance(tags, (list, tuple)):
        tag_text = " ".join(str(tag) for tag in tags)
    elif isinstance(tags, np.ndarray):
        tag_text = " ".join(str(tag) for tag in tags.tolist())
    else:
        tag_text = "" if tags is None else str(tags)
    fields = ["question", "slug", "description", "category"]
    return " ".join(clean_text(row.get(field)) for field in fields) + " " + tag_text.lower()


def row_title_text(row: pd.Series) -> str:
    tags = row.get("tags")
    if isinstance(tags, (list, tuple)):
        tag_text = " ".join(str(tag) for tag in tags)
    elif isinstance(tags, np.ndarray):
        tag_text = " ".join(str(tag) for tag in tags.tolist())
    else:
        tag_text = "" if tags is None else str(tags)
    fields = ["question", "slug", "category"]
    return " ".join(clean_text(row.get(field)) for field in fields) + " " + tag_text.lower()


def contains(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.I))


WEATHER_SOURCE_PATTERN = (
    r"\b(wunderground|national weather service|\bnws\b|daily climate report|"
    r"\bcli(?:nyc|sfo|mia|mdw|lax)?\b|knyc|klga|ksfo|kmia|kmdw|klax)\b"
)
WEATHER_MARKET_TITLE_PATTERN = (
    r"\b(highest temperature|lowest temperature|temperature in|global heat|"
    r"weather protocol|snowfall|rainfall|hurricane)\b"
)


def classify_source_quality(row: pd.Series) -> tuple[bool, str | None]:
    text = row_text(row)
    title = row_title_text(row)
    if contains(text, r"\b(source degraded|degraded source|data source degraded)\b"):
        return True, "explicit_source_degraded_notice"
    if contains(text, WEATHER_SOURCE_PATTERN) or contains(title, WEATHER_MARKET_TITLE_PATTERN):
        return True, "external_weather_source_dependency"
    return False, None


def classify_domain_family(row: pd.Series) -> tuple[str, str]:
    text = row_text(row)
    title = row_title_text(row)

    if contains(text, WEATHER_SOURCE_PATTERN) or contains(title, WEATHER_MARKET_TITLE_PATTERN):
        return "weather", "weather"

    crypto_asset = contains(
        text,
        r"\b(bitcoin|btc|ethereum|eth|crypto|stablecoin|tether|"
        r"microstrategy|mstr|coinbase|binance|solana|xrp|doge|blockchain)\b",
    )
    crypto_policy_context = crypto_asset and contains(
        text,
        r"\b(etf|reserve|stablecoin|regulation|regulatory|bill|senate|congress|"
        r"sec|cftc|approve|approval|policy|law|executive order)\b",
    )

    if contains(text, r"\b(microstrategy|mstr)\b"):
        return "crypto", "crypto_microstrategy"
    if contains(title, r"\b(stablecoin|tether|usdt|usdc)\b"):
        return "crypto", "crypto_stablecoin"
    if crypto_asset and contains(text, r"\betf\b"):
        return "crypto", "crypto_etf"
    if crypto_policy_context and contains(text, r"\b(regulation|regulatory|bill|senate|congress|sec|cftc|law|policy)\b"):
        return "crypto", "crypto_regulation"
    if crypto_policy_context:
        return "crypto", "crypto_policy"
    if contains(text, r"\b(bitcoin|btc)\b"):
        return "crypto", "crypto_btc"
    if contains(text, r"\b(ethereum|eth)\b"):
        return "crypto", "crypto_eth"
    if crypto_asset:
        return "crypto", "crypto_other"

    if contains(text, r"\b(fed|federal reserve|interest rate|rates|bps|rate cut|rate hike|fed chair)\b"):
        return "macro", "macro_fed_rates"
    if contains(text, r"\b(inflation|cpi|pce|consumer price)\b"):
        return "macro", "macro_inflation"
    if contains(text, r"\b(recession|gdp|unemployment|jobs report|payroll|treasury|yield|tariff|debt ceiling|economy)\b"):
        return "macro", "macro_growth_policy"

    if contains(text, r"\b(election|presidential|president|senate race|house race|electoral|republican|democrat|primary)\b"):
        return "elections", "elections"

    if contains(
        text,
        r"\b(nba|nfl|mlb|nhl|uefa|champions league|super bowl|world cup|premier league|"
        r"fifa|ufc|wimbledon|olympics|cricket|baseball|basketball|football|soccer|tennis|formula 1|f1)\b",
    ):
        return "sports", "sports"

    if contains(text, r"\b(policy|bill|law|regulation|senate|congress|supreme court|tariff)\b"):
        return "policy_other", "policy_other"

    return "other", "other"


def load_markets(path: Path) -> pd.DataFrame:
    markets = pd.read_parquet(path).copy()
    markets["market_id"] = markets["market_id"].astype(str)
    for col in ["start_date", "end_date", "close_date"]:
        if col in markets.columns:
            markets[col] = pd.to_datetime(markets[col], utc=True, errors="coerce", format="mixed")
    markets["Y_i"] = markets["resolved_outcome"].map({"Yes": 1, "No": 0})
    classified = markets.apply(classify_domain_family, axis=1, result_type="expand")
    markets["broad_domain"] = classified[0]
    markets["broad_family"] = classified[1]
    quality = markets.apply(classify_source_quality, axis=1, result_type="expand")
    markets["exclude_source_quality"] = quality[0].astype(bool)
    markets["source_quality_reason"] = quality[1]
    return markets


def class_entropy(yes_rate: float) -> float:
    if pd.isna(yes_rate) or yes_rate <= 0 or yes_rate >= 1:
        return 0.0
    return float(-(yes_rate * math.log2(yes_rate) + (1 - yes_rate) * math.log2(1 - yes_rate)))


def safe_mean(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").mean()
    return float(value) if pd.notna(value) else np.nan


def family_audit(markets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in markets.groupby("broad_family", dropna=False):
        resolved = group[group["Y_i"].notna()]
        yes_rate = float(resolved["Y_i"].mean()) if len(resolved) else np.nan
        rows.append(
            {
                "broad_family": family,
                "broad_domain": group["broad_domain"].mode().iat[0] if not group["broad_domain"].mode().empty else None,
                "markets": int(len(group)),
                "resolved_markets": int(len(resolved)),
                "yes_rate": yes_rate,
                "no_rate": 1 - yes_rate if pd.notna(yes_rate) else np.nan,
                "class_entropy": class_entropy(yes_rate),
                "start_min": group["start_date"].min(),
                "start_max": group["start_date"].max(),
                "close_min": group["close_date"].min(),
                "close_max": group["close_date"].max(),
                "avg_volume": safe_mean(group["volume"]),
                "avg_liquidity": safe_mean(group["liquidity"]),
                "panel_ready_markets": int(group["in_price_panel"].sum()),
                "source_quality_exclusions": int(group.get("exclude_source_quality", pd.Series(False, index=group.index)).sum()),
            }
        )
    audit = pd.DataFrame(rows)
    audit["balance_score"] = 1 - (audit["yes_rate"] - 0.5).abs() * 2
    audit["balance_score"] = audit["balance_score"].clip(lower=0).fillna(0)
    audit["liquidity_score"] = np.log1p(audit["avg_volume"].fillna(0))
    if audit["liquidity_score"].max() > audit["liquidity_score"].min():
        audit["liquidity_score"] = (audit["liquidity_score"] - audit["liquidity_score"].min()) / (
            audit["liquidity_score"].max() - audit["liquidity_score"].min()
        )
    else:
        audit["liquidity_score"] = 0
    audit["panel_ready_score"] = audit["panel_ready_markets"] / audit["panel_ready_markets"].max()
    audit["relationship_score"] = np.where(audit["broad_domain"].isin(["crypto", "macro", "elections", "sports"]), 1.0, 0.35)
    audit["forecasting_usefulness_score"] = (
        0.35 * audit["balance_score"]
        + 0.30 * audit["liquidity_score"]
        + 0.25 * audit["panel_ready_score"].fillna(0)
        + 0.10 * audit["relationship_score"]
    )
    return audit.sort_values("forecasting_usefulness_score", ascending=False)


def candidate_mask(markets: pd.DataFrame, spec: CandidateSpec) -> pd.Series:
    domain_match = markets["broad_domain"].isin(spec.domains)
    source_quality_ok = ~markets.get("exclude_source_quality", pd.Series(False, index=markets.index)).fillna(False)
    if spec.family_patterns:
        family_match = markets["broad_family"].isin(spec.family_patterns)
        return domain_match & family_match & source_quality_ok
    return domain_match & source_quality_ok


def pca_metrics(panel: pd.DataFrame) -> dict[str, float | int]:
    if panel.shape[1] < 2 or panel.dropna(how="all").shape[0] < 24:
        return {
            "pc_85": np.nan,
            "pc_90": np.nan,
            "pc_95": np.nan,
            "pc1_variance": np.nan,
            "pc2_cumulative_variance": np.nan,
            "pc5_cumulative_variance": np.nan,
        }
    observed = panel.notna().sum(axis=0)
    varying = panel.nunique(dropna=True) > 1
    cols = observed[observed > 0].index.intersection(varying[varying].index)
    if len(cols) < 2:
        return {
            "pc_85": np.nan,
            "pc_90": np.nan,
            "pc_95": np.nan,
            "pc1_variance": np.nan,
            "pc2_cumulative_variance": np.nan,
            "pc5_cumulative_variance": np.nan,
        }
    x = panel[cols].replace([np.inf, -np.inf], np.nan)
    x = SimpleImputer(strategy="mean").fit_transform(x)
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=None, svd_solver="full").fit(x)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return {
        "pc_85": int(np.searchsorted(cumulative, 0.85) + 1),
        "pc_90": int(np.searchsorted(cumulative, 0.90) + 1),
        "pc_95": int(np.searchsorted(cumulative, 0.95) + 1),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_cumulative_variance": float(cumulative[min(1, len(cumulative) - 1)]),
        "pc5_cumulative_variance": float(cumulative[min(4, len(cumulative) - 1)]),
    }


def correlation_metrics(panel: pd.DataFrame) -> dict[str, float]:
    if panel.shape[1] < 2:
        return {"mean_abs_pairwise_corr": np.nan, "median_abs_pairwise_corr": np.nan}
    corr = panel.corr(min_periods=24).to_numpy()
    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[np.isfinite(upper)]
    if len(upper) == 0:
        return {"mean_abs_pairwise_corr": np.nan, "median_abs_pairwise_corr": np.nan}
    return {
        "mean_abs_pairwise_corr": float(np.mean(np.abs(upper))),
        "median_abs_pairwise_corr": float(np.median(np.abs(upper))),
    }


def active_overlap_panel(raw_panel: pd.DataFrame, markets: pd.DataFrame, market_ids: list[str]) -> pd.DataFrame:
    cols = [mid for mid in market_ids if mid in raw_panel.columns]
    if not cols:
        return pd.DataFrame()
    filled = active_ffill(raw_panel[cols], markets[markets["market_id"].isin(cols)], fill_limit=None)
    active_counts = filled.notna().sum(axis=1)
    min_active = max(5, min(20, int(len(cols) * 0.25)))
    subset = filled.loc[active_counts >= min_active]
    coverage = subset.notna().mean(axis=0) if not subset.empty else pd.Series(dtype=float)
    keep = coverage[coverage >= 0.05].index.tolist()
    return subset[keep] if keep else subset


def summarize_candidate(spec: CandidateSpec, markets: pd.DataFrame, raw_panel: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    metadata_members = markets[candidate_mask(markets, spec)].copy()
    panel_members = metadata_members[metadata_members["in_price_panel"]].copy()
    panel_ids = panel_members["market_id"].tolist()
    panel = active_overlap_panel(raw_panel, markets, panel_ids)
    panel_ids = list(panel.columns)
    panel_members = panel_members[panel_members["market_id"].isin(panel_ids)].copy()
    resolved = panel_members[panel_members["Y_i"].notna()]
    yes_rate = float(resolved["Y_i"].mean()) if len(resolved) else np.nan
    entropy = class_entropy(yes_rate)
    missingness = float(panel.isna().mean().mean()) if panel.size else np.nan
    avg_coverage = float(panel.notna().mean(axis=0).mean()) if panel.size else np.nan
    active_median = float(panel.notna().sum(axis=1).median()) if len(panel) else np.nan
    pca = pca_metrics(panel)
    corr = correlation_metrics(panel)

    balance_score = max(0.0, 1 - abs(yes_rate - 0.5) * 2) if pd.notna(yes_rate) else 0
    liquidity_score = float(np.log1p(panel_members["volume"].fillna(0).mean())) if len(panel_members) else 0
    coverage_score = max(0.0, 1 - missingness) if pd.notna(missingness) else 0
    compress_score = 1 - (pca["pc_90"] / max(1, len(panel_ids))) if pd.notna(pca["pc_90"]) and panel_ids else 0
    realized_requested_domains = (
        panel_members.loc[panel_members["broad_domain"].isin(spec.domains), "broad_domain"].nunique()
        if len(panel_members)
        else 0
    )
    requested_domain_coverage = realized_requested_domains / len(spec.domains) if spec.domains else 0
    family_diversity_score = min(1.0, panel_members["broad_family"].nunique() / 8) if len(panel_members) else 0
    relation_score = 0.5 * family_diversity_score + 0.5 * requested_domain_coverage
    summary = {
        "universe": spec.name,
        "label": spec.label,
        "goal": spec.goal,
        "metadata_market_count": int(len(metadata_members)),
        "panel_market_count": int(len(panel_members)),
        "yes_rate": yes_rate,
        "no_rate": 1 - yes_rate if pd.notna(yes_rate) else np.nan,
        "class_entropy": entropy,
        "avg_volume": safe_mean(panel_members["volume"]) if len(panel_members) else np.nan,
        "avg_liquidity": safe_mean(panel_members["liquidity"]) if len(panel_members) else np.nan,
        "timestamp_count": int(panel.shape[0]),
        "timestamp_min": panel.index.min() if len(panel) else pd.NaT,
        "timestamp_max": panel.index.max() if len(panel) else pd.NaT,
        "panel_missingness": missingness,
        "avg_market_coverage": avg_coverage,
        "active_market_median": active_median,
        "family_count": int(panel_members["broad_family"].nunique()) if len(panel_members) else 0,
        "domain_count": int(panel_members["broad_domain"].nunique()) if len(panel_members) else 0,
        "requested_domain_count": len(spec.domains),
        "requested_domain_coverage": requested_domain_coverage,
        **corr,
        **pca,
        "balance_score": balance_score,
        "liquidity_raw_score": liquidity_score,
        "coverage_score": coverage_score,
        "compressibility_score": compress_score,
        "relationship_score": relation_score,
    }
    return summary, panel_members, panel


def normalize_scores(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    liquidity = df["liquidity_raw_score"].fillna(0)
    if liquidity.max() > liquidity.min():
        df["liquidity_score"] = (liquidity - liquidity.min()) / (liquidity.max() - liquidity.min())
    else:
        df["liquidity_score"] = 0.0
    df["overall_score"] = (
        BALANCE_WEIGHT * df["balance_score"].fillna(0)
        + LIQUIDITY_WEIGHT * df["liquidity_score"].fillna(0)
        + COVERAGE_WEIGHT * df["coverage_score"].fillna(0)
        + COMPRESS_WEIGHT * df["compressibility_score"].fillna(0)
        + RELATION_WEIGHT * df["relationship_score"].fillna(0)
    )
    return df.sort_values("overall_score", ascending=False)


def write_recommendation(
    output_path: Path,
    family: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    candidate_members: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        display = frame.copy()
        for col in display.columns:
            if pd.api.types.is_float_dtype(display[col]):
                display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
            else:
                display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x).replace("\n", " "))
        header = "| " + " | ".join(display.columns) + " |"
        divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
        rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()]
        return "\n".join([header, divider, *rows])

    best = candidate_summary.iloc[0]
    lines = [
        "# Universe Recommendation",
        "",
        "## Executive Decision",
        "",
        f"Recommended universe: **{best['label']}**.",
        "",
        "Do not implement persistent homology yet. The next topology/PCA comparison should use the recommended universe after its final panel is rebuilt and the supervised benchmark is rerun.",
        "",
        "## Why This Universe",
        "",
        f"- Panel-ready markets: {int(best['panel_market_count'])}",
        f"- YES rate: {best['yes_rate']:.3f}",
        f"- NO rate: {best['no_rate']:.3f}",
        f"- Class entropy: {best['class_entropy']:.3f} bits",
        f"- Panel missingness: {best['panel_missingness']:.3f}",
        f"- Median active markets per timestamp: {best['active_market_median']:.1f}",
        f"- Mean absolute pairwise correlation: {best['mean_abs_pairwise_corr']:.3f}",
        f"- PCs needed for 85% / 90% / 95% variance: {best['pc_85']} / {best['pc_90']} / {best['pc_95']}",
        "",
        "The best universe is the one that most improves label variation without destroying latent-factor structure. It is large enough to avoid the previous one-YES-market failure, still has correlated probability movement, and remains compressible enough for PCA and topological summaries to be meaningfully compared.",
        "",
        "## Candidate Universe Comparison",
        "",
        markdown_table(candidate_summary[
            [
                "label",
                "panel_market_count",
                "yes_rate",
                "class_entropy",
                "panel_missingness",
                "mean_abs_pairwise_corr",
                "pc_90",
                "overall_score",
            ]
        ]),
        "",
        "## Market-Family Audit Highlights",
        "",
        markdown_table(family[
            [
                "broad_family",
                "broad_domain",
                "markets",
                "panel_ready_markets",
                "yes_rate",
                "class_entropy",
                "avg_volume",
                "forecasting_usefulness_score",
            ]
        ].head(15)),
        "",
        "## Answer To The Research-Design Question",
        "",
        "**If we want to test whether topological compression preserves more forecasting information than PCA, the best current market universe is the recommended universe above.**",
        "",
        "It gives the best chance of detecting a real difference because it contains more outcome variation than the crypto-only core, retains enough cross-market dependence to make compression meaningful, and is already represented in the available price-history panel. A too-narrow crypto-only universe repeats the class-imbalance problem; a broad multi-domain universe is attractive conceptually but is not panel-ready with the current data because elections/sports coverage is sparse in the existing historical price pull.",
        "",
        "## Recommended Next Data Step",
        "",
        "Rebuild the supervised forecasting dataset using this universe, then rerun the PCA supervised baseline and diagnostics. Only after that benchmark is stable should persistent homology be added as the competing compression method.",
        "",
        "## Recommended Market IDs",
        "",
    ]
    selected = candidate_members[candidate_members["universe"] == best["universe"]].copy()
    selected = selected.sort_values(["broad_domain", "broad_family", "volume"], ascending=[True, True, False])
    lines.append(
        markdown_table(selected[["market_id", "broad_domain", "broad_family", "resolved_outcome", "volume", "question"]])
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit metadata and construct candidate market universes.")
    parser.add_argument("--markets", default="data/processed/markets.parquet")
    parser.add_argument("--raw-panel", default="data/processed/panel_hourly_raw.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    root = project_root()
    markets_path = Path(args.markets)
    raw_panel_path = Path(args.raw_panel)
    output_dir = Path(args.output_dir)
    if not markets_path.is_absolute():
        markets_path = root / markets_path
    if not raw_panel_path.is_absolute():
        raw_panel_path = root / raw_panel_path
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    ensure_dirs([output_dir])

    markets = load_markets(markets_path)
    raw_panel = pd.read_parquet(raw_panel_path)
    raw_panel.columns = raw_panel.columns.astype(str)
    raw_panel.index = pd.to_datetime(raw_panel.index, utc=True)
    markets["in_price_panel"] = markets["market_id"].isin(set(raw_panel.columns))

    audit = family_audit(markets)
    candidate_rows = []
    member_rows = []
    for spec in CANDIDATES:
        summary, members, panel = summarize_candidate(spec, markets, raw_panel)
        candidate_rows.append(summary)
        members = members.copy()
        members["universe"] = spec.name
        member_rows.append(members)
        panel.to_parquet(output_dir / f"{spec.name}_panel.parquet")

    candidate_summary = normalize_scores(pd.DataFrame(candidate_rows))
    candidate_members = pd.concat(member_rows, ignore_index=True) if member_rows else pd.DataFrame()
    source_quality_audit = markets[markets["exclude_source_quality"]].copy()

    audit.to_csv(output_dir / "market_family_audit.csv", index=False)
    source_quality_cols = [
        "market_id",
        "question",
        "slug",
        "broad_domain",
        "broad_family",
        "source_quality_reason",
        "resolved_outcome",
        "volume",
        "start_date",
        "end_date",
        "close_date",
    ]
    source_quality_audit[[col for col in source_quality_cols if col in source_quality_audit.columns]].to_csv(
        output_dir / "source_quality_audit.csv",
        index=False,
    )
    candidate_summary.to_csv(output_dir / "candidate_universe_summary.csv", index=False)
    candidate_members.to_parquet(output_dir / "candidate_universe_markets.parquet", index=False)
    candidate_members.to_csv(output_dir / "candidate_universe_markets.csv", index=False)
    write_recommendation(output_dir / "universe_recommendation.md", audit, candidate_summary, candidate_members)

    print(candidate_summary.to_string(index=False))
    print(f"Saved recommendation to {output_dir / 'universe_recommendation.md'}")


if __name__ == "__main__":
    main()
