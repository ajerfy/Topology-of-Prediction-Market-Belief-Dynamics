from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from overlap_analysis import active_mask, load_price_panel, load_universe_b_markets, market_lifetimes
from supervised_forecasting import EPS, Fold, make_chronological_folds, probability_clip
from universe_selection import class_entropy
from utils import ensure_dirs, project_root, setup_logging


FIXED_COMPONENTS = (2, 5, 10)
VARIANCE_THRESHOLDS = (0.85, 0.90, 0.95)
MODEL_VARIANTS = (("standard", None), ("balanced", "balanced"))


@dataclass
class ActivePcaTransform:
    feature_cols: list[str]
    means: pd.Series
    stds: pd.Series
    train_scaled: np.ndarray


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def clean_panel(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel.index = pd.to_datetime(panel.index, utc=True)
    panel.columns = panel.columns.astype(str)
    panel = panel.sort_index()
    return panel.replace([np.inf, -np.inf], np.nan).clip(0, 1)


def load_inputs(candidate_markets_path: Path, prices_path: Path, panel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = clean_panel(pd.read_parquet(panel_path))
    market_ids = panel.columns.astype(str).tolist()
    markets = load_universe_b_markets(candidate_markets_path)
    markets = markets[markets["market_id"].isin(market_ids)].copy()
    raw = load_price_panel(prices_path, market_ids).reindex(panel.index)
    raw = raw.reindex(columns=market_ids)
    lifetimes = market_lifetimes(markets, raw)
    mask = active_mask(panel.index, lifetimes).reindex(index=panel.index, columns=market_ids).fillna(False)
    return markets, panel, raw, mask


def build_active_supervised_dataset(panel: pd.DataFrame, mask: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    active_values = panel.where(mask)
    rows = active_values.stack(future_stack=True).dropna().rename("p_i_t").reset_index()
    rows.columns = ["timestamp", "market_id", "p_i_t"]
    active_counts = mask.sum(axis=1).rename("active_market_count_t").reset_index().rename(columns={"index": "timestamp"})
    meta_cols = ["market_id", "Y_i", "broad_domain", "broad_family", "volume", "resolved_outcome"]
    rows = rows.merge(active_counts, on="timestamp", how="left")
    rows = rows.merge(markets[meta_cols], on="market_id", how="left")
    rows = rows[rows["Y_i"].notna()].copy()
    rows["Y_i"] = rows["Y_i"].astype(int)
    rows["p_i_t"] = probability_clip(rows["p_i_t"])
    return rows.sort_values(["timestamp", "market_id"]).reset_index(drop=True)


def active_feature_columns(train_panel: pd.DataFrame, train_mask: pd.DataFrame) -> list[str]:
    active_train = train_panel.where(train_mask)
    observed = active_train.notna().sum(axis=0)
    varying = active_train.nunique(dropna=True) > 1
    cols = observed[observed > 0].index.intersection(varying[varying].index)
    return cols.astype(str).tolist()


def fit_active_pca_preprocess(train_panel: pd.DataFrame, train_mask: pd.DataFrame) -> ActivePcaTransform:
    cols = active_feature_columns(train_panel, train_mask)
    if not cols:
        raise ValueError("No active, varying market columns available for PCA.")
    active_train = train_panel[cols].where(train_mask[cols])
    means = active_train.mean(axis=0)
    stds = active_train.std(axis=0, ddof=0).replace(0, np.nan)
    keep = means[means.notna()].index.intersection(stds[stds.notna()].index)
    cols = keep.astype(str).tolist()
    means = means[cols]
    stds = stds[cols]
    train_scaled = transform_active_panel(train_panel[cols], train_mask[cols], means, stds)
    return ActivePcaTransform(cols, means, stds, train_scaled)


def transform_active_panel(panel: pd.DataFrame, mask: pd.DataFrame, means: pd.Series, stds: pd.Series) -> np.ndarray:
    cols = means.index.astype(str).tolist()
    values = panel[cols].copy()
    active = mask[cols].astype(bool)
    filled = values.where(active)
    filled = filled.fillna(means)
    scaled = (filled - means) / stds
    scaled = scaled.where(active, 0.0)
    scaled = scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    arr = scaled.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite values in active-filled PCA matrix.")
    return arr


def component_settings(train_scaled: np.ndarray, n_features: int) -> dict[str, int]:
    max_components = min(train_scaled.shape[0], n_features)
    if max_components < 1:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full = PCA(n_components=max_components, svd_solver="full").fit(train_scaled)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    settings: dict[str, int] = {}
    for fixed in FIXED_COMPONENTS:
        if fixed <= max_components:
            settings[f"fixed_{fixed}"] = fixed
    for threshold in VARIANCE_THRESHOLDS:
        k = int(np.searchsorted(cumulative, threshold) + 1)
        settings[f"var_{int(threshold * 100)}"] = min(k, max_components)
    return settings


def pca_scores(train_scaled: np.ndarray, all_scaled: np.ndarray, n_components: int) -> tuple[pd.DataFrame, PCA]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pca = PCA(n_components=n_components, svd_solver="full").fit(train_scaled)
        scores = pca.transform(all_scaled)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite PCA scores.")
    cols = [f"pca_{idx + 1}" for idx in range(n_components)]
    return pd.DataFrame(scores, columns=cols), pca


def build_family_state(panel: pd.DataFrame, mask: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    family_map = markets.set_index("market_id")["broad_family"].to_dict()
    families = sorted({family_map.get(col) for col in panel.columns if family_map.get(col)})
    parts = []
    active_panel = panel.where(mask)
    for family in families:
        cols = [col for col in panel.columns if family_map.get(col) == family]
        values = active_panel[cols]
        active = mask[cols]
        frame = pd.DataFrame(index=panel.index)
        frame[f"{family}__mean"] = values.mean(axis=1)
        frame[f"{family}__median"] = values.median(axis=1)
        frame[f"{family}__std"] = values.std(axis=1, ddof=0).fillna(0.0)
        frame[f"{family}__count"] = active.sum(axis=1).astype(float)
        parts.append(frame)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=panel.index)


def fit_family_preprocess(train_family: pd.DataFrame) -> tuple[list[str], SimpleImputer, StandardScaler, np.ndarray]:
    observed = train_family.notna().sum(axis=0)
    varying = train_family.nunique(dropna=True) > 1
    cols = observed[observed > 0].index.intersection(varying[varying].index).astype(str).tolist()
    if not cols:
        raise ValueError("No varying family-level features available for PCA.")
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(train_family[cols])
    train_scaled = scaler.fit_transform(train_imputed)
    if not np.isfinite(train_scaled).all():
        raise ValueError("Non-finite family PCA matrix.")
    return cols, imputer, scaler, train_scaled


def transform_family_state(family: pd.DataFrame, cols: list[str], imputer: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    arr = scaler.transform(imputer.transform(family[cols]))
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite transformed family state.")
    return arr


def fold_rows(supervised: pd.DataFrame, fold: Fold, split: str) -> pd.DataFrame:
    if split == "train":
        return supervised[(supervised["timestamp"] >= fold.train_start) & (supervised["timestamp"] <= fold.train_end)].copy()
    if split == "test":
        return supervised[(supervised["timestamp"] >= fold.test_start) & (supervised["timestamp"] <= fold.test_end)].copy()
    raise ValueError(split)


def evaluate(y_true: pd.Series, p_hat: np.ndarray) -> dict[str, float | int]:
    y = y_true.astype(int).to_numpy()
    p = probability_clip(p_hat)
    return {
        "n_obs": int(len(y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "avg_pred": float(np.mean(p)),
        "avg_actual": float(np.mean(y)),
    }


def active_count_bucket(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, 10, 25, 50, 100, np.inf],
        labels=["<=10", "11-25", "26-50", "51-100", ">100"],
    )


def append_result_slices(
    rows: list[dict[str, object]],
    pred: pd.DataFrame,
    *,
    fold: Fold,
    representation: str,
    model: str,
    n_components: int,
    class_weight: str,
    status: str,
) -> None:
    base = {
        "fold": fold.fold,
        "representation": representation,
        "model": model,
        "n_components": n_components,
        "class_weight": class_weight,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "status": status,
    }
    if pred.empty or status != "ok":
        rows.append({**base, "eval_group_type": "overall", "eval_group": "overall", "n_obs": 0})
        return
    rows.append({**base, "eval_group_type": "overall", "eval_group": "overall", **evaluate(pred["Y_i"], pred["p_hat"])})
    for family, group in pred.groupby("broad_family", dropna=False):
        if len(group):
            rows.append({**base, "eval_group_type": "broad_family", "eval_group": str(family), **evaluate(group["Y_i"], group["p_hat"])})
    bucketed = pred.assign(active_market_count_bucket=active_count_bucket(pred["active_market_count_t"]))
    for bucket, group in bucketed.groupby("active_market_count_bucket", observed=True):
        if len(group):
            rows.append({**base, "eval_group_type": "active_market_count_bucket", "eval_group": str(bucket), **evaluate(group["Y_i"], group["p_hat"])})


def calibration_by_decile(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    df = predictions.copy()
    df["prob_decile"] = pd.cut(
        df["p_hat"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
        labels=False,
        duplicates="drop",
    )
    return (
        df.groupby(["fold", "representation", "model", "class_weight", "prob_decile"], observed=True)
        .agg(n_obs=("Y_i", "size"), avg_pred=("p_hat", "mean"), avg_actual=("Y_i", "mean"))
        .reset_index()
    )


def train_predict_logistic(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], class_weight: str | None) -> np.ndarray:
    if train.empty or test.empty or train["Y_i"].nunique() < 2:
        raise ValueError("Training/test rows empty or training labels have one class.")
    train_x = train[feature_cols].astype(float)
    test_x = test[feature_cols].astype(float)
    if not np.isfinite(train_x.to_numpy()).all() or not np.isfinite(test_x.to_numpy()).all():
        raise ValueError("Non-finite logistic features.")
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=class_weight,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(train_x, train["Y_i"].astype(int))
        p_hat = model.predict_proba(test_x)[:, 1]
    return probability_clip(p_hat)


def run_forecast(markets: pd.DataFrame, panel: pd.DataFrame, mask: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    supervised = build_active_supervised_dataset(panel, mask, markets)
    family_state = build_family_state(panel, mask, markets)
    folds = make_chronological_folds(panel)
    result_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []

    for fold in folds:
        train_rows = fold_rows(supervised, fold, "train")
        test_rows = fold_rows(supervised, fold, "test")
        if not test_rows.empty:
            baseline = test_rows.copy()
            baseline["representation"] = "market_probability"
            baseline["model"] = "market_probability"
            baseline["n_components"] = 0
            baseline["class_weight"] = "none"
            baseline["fold"] = fold.fold
            baseline["p_hat"] = probability_clip(baseline["p_i_t"])
            prediction_parts.append(baseline)
            append_result_slices(
                result_rows,
                baseline,
                fold=fold,
                representation="market_probability",
                model="market_probability",
                n_components=0,
                class_weight="none",
                status="ok",
            )

        train_panel = panel.loc[fold.train_start : fold.train_end]
        train_mask = mask.loc[fold.train_start : fold.train_end]

        representation_scores: dict[str, tuple[pd.DataFrame, dict[str, int], dict[str, float]]] = {}

        active_pre = fit_active_pca_preprocess(train_panel, train_mask)
        active_all_scaled = transform_active_panel(panel[active_pre.feature_cols], mask[active_pre.feature_cols], active_pre.means, active_pre.stds)
        active_settings = component_settings(active_pre.train_scaled, len(active_pre.feature_cols))
        representation_scores["active_filled_pca"] = (
            pd.DataFrame(index=panel.index),
            active_settings,
            {"train_feature_count": len(active_pre.feature_cols), "train_timestamp_count": len(train_panel)},
        )

        train_family = family_state.loc[fold.train_start : fold.train_end]
        fam_cols, fam_imputer, fam_scaler, fam_train_scaled = fit_family_preprocess(train_family)
        fam_all_scaled = transform_family_state(family_state, fam_cols, fam_imputer, fam_scaler)
        fam_settings = component_settings(fam_train_scaled, len(fam_cols))
        representation_scores["family_pca"] = (
            pd.DataFrame(index=panel.index),
            fam_settings,
            {"train_feature_count": len(fam_cols), "train_timestamp_count": len(train_family)},
        )

        for representation, train_scaled, all_scaled, settings in [
            ("active_filled_pca", active_pre.train_scaled, active_all_scaled, active_settings),
            ("family_pca", fam_train_scaled, fam_all_scaled, fam_settings),
        ]:
            for model_name, n_components in sorted(settings.items(), key=lambda item: (item[1], item[0])):
                score_frame, pca = pca_scores(train_scaled, all_scaled, n_components)
                pca_cols = score_frame.columns.tolist()
                score_frame.index = panel.index
                train = train_rows.merge(score_frame.reset_index().rename(columns={"index": "timestamp"}), on="timestamp", how="left")
                test = test_rows.merge(score_frame.reset_index().rename(columns={"index": "timestamp"}), on="timestamp", how="left")
                feature_cols = ["p_i_t", "active_market_count_t", *pca_cols]
                for variant, class_weight in MODEL_VARIANTS:
                    full_model_name = f"{model_name}_{variant}"
                    try:
                        p_hat = train_predict_logistic(train, test, feature_cols, class_weight)
                        pred = test.copy()
                        pred["representation"] = representation
                        pred["model"] = full_model_name
                        pred["n_components"] = n_components
                        pred["class_weight"] = variant
                        pred["fold"] = fold.fold
                        pred["p_hat"] = p_hat
                        prediction_parts.append(pred)
                        append_result_slices(
                            result_rows,
                            pred,
                            fold=fold,
                            representation=representation,
                            model=full_model_name,
                            n_components=n_components,
                            class_weight=variant,
                            status="ok",
                        )
                    except ValueError as exc:
                        logging.info("Skipping fold=%s representation=%s model=%s: %s", fold.fold, representation, full_model_name, exc)
                        append_result_slices(
                            result_rows,
                            pd.DataFrame(),
                            fold=fold,
                            representation=representation,
                            model=full_model_name,
                            n_components=n_components,
                            class_weight=variant,
                            status=f"skipped:{exc}",
                        )

    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    results = pd.DataFrame(result_rows)
    calibration = calibration_by_decile(predictions)
    return supervised, predictions, results, calibration


def summarize_results(supervised: pd.DataFrame, results: pd.DataFrame) -> str:
    overall = results[(results["eval_group_type"] == "overall") & (results["status"] == "ok")].copy()
    summary = (
        overall.groupby(["representation", "model", "class_weight"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            n_obs=("n_obs", "sum"),
            n_components_median=("n_components", "median"),
            n_components_min=("n_components", "min"),
            n_components_max=("n_components", "max"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
            avg_pred=("avg_pred", "mean"),
            avg_actual=("avg_actual", "mean"),
        )
    )
    baseline = summary[summary["representation"] == "market_probability"].iloc[0]
    pca = summary[summary["representation"] != "market_probability"].copy()
    active_best = pca[pca["representation"] == "active_filled_pca"].sort_values("brier").iloc[0]
    family_best = pca[pca["representation"] == "family_pca"].sort_values("brier").iloc[0]
    best_log = summary.sort_values("log_loss").iloc[0]
    active_counts = supervised.groupby("timestamp")["market_id"].nunique()
    yes_rate = supervised.drop_duplicates("market_id")["Y_i"].mean()

    def improvement(row: pd.Series, metric: str) -> str:
        delta = row[metric] - baseline[metric]
        return "yes" if delta < 0 else "no"

    balanced = summary[summary["class_weight"] == "balanced"]
    standard = summary[summary["class_weight"] == "standard"]
    balanced_help = (
        balanced["brier"].mean() < standard["brier"].mean()
        if len(balanced) and len(standard)
        else False
    )

    recommendation = "B"
    recommendation_text = "Improve active-set PCA setup first"
    if (active_best["brier"] < baseline["brier"] and active_best["log_loss"] <= baseline["log_loss"]) or (
        family_best["brier"] < baseline["brier"] and family_best["log_loss"] <= baseline["log_loss"]
    ):
        recommendation = "A"
        recommendation_text = "Proceed to persistent homology using active-set representation"

    lines = [
        "ACTIVE-SET PCA FORECASTING SUMMARY",
        "",
        "1. Dataset:",
        f"- number of markets: {supervised['market_id'].nunique()}",
        f"- number of supervised rows: {len(supervised):,}",
        f"- YES rate by unique market: {yes_rate:.3f}",
        f"- active market count distribution: min {active_counts.min()}, median {active_counts.median():.1f}, mean {active_counts.mean():.1f}, max {active_counts.max()}",
        "",
        "2. Baseline:",
        f"- Brier: {baseline['brier']:.4f}",
        f"- log loss: {baseline['log_loss']:.4f}",
        "",
        "3. Best active-filled PCA:",
        f"- model: {active_best['model']}",
        f"- components: median {active_best['n_components_median']:.0f}, range {int(active_best['n_components_min'])}-{int(active_best['n_components_max'])}",
        f"- Brier: {active_best['brier']:.4f}",
        f"- log loss: {active_best['log_loss']:.4f}",
        f"- improves over baseline: Brier {improvement(active_best, 'brier')}, log loss {improvement(active_best, 'log_loss')}",
        "",
        "4. Best family-level PCA:",
        f"- model: {family_best['model']}",
        f"- components: median {family_best['n_components_median']:.0f}, range {int(family_best['n_components_min'])}-{int(family_best['n_components_max'])}",
        f"- Brier: {family_best['brier']:.4f}",
        f"- log loss: {family_best['log_loss']:.4f}",
        f"- improves over baseline: Brier {improvement(family_best, 'brier')}, log loss {improvement(family_best, 'log_loss')}",
        "",
        "5. Calibration:",
        f"- best log-loss model: {best_log['representation']} / {best_log['model']} with log loss {best_log['log_loss']:.4f}",
        f"- class weighting helps on mean Brier across PCA configs: {'yes' if balanced_help else 'no'}",
        "- Calibration should be judged against the decile output file; log loss remains the stricter warning signal for overconfident rare-positive errors.",
        "",
        "6. Interpretation:",
        "- Active-set PCA removes the structural-missingness artifact by building supervised rows only for active markets.",
        "- Active-filled PCA treats inactive markets as zero centered deviations rather than as missing probabilities.",
        "- Family-level PCA gives a fixed belief-state representation that is independent of exact market membership.",
        "- Evidence for predictive information beyond market probability is present only if PCA improves both Brier and log loss; otherwise it may still be exploiting class balance or calibration shrinkage.",
        "",
        "7. Recommendation:",
        f"- {recommendation}) {recommendation_text}",
        "",
        "Justification:",
        "- Persistent homology should only follow if this active-set PCA benchmark is stable and fair.",
        "- If PCA still improves only Brier but not log loss, the next step is benchmark refinement rather than topology.",
    ]
    return "\n".join(lines) + "\n"


def run_all(candidate_markets_path: Path, prices_path: Path, panel_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    ensure_dirs([output_dir])
    markets, panel, _raw, mask = load_inputs(candidate_markets_path, prices_path, panel_path)
    supervised, predictions, results, calibration = run_forecast(markets, panel, mask)
    summary = summarize_results(supervised, results)
    supervised.to_parquet(output_dir / "active_set_supervised_dataset.parquet", index=False)
    predictions.to_parquet(output_dir / "active_set_pca_predictions.parquet", index=False)
    results.to_csv(output_dir / "active_set_pca_forecast_results.csv", index=False)
    calibration.to_csv(output_dir / "active_set_pca_calibration_by_decile.csv", index=False)
    (output_dir / "active_set_pca_summary.md").write_text(summary, encoding="utf-8")
    return supervised, predictions, results, calibration, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run active-set PCA supervised forecasting benchmark.")
    parser.add_argument("--candidate-markets", default="data/processed/candidate_universe_markets.parquet")
    parser.add_argument("--prices", default="data/processed/prices_long.parquet")
    parser.add_argument("--panel", default="data/processed/universe_b_macro_crypto_panel.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    root = project_root()
    supervised, predictions, results, _calibration, summary = run_all(
        resolve_path(root, args.candidate_markets),
        resolve_path(root, args.prices),
        resolve_path(root, args.panel),
        resolve_path(root, args.output_dir),
    )
    logging.info("Saved active-set supervised=%s predictions=%s results=%s", len(supervised), len(predictions), len(results))
    print(summary)


if __name__ == "__main__":
    main()
