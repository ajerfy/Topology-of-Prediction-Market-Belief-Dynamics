from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from active_set_forecasting import (
    MODEL_VARIANTS,
    append_result_slices,
    build_active_supervised_dataset,
    build_family_state,
    calibration_by_decile,
    fit_family_preprocess,
    fold_rows,
    load_inputs,
    pca_scores,
    resolve_path,
    transform_family_state,
)
from persistent_homology_forecasting import (
    MARKET_BENCHMARK_BRIER,
    MARKET_BENCHMARK_LOG_LOSS,
    PCA_BENCHMARK_BRIER,
    PCA_BENCHMARK_LOG_LOSS,
    TDA_FEATURE_COLS,
    overall_model_summary as scalar_tda_summary,
)
from supervised_forecasting import make_chronological_folds, probability_clip
from utils import ensure_dirs, project_root, setup_logging


PCA_COMPONENTS = 2
DEFAULT_TDA_WINDOWS = (24, 72, 168)


def train_predict_logistic(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], class_weight: str | None) -> np.ndarray:
    """Fit the same logistic model as the active-set benchmark, with a higher iteration cap.

    The PCA+TDA feature sets are a little less numerically friendly than the PCA-only
    features, so this keeps the benchmark specification fixed while avoiding premature
    optimizer stops.
    """
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
        max_iter=5000,
        class_weight=class_weight,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(train_x, train["Y_i"].astype(int))
        p_hat = model.predict_proba(test_x)[:, 1]
    return probability_clip(p_hat)


def load_or_build_supervised(output_dir: Path, panel: pd.DataFrame, mask: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    path = output_dir / "active_set_supervised_dataset.parquet"
    if path.exists():
        supervised = pd.read_parquet(path)
        supervised["timestamp"] = pd.to_datetime(supervised["timestamp"], utc=True)
        supervised["market_id"] = supervised["market_id"].astype(str)
        return supervised
    return build_active_supervised_dataset(panel, mask, markets)


def add_window_hours(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    parsed = results["model"].astype(str).str.extract(r"_(\d+)h", expand=False)
    results["window_hours"] = pd.to_numeric(parsed, errors="coerce").fillna(0).astype(int)
    return results


def model_summary(results: pd.DataFrame) -> pd.DataFrame:
    overall = results[(results["eval_group_type"].eq("overall")) & (results["status"].eq("ok"))].copy()
    if overall.empty:
        return pd.DataFrame()
    if "window_hours" not in overall.columns:
        overall = add_window_hours(overall)
    return (
        overall.groupby(["representation", "model", "class_weight", "window_hours"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            n_obs=("n_obs", "sum"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
            avg_pred=("avg_pred", "mean"),
            avg_actual=("avg_actual", "mean"),
        )
        .sort_values(["brier", "log_loss"])
    )


def run_pca_plus_topology_forecast(
    supervised: pd.DataFrame,
    family_state: pd.DataFrame,
    tda_features: pd.DataFrame,
    panel: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_TDA_WINDOWS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = make_chronological_folds(panel)
    result_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []

    for fold in folds:
        logging.info("Running PCA + topology fold=%s", fold.fold)
        train_rows = fold_rows(supervised, fold, "train")
        test_rows = fold_rows(supervised, fold, "test")
        if test_rows.empty:
            continue

        baseline = test_rows.copy()
        baseline["representation"] = "market_probability"
        baseline["model"] = "market_probability"
        baseline["n_components"] = 0
        baseline["window_hours"] = 0
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

        train_family = family_state.loc[fold.train_start : fold.train_end]
        fam_cols, fam_imputer, fam_scaler, fam_train_scaled = fit_family_preprocess(train_family)
        fam_all_scaled = transform_family_state(family_state, fam_cols, fam_imputer, fam_scaler)
        pca_frame, _pca = pca_scores(fam_train_scaled, fam_all_scaled, PCA_COMPONENTS)
        pca_frame.index = family_state.index
        pca_cols = pca_frame.columns.tolist()
        pca_features = pca_frame.reset_index().rename(columns={"index": "timestamp"})

        train_pca = train_rows.merge(pca_features, on="timestamp", how="left")
        test_pca = test_rows.merge(pca_features, on="timestamp", how="left")
        pca_feature_cols = ["p_i_t", "active_market_count_t", *pca_cols]

        for variant, class_weight in MODEL_VARIANTS:
            model_name = f"family_pca_fixed_{PCA_COMPONENTS}_{variant}"
            try:
                p_hat = train_predict_logistic(train_pca, test_pca, pca_feature_cols, class_weight)
                pred = test_pca.copy()
                pred["representation"] = "family_pca"
                pred["model"] = model_name
                pred["n_components"] = PCA_COMPONENTS
                pred["window_hours"] = 0
                pred["class_weight"] = variant
                pred["fold"] = fold.fold
                pred["p_hat"] = p_hat
                prediction_parts.append(pred)
                append_result_slices(
                    result_rows,
                    pred,
                    fold=fold,
                    representation="family_pca",
                    model=model_name,
                    n_components=PCA_COMPONENTS,
                    class_weight=variant,
                    status="ok",
                )
            except ValueError as exc:
                logging.info("Skipping PCA fold=%s variant=%s: %s", fold.fold, variant, exc)

        fold_tda = tda_features[tda_features["fold"].eq(fold.fold)].copy()
        for window in windows:
            tda_window = fold_tda[fold_tda["window_hours"].eq(window)]
            if tda_window.empty:
                logging.info("No TDA features for fold=%s window=%s", fold.fold, window)
                continue
            tda_frame = tda_window[["timestamp", *TDA_FEATURE_COLS]].copy()
            train_combo = train_pca.merge(tda_frame, on="timestamp", how="inner")
            test_combo = test_pca.merge(tda_frame, on="timestamp", how="inner")

            tda_only_cols = ["p_i_t", "active_market_count_t", *TDA_FEATURE_COLS]
            combo_cols = ["p_i_t", "active_market_count_t", *pca_cols, *TDA_FEATURE_COLS]

            for variant, class_weight in MODEL_VARIANTS:
                for representation, feature_cols, n_components in [
                    ("scalar_tda", tda_only_cols, 0),
                    ("pca_plus_scalar_tda", combo_cols, PCA_COMPONENTS),
                ]:
                    model_name = f"{representation}_{window}h_{variant}"
                    try:
                        p_hat = train_predict_logistic(train_combo, test_combo, feature_cols, class_weight)
                        pred = test_combo.copy()
                        pred["representation"] = representation
                        pred["model"] = model_name
                        pred["n_components"] = n_components
                        pred["window_hours"] = window
                        pred["class_weight"] = variant
                        pred["fold"] = fold.fold
                        pred["p_hat"] = p_hat
                        prediction_parts.append(pred)
                        append_result_slices(
                            result_rows,
                            pred,
                            fold=fold,
                            representation=representation,
                            model=model_name,
                            n_components=n_components,
                            class_weight=variant,
                            status="ok",
                        )
                    except ValueError as exc:
                        logging.info(
                            "Skipping representation=%s fold=%s window=%s variant=%s: %s",
                            representation,
                            fold.fold,
                            window,
                            variant,
                            exc,
                        )

    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    results = add_window_hours(pd.DataFrame(result_rows))
    calibration = calibration_by_decile(predictions)
    return predictions, results, calibration


def benchmark_from_existing(output_dir: Path) -> dict[str, tuple[float, float]]:
    benchmarks = {
        "market_probability": (MARKET_BENCHMARK_BRIER, MARKET_BENCHMARK_LOG_LOSS),
        "family_pca": (PCA_BENCHMARK_BRIER, PCA_BENCHMARK_LOG_LOSS),
        "scalar_tda": (0.0497, 0.1930),
    }
    pca_path = output_dir / "active_set_pca_forecast_results.csv"
    if pca_path.exists():
        pca = pd.read_csv(pca_path)
        pca_summary = model_summary(add_window_hours(pca))
        fam = pca_summary[pca_summary["representation"].eq("family_pca")]
        if not fam.empty:
            best = fam.sort_values(["brier", "log_loss"]).iloc[0]
            benchmarks["family_pca"] = (float(best["brier"]), float(best["log_loss"]))
        mp = pca_summary[pca_summary["representation"].eq("market_probability")]
        if not mp.empty:
            best = mp.iloc[0]
            benchmarks["market_probability"] = (float(best["brier"]), float(best["log_loss"]))
    tda_path = output_dir / "tda_forecast_results.csv"
    if tda_path.exists():
        tda = pd.read_csv(tda_path)
        tda_summary = scalar_tda_summary(tda)
        scalar = tda_summary[tda_summary["representation"].eq("tda")]
        if not scalar.empty:
            best = scalar.sort_values(["brier", "log_loss"]).iloc[0]
            benchmarks["scalar_tda"] = (float(best["brier"]), float(best["log_loss"]))
    return benchmarks


def write_summary(output_dir: Path, supervised: pd.DataFrame, results: pd.DataFrame) -> str:
    summary = model_summary(results)
    benchmarks = benchmark_from_existing(output_dir)
    pca = summary[summary["representation"].eq("family_pca")]
    combo = summary[summary["representation"].eq("pca_plus_scalar_tda")]
    scalar = summary[summary["representation"].eq("scalar_tda")]
    market = summary[summary["representation"].eq("market_probability")]

    best_market = market.iloc[0] if not market.empty else pd.Series(dtype=object)
    best_pca = pca.sort_values(["brier", "log_loss"]).iloc[0] if not pca.empty else pd.Series(dtype=object)
    best_scalar = scalar.sort_values(["brier", "log_loss"]).iloc[0] if not scalar.empty else pd.Series(dtype=object)
    best_combo = combo.sort_values(["brier", "log_loss"]).iloc[0] if not combo.empty else pd.Series(dtype=object)

    pca_brier, pca_log = benchmarks["family_pca"]
    combo_beats_pca = bool(not best_combo.empty and best_combo["brier"] < pca_brier and best_combo["log_loss"] < pca_log)
    combo_brier_delta = float(best_combo["brier"] - pca_brier) if not best_combo.empty else np.nan
    combo_log_delta = float(best_combo["log_loss"] - pca_log) if not best_combo.empty else np.nan

    if combo_beats_pca and max(abs(combo_brier_delta), abs(combo_log_delta)) < 0.0005:
        recommendation = "Treat PCA + scalar topology as a marginal enhancement, then test richer or regularized topology features."
        interpretation = "Scalar topological features add detectable but extremely small incremental predictive information beyond market probability and family-level PCA."
    elif combo_beats_pca:
        recommendation = "Continue with PCA + topology as the paper's enhanced model."
        interpretation = "Scalar topological features add incremental predictive information beyond market probability and family-level PCA."
    elif not best_combo.empty and combo_brier_delta < 0 and combo_log_delta >= 0:
        recommendation = "Treat topology as a calibration-risk feature, not a primary result."
        interpretation = "Topology may improve squared-error shrinkage but does not improve probabilistic forecast quality."
    else:
        recommendation = "Use PCA as the main compression result and report topology as non-incremental."
        interpretation = "Scalar topological features do not add robust incremental predictive information beyond PCA."

    def line(label: str, row: pd.Series) -> str:
        if row.empty:
            return f"- {label}: unavailable"
        return f"- {label}: {row['model']} Brier {row['brier']:.4f}, log loss {row['log_loss']:.4f}"

    yes_rate = supervised.drop_duplicates("market_id")["Y_i"].mean()
    lines = [
        "PCA + TOPOLOGY FORECASTING SUMMARY",
        "",
        "Research framing:",
        "- This is not PCA vs topology.",
        "- The benchmark is market-implied probability, then PCA enhancement, then PCA + topology enhancement.",
        "",
        "Dataset:",
        f"- markets: {supervised['market_id'].nunique()}",
        f"- supervised rows: {len(supervised):,}",
        f"- YES rate by unique market: {yes_rate:.3f}",
        "",
        "Prior benchmarks:",
        f"- market probability: Brier {benchmarks['market_probability'][0]:.4f}, log loss {benchmarks['market_probability'][1]:.4f}",
        f"- family-level PCA: Brier {pca_brier:.4f}, log loss {pca_log:.4f}",
        f"- scalar TDA alone: Brier {benchmarks['scalar_tda'][0]:.4f}, log loss {benchmarks['scalar_tda'][1]:.4f}",
        "",
        "This run:",
        line("market probability", best_market),
        line("family PCA", best_pca),
        line("scalar TDA", best_scalar),
        line("PCA + scalar TDA", best_combo),
        "",
        "Incremental topology test:",
        f"- PCA + topology beats family PCA on both Brier and log loss: {'yes' if combo_beats_pca else 'no'}",
        f"- PCA + topology Brier delta vs prior family PCA: {combo_brier_delta:+.4f}",
        f"- PCA + topology log-loss delta vs prior family PCA: {combo_log_delta:+.4f}",
        "",
        "Interpretation:",
        f"- {interpretation}",
        "",
        "Recommendation:",
        f"- {recommendation}",
    ]
    text = "\n".join(lines) + "\n"
    (output_dir / "pca_plus_topology_summary.md").write_text(text, encoding="utf-8")
    return text


def run_all(
    candidate_markets_path: Path,
    prices_path: Path,
    panel_path: Path,
    tda_features_path: Path,
    output_dir: Path,
    windows: tuple[int, ...] = DEFAULT_TDA_WINDOWS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    ensure_dirs([output_dir])
    markets, panel, _raw, mask = load_inputs(candidate_markets_path, prices_path, panel_path)
    supervised = load_or_build_supervised(output_dir, panel, mask, markets)
    family_state = build_family_state(panel, mask, markets)
    tda_features = pd.read_parquet(tda_features_path)
    tda_features["timestamp"] = pd.to_datetime(tda_features["timestamp"], utc=True)
    predictions, results, calibration = run_pca_plus_topology_forecast(supervised, family_state, tda_features, panel, windows)
    predictions.to_parquet(output_dir / "pca_plus_topology_predictions.parquet", index=False)
    results.to_csv(output_dir / "pca_plus_topology_results.csv", index=False)
    calibration.to_csv(output_dir / "pca_plus_topology_calibration_by_decile.csv", index=False)
    summary = write_summary(output_dir, supervised, results)
    return predictions, results, calibration, summary


def parse_windows(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether scalar topology adds incremental signal beyond family-level PCA.")
    parser.add_argument("--candidate-markets", default="data/processed/candidate_universe_markets.parquet")
    parser.add_argument("--prices", default="data/processed/prices_long.parquet")
    parser.add_argument("--panel", default="data/processed/universe_b_macro_crypto_panel.parquet")
    parser.add_argument("--tda-features", default="data/processed/tda_features.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--windows", default="24,72,168")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    root = project_root()
    *_frames, summary = run_all(
        resolve_path(root, args.candidate_markets),
        resolve_path(root, args.prices),
        resolve_path(root, args.panel),
        resolve_path(root, args.tda_features),
        resolve_path(root, args.output_dir),
        windows=parse_windows(args.windows),
    )
    print(summary)


if __name__ == "__main__":
    main()
