from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from persim import plot_diagrams
from ripser import ripser

from active_set_forecasting import (
    MODEL_VARIANTS,
    active_count_bucket,
    append_result_slices,
    build_active_supervised_dataset,
    build_family_state,
    calibration_by_decile,
    evaluate,
    fit_family_preprocess,
    fold_rows,
    load_inputs,
    resolve_path,
    train_predict_logistic,
    transform_family_state,
)
from supervised_forecasting import Fold, make_chronological_folds, probability_clip
from utils import ensure_dirs, project_root, setup_logging


WINDOW_HOURS = (24, 72, 168)
TDA_FEATURE_COLS = [
    "h0_num_components",
    "h0_total_persistence",
    "h0_max_persistence",
    "h0_entropy",
    "h1_num_loops",
    "h1_total_persistence",
    "h1_max_persistence",
    "h1_entropy",
]

PCA_BENCHMARK_BRIER = 0.0445
PCA_BENCHMARK_LOG_LOSS = 0.1625
MARKET_BENCHMARK_BRIER = 0.0489
MARKET_BENCHMARK_LOG_LOSS = 0.1673


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    if diagram is None or len(diagram) == 0:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(diagram, dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]
    arr = arr[arr[:, 1] >= arr[:, 0]]
    return arr


def diagram_stats(diagram: np.ndarray, prefix: str) -> dict[str, float]:
    dgm = finite_diagram(diagram)
    if dgm.size == 0:
        return {
            f"{prefix}_num_components" if prefix == "h0" else f"{prefix}_num_loops": 0,
            f"{prefix}_total_persistence": 0.0,
            f"{prefix}_max_persistence": 0.0,
            f"{prefix}_entropy": 0.0,
        }
    persistence = np.maximum(dgm[:, 1] - dgm[:, 0], 0.0)
    persistence = persistence[persistence > 1e-12]
    count_name = f"{prefix}_num_components" if prefix == "h0" else f"{prefix}_num_loops"
    if len(persistence) == 0:
        return {
            count_name: 0,
            f"{prefix}_total_persistence": 0.0,
            f"{prefix}_max_persistence": 0.0,
            f"{prefix}_entropy": 0.0,
        }
    weights = persistence / persistence.sum()
    entropy = -float(np.sum(weights * np.log2(weights + 1e-12)))
    return {
        count_name: int(len(persistence)),
        f"{prefix}_total_persistence": float(persistence.sum()),
        f"{prefix}_max_persistence": float(persistence.max()),
        f"{prefix}_entropy": entropy,
    }


def compute_window_diagrams(points: np.ndarray) -> list[np.ndarray]:
    if points.shape[0] < 2:
        return [np.empty((0, 2)), np.empty((0, 2))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = ripser(points, maxdim=1)
    return result.get("dgms", [np.empty((0, 2)), np.empty((0, 2))])


def compute_tda_features_for_fold(
    family_state: pd.DataFrame,
    fold: Fold,
    windows: tuple[int, ...] = WINDOW_HOURS,
) -> tuple[pd.DataFrame, dict[int, tuple[pd.Timestamp, list[np.ndarray]]]]:
    train_family = family_state.loc[fold.train_start : fold.train_end]
    cols, imputer, scaler, _train_scaled = fit_family_preprocess(train_family.replace([np.inf, -np.inf], np.nan))
    scaled = transform_family_state(family_state.replace([np.inf, -np.inf], np.nan), cols, imputer, scaler)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = np.clip(scaled, -10.0, 10.0)

    timestamps = pd.Index(family_state.index)
    fold_mask = (timestamps >= fold.train_start) & (timestamps <= fold.test_end)
    fold_positions = np.flatnonzero(fold_mask)
    rows: list[dict[str, object]] = []
    examples: dict[int, tuple[pd.Timestamp, list[np.ndarray]]] = {}

    for window in windows:
        logging.info("Computing TDA features fold=%s window=%sh timestamps=%s", fold.fold, window, len(fold_positions))
        for offset, pos in enumerate(fold_positions):
            if pos < window - 1:
                continue
            start = pos - window + 1
            points = scaled[start : pos + 1]
            if points.shape[0] != window:
                continue
            diagrams = compute_window_diagrams(points)
            h0 = diagram_stats(diagrams[0], "h0")
            h1 = diagram_stats(diagrams[1], "h1")
            timestamp = timestamps[pos]
            row = {
                "fold": fold.fold,
                "timestamp": timestamp,
                "window_hours": window,
                "family_feature_count": len(cols),
                **h0,
                **h1,
            }
            rows.append(row)
            if window not in examples and timestamp >= fold.test_start and h1["h1_num_loops"] > 0:
                examples[window] = (timestamp, diagrams)
            if (offset + 1) % 2500 == 0:
                logging.info("TDA progress fold=%s window=%sh %s/%s", fold.fold, window, offset + 1, len(fold_positions))
    return pd.DataFrame(rows), examples


def build_tda_features(family_state: pd.DataFrame, folds: list[Fold], windows: tuple[int, ...] = WINDOW_HOURS) -> tuple[pd.DataFrame, dict[tuple[int, int], tuple[pd.Timestamp, list[np.ndarray]]]]:
    parts: list[pd.DataFrame] = []
    examples: dict[tuple[int, int], tuple[pd.Timestamp, list[np.ndarray]]] = {}
    for fold in folds:
        features, fold_examples = compute_tda_features_for_fold(family_state, fold, windows)
        parts.append(features)
        for window, value in fold_examples.items():
            examples[(fold.fold, window)] = value
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), examples


def run_tda_forecast(
    supervised: pd.DataFrame,
    tda_features: pd.DataFrame,
    folds: list[Fold],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []

    for fold in folds:
        train_rows = fold_rows(supervised, fold, "train")
        test_rows = fold_rows(supervised, fold, "test")
        if not test_rows.empty:
            baseline = test_rows.copy()
            baseline["representation"] = "market_probability"
            baseline["model"] = "market_probability"
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

        for window, fold_features in tda_features[tda_features["fold"].eq(fold.fold)].groupby("window_hours"):
            feature_frame = fold_features[["timestamp", *TDA_FEATURE_COLS]].copy()
            train = train_rows.merge(feature_frame, on="timestamp", how="inner")
            test = test_rows.merge(feature_frame, on="timestamp", how="inner")
            feature_cols = ["p_i_t", "active_market_count_t", *TDA_FEATURE_COLS]
            for variant, class_weight in MODEL_VARIANTS:
                model_name = f"tda_{int(window)}h_{variant}"
                try:
                    p_hat = train_predict_logistic(train, test, feature_cols, class_weight)
                    pred = test.copy()
                    pred["representation"] = "tda"
                    pred["model"] = model_name
                    pred["window_hours"] = int(window)
                    pred["n_components"] = 0
                    pred["class_weight"] = variant
                    pred["fold"] = fold.fold
                    pred["p_hat"] = p_hat
                    prediction_parts.append(pred)
                    append_result_slices(
                        result_rows,
                        pred,
                        fold=fold,
                        representation="tda",
                        model=model_name,
                        n_components=0,
                        class_weight=variant,
                        status="ok",
                    )
                except ValueError as exc:
                    logging.info("Skipping fold=%s window=%s model=%s: %s", fold.fold, window, model_name, exc)
                    append_result_slices(
                        result_rows,
                        pd.DataFrame(),
                        fold=fold,
                        representation="tda",
                        model=model_name,
                        n_components=0,
                        class_weight=variant,
                        status=f"skipped:{exc}",
                    )

    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    results = pd.DataFrame(result_rows)
    if not results.empty:
        parsed_window = results["model"].astype(str).str.extract(r"tda_(\d+)h", expand=False)
        results["window_hours"] = pd.to_numeric(parsed_window, errors="coerce").fillna(0).astype(int)
    calibration = calibration_by_decile(predictions)
    return predictions, results, calibration


def overall_model_summary(results: pd.DataFrame) -> pd.DataFrame:
    overall = results[(results["eval_group_type"].eq("overall")) & (results["status"].eq("ok"))].copy()
    if overall.empty:
        return pd.DataFrame()
    if "window_hours" not in overall.columns:
        parsed_window = overall["model"].astype(str).str.extract(r"tda_(\d+)h", expand=False)
        overall["window_hours"] = pd.to_numeric(parsed_window, errors="coerce").fillna(0).astype(int)
    group_cols = ["representation", "model", "class_weight"]
    group_cols.append("window_hours")
    return (
        overall.groupby(group_cols, dropna=False, as_index=False)
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


def tda_diagnostics(tda_features: pd.DataFrame) -> pd.DataFrame:
    if tda_features.empty:
        return pd.DataFrame()
    return (
        tda_features.groupby(["window_hours"], as_index=False)
        .agg(
            timestamps=("timestamp", "nunique"),
            h1_nontrivial_rate=("h1_num_loops", lambda x: float((x > 0).mean())),
            avg_h0_total_persistence=("h0_total_persistence", "mean"),
            avg_h1_total_persistence=("h1_total_persistence", "mean"),
            avg_h0_entropy=("h0_entropy", "mean"),
            avg_h1_entropy=("h1_entropy", "mean"),
            h1_entropy_std=("h1_entropy", "std"),
            h1_total_persistence_std=("h1_total_persistence", "std"),
        )
    )


def save_tda_figures(
    tda_features: pd.DataFrame,
    examples: dict[tuple[int, int], tuple[pd.Timestamp, list[np.ndarray]]],
    output_dir: Path,
) -> None:
    figure_dir = output_dir / "figures" / "tda"
    ensure_dirs([figure_dir])
    if tda_features.empty:
        return

    for col, label, filename in [
        ("h1_entropy", "H1 persistence entropy", "h1_entropy_over_time.png"),
        ("h1_total_persistence", "H1 total persistence", "h1_total_persistence_over_time.png"),
        ("h0_total_persistence", "H0 total persistence", "h0_total_persistence_over_time.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        for window, group in tda_features.groupby("window_hours"):
            ordered = group.sort_values("timestamp")
            ax.plot(ordered["timestamp"], ordered[col], linewidth=0.8, alpha=0.75, label=f"{int(window)}h")
        ax.set_title(label)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(label)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=180)
        plt.close(fig)

    for (fold, window), (timestamp, diagrams) in list(examples.items())[:3]:
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_diagrams(diagrams, show=False, ax=ax)
        ax.set_title(f"Persistence diagram fold {fold}, {window}h, {timestamp}")
        fig.tight_layout()
        fig.savefig(figure_dir / f"persistence_diagram_fold{fold}_{window}h.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
        for dim, axis in enumerate(axes):
            dgm = finite_diagram(diagrams[dim])
            for idx, (birth, death) in enumerate(dgm):
                axis.hlines(idx, birth, death, color="#335C81" if dim == 0 else "#C85A3A", linewidth=1.5)
            axis.set_title(f"H{dim} barcode")
            axis.set_xlabel("Filtration")
            axis.set_ylabel("Feature")
        fig.suptitle(f"Barcode fold {fold}, {window}h")
        fig.tight_layout()
        fig.savefig(figure_dir / f"barcode_fold{fold}_{window}h.png", dpi=180)
        plt.close(fig)


def load_pca_benchmark(output_dir: Path) -> tuple[float, float]:
    path = output_dir / "active_set_pca_forecast_results.csv"
    if not path.exists():
        return PCA_BENCHMARK_BRIER, PCA_BENCHMARK_LOG_LOSS
    results = pd.read_csv(path)
    overall = results[(results["eval_group_type"].eq("overall")) & (results["status"].eq("ok"))]
    pca = overall[overall["representation"].eq("family_pca")]
    if pca.empty:
        return PCA_BENCHMARK_BRIER, PCA_BENCHMARK_LOG_LOSS
    summary = pca.groupby(["representation", "model", "class_weight"], as_index=False).agg(
        brier=("brier", "mean"),
        log_loss=("log_loss", "mean"),
    )
    best = summary.sort_values(["brier", "log_loss"]).iloc[0]
    return float(best["brier"]), float(best["log_loss"])


def summarize_tda(
    supervised: pd.DataFrame,
    family_state: pd.DataFrame,
    results: pd.DataFrame,
    tda_features: pd.DataFrame,
    output_dir: Path,
) -> str:
    summary = overall_model_summary(results)
    baseline = summary[summary["representation"].eq("market_probability")].iloc[0]
    tda = summary[summary["representation"].eq("tda")].copy()
    best_tda = tda.sort_values(["brier", "log_loss"]).iloc[0] if not tda.empty else pd.Series(dtype=object)
    pca_brier, pca_log_loss = load_pca_benchmark(output_dir)
    diagnostics = tda_diagnostics(tda_features)
    yes_rate = supervised.drop_duplicates("market_id")["Y_i"].mean()
    best_diag = diagnostics[diagnostics["window_hours"].eq(best_tda["window_hours"])].iloc[0] if not diagnostics.empty and not best_tda.empty else None

    beats_market = bool(best_tda["brier"] < baseline["brier"] and best_tda["log_loss"] < baseline["log_loss"]) if not best_tda.empty else False
    beats_pca = bool(best_tda["brier"] < pca_brier and best_tda["log_loss"] < pca_log_loss) if not best_tda.empty else False
    h1_common = bool(best_diag is not None and best_diag["h1_nontrivial_rate"] > 0.5)

    if beats_pca:
        recommendation = "A) Continue developing persistent homology features"
        justification = "The first persistence summaries beat the strongest PCA benchmark on both Brier score and log loss, so the topology signal is already forecast-relevant."
    elif beats_market:
        recommendation = "B) Try richer topology constructions (persistence images, landscapes, kernels, etc.)"
        justification = "The first persistence summaries beat raw market probability but do not beat the PCA benchmark, which suggests topology may contain signal but the current hand-built summaries are too lossy."
    else:
        recommendation = "B) Try richer topology constructions (persistence images, landscapes, kernels, etc.)"
        justification = "The basic scalar persistence summaries do not yet beat the PCA benchmark; stopping now would only test a weak TDA representation, not topology as a class."

    diag_lines = []
    for _, row in diagnostics.iterrows():
        diag_lines.append(
            f"- {int(row['window_hours'])}h: H1 nontrivial rate {row['h1_nontrivial_rate']:.3f}, "
            f"avg H0 persistence {row['avg_h0_total_persistence']:.3f}, avg H1 persistence {row['avg_h1_total_persistence']:.3f}"
        )

    lines = [
        "TDA FORECASTING SUMMARY",
        "",
        "1. Dataset",
        f"- number of markets: {supervised['market_id'].nunique()}",
        f"- number of timestamps: {family_state.shape[0]:,}",
        f"- number of supervised rows: {len(supervised):,}",
        f"- YES rate by unique market: {yes_rate:.3f}",
        "",
        "2. Baselines",
        f"- market probability Brier/log loss: {baseline['brier']:.4f} / {baseline['log_loss']:.4f}",
        f"- family-level PCA Brier/log loss: {pca_brier:.4f} / {pca_log_loss:.4f}",
        "",
        "3. Best TDA model",
        f"- window size: {int(best_tda['window_hours']) if not best_tda.empty else 'none'}h",
        "- feature set: H0/H1 count, total persistence, max persistence, persistence entropy",
        f"- Brier: {best_tda['brier']:.4f}" if not best_tda.empty else "- Brier: unavailable",
        f"- log loss: {best_tda['log_loss']:.4f}" if not best_tda.empty else "- log loss: unavailable",
        "",
        "4. Comparison",
        f"- does TDA beat market probability? {'yes' if beats_market else 'no'}",
        f"- does TDA beat family-level PCA? {'yes' if beats_pca else 'no'}",
        "",
        "5. Topological findings",
        f"- are H1 features common or rare? {'common' if h1_common else 'rare/moderate'}",
        *diag_lines,
        "- persistence statistics are stable if their time-series plots show gradual movement rather than isolated spikes; inspect data/processed/figures/tda.",
        "- the point cloud is topologically nontrivial only if H1 persists regularly and improves forecasting beyond PCA.",
        "",
        "6. Interpretation",
        f"- topology adds predictive information beyond PCA: {'yes' if beats_pca else 'not yet'}",
        f"- support for the paper thesis: {'initially positive' if beats_pca else 'inconclusive with scalar persistence summaries'}",
        "",
        "7. Recommendation",
        f"- {recommendation}",
        "",
        "Justification:",
        f"- {justification}",
        "- The comparison used the same active-set family-state representation and chronological folds as the successful PCA benchmark, so differences are attributable to the compression features rather than a changed data object.",
        "",
        "MOST IMPORTANT QUESTION",
        f"- Can topological summaries outperform the strongest PCA benchmark? {'Yes in this run.' if beats_pca else 'No, not with this first scalar-feature construction.'}",
    ]
    return "\n".join(lines) + "\n"


def run_all(
    candidate_markets_path: Path,
    prices_path: Path,
    panel_path: Path,
    output_dir: Path,
    windows: tuple[int, ...] = WINDOW_HOURS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    ensure_dirs([output_dir, output_dir / "figures" / "tda"])
    markets, panel, _raw, mask = load_inputs(candidate_markets_path, prices_path, panel_path)
    supervised = build_active_supervised_dataset(panel, mask, markets)
    family_state = build_family_state(panel, mask, markets)
    folds = make_chronological_folds(panel)
    tda_features, examples = build_tda_features(family_state, folds, windows)
    predictions, results, calibration = run_tda_forecast(supervised, tda_features, folds)
    summary = summarize_tda(supervised, family_state, results, tda_features, output_dir)

    tda_features.to_parquet(output_dir / "tda_features.parquet", index=False)
    predictions.to_parquet(output_dir / "tda_predictions.parquet", index=False)
    results.to_csv(output_dir / "tda_forecast_results.csv", index=False)
    calibration.to_csv(output_dir / "tda_calibration_by_decile.csv", index=False)
    (output_dir / "tda_summary.md").write_text(summary, encoding="utf-8")
    save_tda_figures(tda_features, examples, output_dir)
    return supervised, family_state, tda_features, predictions, results, calibration, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run persistent-homology forecasting benchmark on family-state windows.")
    parser.add_argument("--candidate-markets", default="data/processed/candidate_universe_markets.parquet")
    parser.add_argument("--prices", default="data/processed/prices_long.parquet")
    parser.add_argument("--panel", default="data/processed/universe_b_macro_crypto_panel.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--windows", default="24,72,168", help="Comma-separated sliding-window sizes in hours.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    root = project_root()
    windows = tuple(int(value.strip()) for value in args.windows.split(",") if value.strip())
    *_frames, summary = run_all(
        resolve_path(root, args.candidate_markets),
        resolve_path(root, args.prices),
        resolve_path(root, args.panel),
        resolve_path(root, args.output_dir),
        windows=windows,
    )
    print(summary)


if __name__ == "__main__":
    main()
