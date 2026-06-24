from __future__ import annotations

import argparse
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.metrics import brier_score_loss, log_loss

from active_set_forecasting import (
    MODEL_VARIANTS,
    append_result_slices,
    build_active_supervised_dataset,
    build_family_state,
    calibration_by_decile,
    fit_family_preprocess,
    fold_rows,
    load_inputs,
    resolve_path,
    transform_family_state,
)
from persistent_homology_forecasting import (
    MARKET_BENCHMARK_BRIER,
    MARKET_BENCHMARK_LOG_LOSS,
    PCA_BENCHMARK_BRIER,
    PCA_BENCHMARK_LOG_LOSS,
    compute_window_diagrams,
    finite_diagram,
    load_pca_benchmark,
)
from supervised_forecasting import Fold, make_chronological_folds, probability_clip
from utils import ensure_dirs, project_root, setup_logging


WINDOW_HOURS = (24, 72, 168)
IMAGE_RESOLUTIONS = (10, 20, 30)
LANDSCAPE_LEVELS = (3, 5)
LANDSCAPE_STEPS = 50
SCALAR_TDA_BRIER = 0.0497
SCALAR_TDA_LOG_LOSS = 0.1930


def diagram_to_birth_persistence(diagram: np.ndarray) -> np.ndarray:
    dgm = finite_diagram(diagram)
    if dgm.size == 0:
        return np.empty((0, 2), dtype=float)
    out = np.column_stack([dgm[:, 0], dgm[:, 1] - dgm[:, 0]])
    out = out[np.isfinite(out).all(axis=1)]
    out = out[out[:, 1] > 1e-12]
    return out


def fit_ranges(diagrams: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = [diagram_to_birth_persistence(dgm) for dgm in diagrams]
    points = [arr for arr in points if len(arr)]
    if not points:
        return (0.0, 1.0), (0.0, 1.0)
    all_points = np.vstack(points)
    birth_max = float(np.nanpercentile(all_points[:, 0], 99))
    pers_max = float(np.nanpercentile(all_points[:, 1], 99))
    birth_max = max(birth_max, 1e-3)
    pers_max = max(pers_max, 1e-3)
    return (0.0, birth_max * 1.05), (0.0, pers_max * 1.05)


def persistence_image_vector(
    diagram: np.ndarray,
    resolution: int,
    birth_range: tuple[float, float],
    pers_range: tuple[float, float],
) -> np.ndarray:
    points = diagram_to_birth_persistence(diagram)
    if len(points) == 0:
        return np.zeros(resolution * resolution, dtype=np.float32)
    birth = np.clip(points[:, 0], birth_range[0], birth_range[1])
    pers = np.clip(points[:, 1], pers_range[0], pers_range[1])
    weights = pers
    hist, _, _ = np.histogram2d(
        birth,
        pers,
        bins=resolution,
        range=[birth_range, pers_range],
        weights=weights,
    )
    sigma = max(0.5, resolution / 20.0)
    image = gaussian_filter(hist, sigma=sigma, mode="constant")
    return image.astype(np.float32).reshape(-1)


def persistence_landscape_vector(
    diagram: np.ndarray,
    levels: int,
    num_steps: int,
    birth_range: tuple[float, float],
    pers_range: tuple[float, float],
) -> np.ndarray:
    points = diagram_to_birth_persistence(diagram)
    grid_stop = max(birth_range[1] + pers_range[1], 1e-3)
    grid = np.linspace(0.0, grid_stop, num_steps)
    if len(points) == 0:
        return np.zeros(levels * num_steps, dtype=np.float32)
    tents = []
    for birth, persistence in points:
        death = birth + persistence
        values = np.maximum(0.0, np.minimum(grid - birth, death - grid))
        if values.max() > 0:
            tents.append(values)
    if not tents:
        return np.zeros(levels * num_steps, dtype=np.float32)
    stacked = np.vstack(tents)
    top = np.sort(stacked, axis=0)[::-1]
    if top.shape[0] < levels:
        pad = np.zeros((levels - top.shape[0], num_steps), dtype=float)
        top = np.vstack([top, pad])
    return top[:levels].astype(np.float32).reshape(-1)


def scaled_family_for_fold(family_state: pd.DataFrame, fold: Fold) -> np.ndarray:
    train_family = family_state.loc[fold.train_start : fold.train_end]
    cols, imputer, scaler, _train_scaled = fit_family_preprocess(train_family.replace([np.inf, -np.inf], np.nan))
    scaled = transform_family_state(family_state.replace([np.inf, -np.inf], np.nan), cols, imputer, scaler)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(scaled, -10.0, 10.0)


def diagrams_for_fold_window(family_state: pd.DataFrame, scaled: np.ndarray, fold: Fold, window: int) -> pd.DataFrame:
    timestamps = pd.Index(family_state.index)
    fold_mask = (timestamps >= fold.train_start) & (timestamps <= fold.test_end)
    positions = np.flatnonzero(fold_mask)
    rows = []
    for offset, pos in enumerate(positions):
        if pos < window - 1:
            continue
        points = scaled[pos - window + 1 : pos + 1]
        if points.shape[0] != window:
            continue
        diagrams = compute_window_diagrams(points)
        rows.append(
            {
                "fold": fold.fold,
                "timestamp": timestamps[pos],
                "window_hours": window,
                "is_train_timestamp": bool(timestamps[pos] <= fold.train_end),
                "h1_diagram": finite_diagram(diagrams[1]).astype(np.float32),
            }
        )
        if (offset + 1) % 3000 == 0:
            logging.info("Diagram progress fold=%s window=%sh %s/%s", fold.fold, window, offset + 1, len(positions))
    return pd.DataFrame(rows)


def build_vectorizations(
    family_state: pd.DataFrame,
    folds: list[Fold],
    windows: tuple[int, ...] = WINDOW_HOURS,
    resolutions: tuple[int, ...] = IMAGE_RESOLUTIONS,
    landscape_levels: tuple[int, ...] = LANDSCAPE_LEVELS,
    landscape_steps: int = LANDSCAPE_STEPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_rows = []
    landscape_rows = []
    for fold in folds:
        scaled = scaled_family_for_fold(family_state, fold)
        for window in windows:
            logging.info("Computing diagrams/vectorizations fold=%s window=%sh", fold.fold, window)
            diagrams = diagrams_for_fold_window(family_state, scaled, fold, window)
            if diagrams.empty:
                continue
            train_diagrams = diagrams.loc[diagrams["is_train_timestamp"], "h1_diagram"].tolist()
            birth_range, pers_range = fit_ranges(train_diagrams)
            for _, row in diagrams.iterrows():
                base = {
                    "fold": int(row["fold"]),
                    "timestamp": row["timestamp"],
                    "window_hours": int(row["window_hours"]),
                    "homology_dim": 1,
                    "birth_min": birth_range[0],
                    "birth_max": birth_range[1],
                    "pers_min": pers_range[0],
                    "pers_max": pers_range[1],
                }
                dgm = row["h1_diagram"]
                for resolution in resolutions:
                    vec = persistence_image_vector(dgm, resolution, birth_range, pers_range)
                    image_rows.append(
                        {
                            **base,
                            "resolution": resolution,
                            "feature_count": int(vec.size),
                            "features": vec.tolist(),
                        }
                    )
                for levels in landscape_levels:
                    vec = persistence_landscape_vector(dgm, levels, landscape_steps, birth_range, pers_range)
                    landscape_rows.append(
                        {
                            **base,
                            "levels": levels,
                            "num_steps": landscape_steps,
                            "feature_count": int(vec.size),
                            "features": vec.tolist(),
                        }
                    )
    return pd.DataFrame(image_rows), pd.DataFrame(landscape_rows)


def vector_matrix(features: pd.Series) -> np.ndarray:
    if features.empty:
        return np.empty((0, 0), dtype=float)
    return np.vstack(features.map(lambda value: np.asarray(value, dtype=np.float32)).to_numpy()).astype(float)


def train_predict_vector_logistic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    class_weight: str | None,
) -> np.ndarray:
    if train.empty or test.empty or train["Y_i"].nunique() < 2:
        raise ValueError("Training/test rows empty or training labels have one class.")
    train_vec = vector_matrix(train["features"])
    test_vec = vector_matrix(test["features"])
    train_x = np.column_stack([train["p_i_t"].astype(float).to_numpy(), train_vec])
    test_x = np.column_stack([test["p_i_t"].astype(float).to_numpy(), test_vec])
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise ValueError("Non-finite vectorized topology features.")
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[~np.isfinite(std) | (std < 1e-12)] = 1.0
    train_x = np.nan_to_num((train_x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    test_x = np.nan_to_num((test_x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    train_x = np.clip(train_x, -10.0, 10.0)
    test_x = np.clip(test_x, -10.0, 10.0)
    y = train["Y_i"].astype(float).to_numpy()
    sample_weight = np.ones_like(y)
    if class_weight == "balanced":
        positives = max(float(y.sum()), 1.0)
        negatives = max(float(len(y) - y.sum()), 1.0)
        sample_weight = np.where(y > 0, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))

    rng = np.random.default_rng(0)
    weights = np.zeros(train_x.shape[1], dtype=float)
    base_rate = float(np.clip(np.average(y, weights=sample_weight), 1e-4, 1 - 1e-4))
    bias = float(np.log(base_rate / (1.0 - base_rate)))
    batch_size = min(8192, len(y))
    learning_rate = 0.01
    l2 = 1e-3
    indices = np.arange(len(y))

    for _epoch in range(4):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            xb = train_x[batch]
            yb = y[batch]
            wb = sample_weight[batch]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                logits = np.clip(xb @ weights + bias, -30.0, 30.0)
            pred = 1.0 / (1.0 + np.exp(-logits))
            err = (pred - yb) * wb
            denom = max(float(wb.sum()), 1.0)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                grad_w = (xb.T @ err) / denom + l2 * weights
            grad_b = float(err.sum() / denom)
            grad_w = np.nan_to_num(grad_w, nan=0.0, posinf=0.0, neginf=0.0)
            grad_w = np.clip(grad_w, -5.0, 5.0)
            grad_b = float(np.clip(np.nan_to_num(grad_b, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0))
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b
            weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
            bias = float(np.clip(np.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0))

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        logits = np.clip(test_x @ weights + bias, -30.0, 30.0)
    return probability_clip(1.0 / (1.0 + np.exp(-logits)))


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


def result_rows_for_predictions(
    rows: list[dict[str, object]],
    pred: pd.DataFrame,
    fold: Fold,
    representation: str,
    model: str,
    class_weight: str,
    window_hours: int,
    status: str = "ok",
) -> None:
    append_result_slices(
        rows,
        pred,
        fold=fold,
        representation=representation,
        model=model,
        n_components=0,
        class_weight=class_weight,
        status=status,
    )
    if rows:
        rows[-1]["window_hours"] = window_hours


def add_window_hours(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    parsed = results["model"].astype(str).str.extract(r"_(\d+)h", expand=False)
    results["window_hours"] = pd.to_numeric(parsed, errors="coerce").fillna(0).astype(int)
    return results


def run_vector_forecast(
    supervised: pd.DataFrame,
    vectors: pd.DataFrame,
    folds: list[Fold],
    representation: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    config_cols = ["window_hours", "resolution"] if representation == "persistence_image" else ["window_hours", "levels", "num_steps"]

    for fold in folds:
        train_rows = fold_rows(supervised, fold, "train")
        test_rows = fold_rows(supervised, fold, "test")
        for config, config_vectors in vectors[vectors["fold"].eq(fold.fold)].groupby(config_cols):
            if not isinstance(config, tuple):
                config = (config,)
            config_dict = dict(zip(config_cols, config))
            feature_frame = config_vectors[["timestamp", "features"]].copy()
            train = train_rows.merge(feature_frame, on="timestamp", how="inner")
            test = test_rows.merge(feature_frame, on="timestamp", how="inner")
            window = int(config_dict["window_hours"])
            if representation == "persistence_image":
                model_base = f"pi_{window}h_{int(config_dict['resolution'])}x{int(config_dict['resolution'])}"
            else:
                model_base = f"pl_{window}h_L{int(config_dict['levels'])}_S{int(config_dict['num_steps'])}"
            for variant, class_weight in MODEL_VARIANTS:
                model_name = f"{model_base}_{variant}"
                try:
                    p_hat = train_predict_vector_logistic(train, test, class_weight)
                    pred = test.copy()
                    pred["representation"] = representation
                    pred["model"] = model_name
                    pred["window_hours"] = window
                    pred["class_weight"] = variant
                    pred["fold"] = fold.fold
                    pred["p_hat"] = p_hat
                    for key, value in config_dict.items():
                        pred[key] = value
                    prediction_parts.append(pred)
                    append_result_slices(
                        result_rows,
                        pred,
                        fold=fold,
                        representation=representation,
                        model=model_name,
                        n_components=0,
                        class_weight=variant,
                        status="ok",
                    )
                except ValueError as exc:
                    logging.info("Skipping %s fold=%s model=%s: %s", representation, fold.fold, model_name, exc)
                    append_result_slices(
                        result_rows,
                        pd.DataFrame(),
                        fold=fold,
                        representation=representation,
                        model=model_name,
                        n_components=0,
                        class_weight=variant,
                        status=f"skipped:{exc}",
                    )
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    results = add_window_hours(pd.DataFrame(result_rows))
    calibration = calibration_by_decile(predictions) if not predictions.empty else pd.DataFrame()
    return predictions, results, calibration


def model_summary(results: pd.DataFrame) -> pd.DataFrame:
    overall = results[(results["eval_group_type"].eq("overall")) & (results["status"].eq("ok"))].copy()
    if overall.empty:
        return pd.DataFrame()
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


def load_scalar_tda_benchmark(output_dir: Path) -> tuple[float, float]:
    path = output_dir / "tda_forecast_results.csv"
    if not path.exists():
        return SCALAR_TDA_BRIER, SCALAR_TDA_LOG_LOSS
    summary = model_summary(pd.read_csv(path))
    tda = summary[summary["representation"].eq("tda")]
    if tda.empty:
        return SCALAR_TDA_BRIER, SCALAR_TDA_LOG_LOSS
    best = tda.sort_values(["brier", "log_loss"]).iloc[0]
    return float(best["brier"]), float(best["log_loss"])


def write_summary(
    output_dir: Path,
    supervised: pd.DataFrame,
    image_results: pd.DataFrame,
    landscape_results: pd.DataFrame,
) -> str:
    image_summary = model_summary(image_results)
    landscape_summary = model_summary(landscape_results)
    best_image = image_summary.iloc[0] if not image_summary.empty else pd.Series(dtype=object)
    best_landscape = landscape_summary.iloc[0] if not landscape_summary.empty else pd.Series(dtype=object)
    scalar_brier, scalar_log = load_scalar_tda_benchmark(output_dir)
    pca_brier, pca_log = load_pca_benchmark(output_dir)
    yes_rate = supervised.drop_duplicates("market_id")["Y_i"].mean()

    def beats(row: pd.Series, brier: float, logloss: float) -> bool:
        return bool(not row.empty and row["brier"] < brier and row["log_loss"] < logloss)

    image_beats_scalar = beats(best_image, scalar_brier, scalar_log)
    landscape_beats_scalar = beats(best_landscape, scalar_brier, scalar_log)
    image_beats_pca = beats(best_image, pca_brier, pca_log)
    landscape_beats_pca = beats(best_landscape, pca_brier, pca_log)
    best_topology = pd.concat([image_summary, landscape_summary], ignore_index=True).sort_values(["brier", "log_loss"]).iloc[0]

    if image_beats_pca or landscape_beats_pca:
        recommendation = "A) Continue developing topology"
        reason = "A richer topological vectorization beat the PCA benchmark on both metrics."
    elif image_beats_scalar or landscape_beats_scalar:
        recommendation = "B) Try more sophisticated TDA kernels"
        reason = "Richer vectorizations improved on scalar TDA but still did not beat PCA, so topology may need kernels or learned vectorizations."
    else:
        recommendation = "D) Conclude PCA is the superior compression"
        reason = "Neither persistence images nor landscapes improved on scalar TDA or approached the family-level PCA benchmark in this supervised test."

    def row_line(label: str, row: pd.Series) -> str:
        if row.empty:
            return f"- {label}: unavailable"
        return f"- {label}: {row['model']} Brier {row['brier']:.4f}, log loss {row['log_loss']:.4f}"

    lines = [
        "PERSISTENCE VECTORIZATION SUMMARY",
        "",
        "Dataset:",
        f"- markets: {supervised['market_id'].nunique()}",
        f"- supervised rows: {len(supervised):,}",
        f"- YES rate by unique market: {yes_rate:.3f}",
        "",
        "Benchmarks:",
        f"- market probability: Brier {MARKET_BENCHMARK_BRIER:.4f}, log loss {MARKET_BENCHMARK_LOG_LOSS:.4f}",
        f"- family-level PCA: Brier {pca_brier:.4f}, log loss {pca_log:.4f}",
        f"- scalar TDA: Brier {scalar_brier:.4f}, log loss {scalar_log:.4f}",
        "",
        "Best richer topology models:",
        row_line("persistence image", best_image),
        row_line("persistence landscape", best_landscape),
        row_line("best topology overall", best_topology),
        "",
        "Answers:",
        f"1. Do persistence images outperform scalar TDA? {'yes' if image_beats_scalar else 'no'}",
        f"2. Do persistence landscapes outperform scalar TDA? {'yes' if landscape_beats_scalar else 'no'}",
        f"3. Does either topology representation beat family-level PCA? {'yes' if (image_beats_pca or landscape_beats_pca) else 'no'}",
        f"4. If not, how close do they get? Best topology Brier gap vs PCA {best_topology['brier'] - pca_brier:+.4f}, log-loss gap vs PCA {best_topology['log_loss'] - pca_log:+.4f}.",
        "5. Is topology providing unique forecasting information? Not convincingly unless it improves over scalar TDA and narrows the PCA gap under the same folds.",
        "",
        "Recommendation:",
        f"- {recommendation}",
        "",
        "Justification:",
        f"- {reason}",
        "- This is a representation test, not a search for a topology win; PCA remains the comparator to beat.",
    ]
    summary = "\n".join(lines) + "\n"
    (output_dir / "persistence_vectorization_summary.md").write_text(summary, encoding="utf-8")
    return summary


def run_all(
    candidate_markets_path: Path,
    prices_path: Path,
    panel_path: Path,
    output_dir: Path,
    windows: tuple[int, ...] = WINDOW_HOURS,
    resolutions: tuple[int, ...] = IMAGE_RESOLUTIONS,
    landscape_levels: tuple[int, ...] = LANDSCAPE_LEVELS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    ensure_dirs([output_dir])
    markets, panel, _raw, mask = load_inputs(candidate_markets_path, prices_path, panel_path)
    supervised = build_active_supervised_dataset(panel, mask, markets)
    family_state = build_family_state(panel, mask, markets)
    folds = make_chronological_folds(panel)
    images, landscapes = build_vectorizations(family_state, folds, windows, resolutions, landscape_levels)
    images.to_parquet(output_dir / "persistence_images.parquet", index=False)
    landscapes.to_parquet(output_dir / "persistence_landscapes.parquet", index=False)

    image_predictions, image_results, image_calibration = run_vector_forecast(supervised, images, folds, "persistence_image")
    landscape_predictions, landscape_results, landscape_calibration = run_vector_forecast(supervised, landscapes, folds, "persistence_landscape")

    image_results.to_csv(output_dir / "persistence_image_results.csv", index=False)
    landscape_results.to_csv(output_dir / "persistence_landscape_results.csv", index=False)
    image_predictions.to_parquet(output_dir / "persistence_image_predictions.parquet", index=False)
    landscape_predictions.to_parquet(output_dir / "persistence_landscape_predictions.parquet", index=False)
    image_calibration.to_csv(output_dir / "persistence_image_calibration_by_decile.csv", index=False)
    landscape_calibration.to_csv(output_dir / "persistence_landscape_calibration_by_decile.csv", index=False)
    summary = write_summary(output_dir, supervised, image_results, landscape_results)
    return images, landscapes, image_results, landscape_results, summary


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run persistence image/landscape forecasting benchmarks.")
    parser.add_argument("--candidate-markets", default="data/processed/candidate_universe_markets.parquet")
    parser.add_argument("--prices", default="data/processed/prices_long.parquet")
    parser.add_argument("--panel", default="data/processed/universe_b_macro_crypto_panel.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--windows", default="24,72,168")
    parser.add_argument("--resolutions", default="10,20,30")
    parser.add_argument("--landscape-levels", default="3,5")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    root = project_root()
    *_frames, summary = run_all(
        resolve_path(root, args.candidate_markets),
        resolve_path(root, args.prices),
        resolve_path(root, args.panel),
        resolve_path(root, args.output_dir),
        windows=parse_int_tuple(args.windows),
        resolutions=parse_int_tuple(args.resolutions),
        landscape_levels=parse_int_tuple(args.landscape_levels),
    )
    print(summary)


if __name__ == "__main__":
    main()
