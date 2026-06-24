from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from active_set_forecasting import (
    active_count_bucket,
    build_family_state,
    fit_family_preprocess,
    fold_rows,
    load_inputs,
    resolve_path,
    transform_family_state,
)
from pca_plus_topology_forecasting import load_or_build_supervised
from persistent_homology_forecasting import compute_window_diagrams, diagram_stats, finite_diagram
from supervised_forecasting import Fold, make_chronological_folds, probability_clip
from utils import ensure_dirs, project_root, setup_logging


PCA_SETTING = "fixed_5"
PCA_COMPONENTS = 5
LOGISTIC_C = 0.01
WINDOW_HOURS = (24, 72, 168)
RANDOM_SEED = 17
MARGINAL_THRESHOLD = 0.0005

SCALAR_COLS = [
    "h0_num_components",
    "h0_total_persistence",
    "h0_max_persistence",
    "h0_entropy",
    "h1_num_loops",
    "h1_total_persistence",
    "h1_max_persistence",
    "h1_entropy",
]


def clean_supervised(supervised: pd.DataFrame) -> pd.DataFrame:
    df = supervised.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["market_id"] = df["market_id"].astype(str)
    df["p_i_t"] = probability_clip(df["p_i_t"])
    df["Y_i"] = df["Y_i"].astype(int)
    return df.sort_values(["timestamp", "market_id"]).reset_index(drop=True)


def pca_scores_and_residuals(family_state: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    train_family = family_state.loc[fold.train_start : fold.train_end].replace([np.inf, -np.inf], np.nan)
    cols, imputer, scaler, train_scaled = fit_family_preprocess(train_family)
    all_scaled = transform_family_state(family_state.replace([np.inf, -np.inf], np.nan), cols, imputer, scaler)
    n_components = min(PCA_COMPONENTS, train_scaled.shape[0], train_scaled.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pca = PCA(n_components=n_components, svd_solver="full").fit(train_scaled)
        scores = pca.transform(all_scaled)
        reconstruction = pca.inverse_transform(scores)
    residuals = all_scaled - reconstruction
    residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)
    residuals = np.clip(residuals, -10.0, 10.0)
    if not np.isfinite(scores).all() or not np.isfinite(residuals).all():
        raise ValueError("Non-finite PCA scores or residuals.")
    pca_cols = [f"pca_{idx + 1}" for idx in range(n_components)]
    pca_frame = pd.DataFrame(scores, columns=pca_cols, index=family_state.index)
    return pca_frame.reset_index().rename(columns={"index": "timestamp"}), residuals, pca_cols


def diagram_to_birth_persistence(diagram: np.ndarray) -> np.ndarray:
    dgm = finite_diagram(diagram)
    if dgm.size == 0:
        return np.empty((0, 2), dtype=float)
    points = np.column_stack([dgm[:, 0], dgm[:, 1] - dgm[:, 0]])
    points = points[np.isfinite(points).all(axis=1)]
    return points[points[:, 1] > 1e-12]


def fit_image_ranges(diagrams: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = [diagram_to_birth_persistence(dgm) for dgm in diagrams]
    points = [arr for arr in points if len(arr)]
    if not points:
        return (0.0, 1.0), (0.0, 1.0)
    stacked = np.vstack(points)
    birth_max = max(float(np.nanpercentile(stacked[:, 0], 99)), 1e-3)
    pers_max = max(float(np.nanpercentile(stacked[:, 1], 99)), 1e-3)
    return (0.0, birth_max * 1.05), (0.0, pers_max * 1.05)


def persistence_image_vector(
    diagram: np.ndarray,
    birth_range: tuple[float, float],
    pers_range: tuple[float, float],
    resolution: int = 10,
) -> np.ndarray:
    points = diagram_to_birth_persistence(diagram)
    if len(points) == 0:
        return np.zeros(resolution * resolution, dtype=np.float32)
    birth = np.clip(points[:, 0], birth_range[0], birth_range[1])
    persistence = np.clip(points[:, 1], pers_range[0], pers_range[1])
    hist, _, _ = np.histogram2d(
        birth,
        persistence,
        bins=resolution,
        range=[birth_range, pers_range],
        weights=persistence,
    )
    return gaussian_filter(hist, sigma=0.5, mode="constant").astype(np.float32).reshape(-1)


def compute_residual_topology_features(
    family_state: pd.DataFrame,
    residuals: np.ndarray,
    fold: Fold,
    windows: tuple[int, ...] = WINDOW_HOURS,
) -> pd.DataFrame:
    timestamps = pd.Index(family_state.index)
    fold_positions = np.flatnonzero((timestamps >= fold.train_start) & (timestamps <= fold.test_end))
    rows: list[dict[str, object]] = []
    for window in windows:
        logging.info("Computing residual PH fold=%s window=%sh timestamps=%s", fold.fold, window, len(fold_positions))
        window_rows: list[dict[str, object]] = []
        train_h1_diagrams: list[np.ndarray] = []
        for offset, pos in enumerate(fold_positions):
            if pos < window - 1:
                continue
            points = residuals[pos - window + 1 : pos + 1]
            if points.shape[0] != window:
                continue
            diagrams = compute_window_diagrams(points)
            h0 = diagram_stats(diagrams[0], "h0")
            h1 = diagram_stats(diagrams[1], "h1")
            timestamp = timestamps[pos]
            h1_dgm = finite_diagram(diagrams[1]).astype(np.float32)
            row = {
                "fold": fold.fold,
                "timestamp": timestamp,
                "window_hours": window,
                "is_train_timestamp": bool(timestamp <= fold.train_end),
                "h1_diagram": h1_dgm,
                **h0,
                **h1,
            }
            window_rows.append(row)
            if timestamp <= fold.train_end:
                train_h1_diagrams.append(h1_dgm)
            if (offset + 1) % 3000 == 0:
                logging.info("Residual PH progress fold=%s window=%sh %s/%s", fold.fold, window, offset + 1, len(fold_positions))
        birth_range, pers_range = fit_image_ranges(train_h1_diagrams)
        for row in window_rows:
            image = persistence_image_vector(row["h1_diagram"], birth_range, pers_range)
            row.pop("h1_diagram", None)
            row["h1_image_10x10"] = image.tolist()
            row["h1_image_mass"] = float(image.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def scale_timestamp_features(
    train_t: pd.DataFrame,
    test_t: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    out_cols = [f"{col}__scaled" for col in cols]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = scaler.fit_transform(imputer.fit_transform(train_t[cols]))
    test_values = scaler.transform(imputer.transform(test_t[cols]))
    train_out = train_t[["timestamp"]].copy()
    test_out = test_t[["timestamp"]].copy()
    train_out[out_cols] = train_values
    test_out[out_cols] = test_values
    if not np.isfinite(train_out[out_cols].to_numpy()).all() or not np.isfinite(test_out[out_cols].to_numpy()).all():
        raise ValueError("Non-finite scaled timestamp features.")
    return train_out, test_out, out_cols


def vector_matrix(values: pd.Series) -> np.ndarray:
    if values.empty:
        return np.empty((0, 0), dtype=float)
    return np.vstack(values.map(lambda item: np.asarray(item, dtype=np.float32)).to_numpy()).astype(float)


def fit_predict_logistic(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], c_value: float = LOGISTIC_C) -> np.ndarray:
    if train.empty or test.empty or train["Y_i"].nunique() < 2:
        raise ValueError("Training/test rows empty or training labels have one class.")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[feature_cols].astype(float)))
    x_test = scaler.transform(imputer.transform(test[feature_cols].astype(float)))
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Non-finite logistic features.")
    model = LogisticRegression(
        penalty="l2",
        C=c_value,
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(x_train, train["Y_i"].astype(int))
        p_hat = model.predict_proba(x_test)[:, 1]
    if not np.isfinite(p_hat).all():
        raise ValueError("Non-finite logistic predictions.")
    return probability_clip(p_hat)


def fit_predict_image_logistic(train: pd.DataFrame, test: pd.DataFrame, base_cols: list[str], c_value: float = LOGISTIC_C) -> np.ndarray:
    if train.empty or test.empty or train["Y_i"].nunique() < 2:
        raise ValueError("Training/test rows empty or training labels have one class.")
    train_base = train[base_cols].astype(float).to_numpy()
    test_base = test[base_cols].astype(float).to_numpy()
    train_img = vector_matrix(train["h1_image_10x10"])
    test_img = vector_matrix(test["h1_image_10x10"])
    x_train = np.column_stack([train_base, train_img])
    x_test = np.column_stack([test_base, test_img])
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(x_train))
    x_test = scaler.transform(imputer.transform(x_test))
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Non-finite persistence-image features.")
    model = LogisticRegression(
        penalty="l2",
        C=c_value,
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(x_train, train["Y_i"].astype(int))
        p_hat = model.predict_proba(x_test)[:, 1]
    if not np.isfinite(p_hat).all():
        raise ValueError("Non-finite persistence-image predictions.")
    return probability_clip(p_hat)


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


def expected_calibration_error(pred: pd.DataFrame) -> float:
    if pred.empty:
        return np.nan
    df = pred.copy()
    df["prob_decile"] = pd.cut(df["p_hat"], bins=np.linspace(0, 1, 11), include_lowest=True, labels=False, duplicates="drop")
    total = len(df)
    return float(
        sum(
            len(group) / total * abs(group["p_hat"].mean() - group["Y_i"].mean())
            for _, group in df.groupby("prob_decile", observed=True)
        )
    )


def append_result_slices(rows: list[dict[str, object]], pred: pd.DataFrame, base: dict[str, object]) -> None:
    if pred.empty:
        rows.append({**base, "eval_group_type": "overall", "eval_group": "overall", "n_obs": 0, "status": "empty"})
        return
    overall = {**base, "eval_group_type": "overall", "eval_group": "overall", "status": "ok", **evaluate(pred["Y_i"], pred["p_hat"])}
    overall["ece"] = expected_calibration_error(pred)
    rows.append(overall)
    for family, group in pred.groupby("broad_family", dropna=False):
        item = {**base, "eval_group_type": "broad_family", "eval_group": str(family), "status": "ok", **evaluate(group["Y_i"], group["p_hat"])}
        item["ece"] = expected_calibration_error(group)
        rows.append(item)
    bucketed = pred.assign(active_market_count_bucket=active_count_bucket(pred["active_market_count_t"]))
    for bucket, group in bucketed.groupby("active_market_count_bucket", observed=True):
        item = {**base, "eval_group_type": "active_market_count_bucket", "eval_group": str(bucket), "status": "ok", **evaluate(group["Y_i"], group["p_hat"])}
        item["ece"] = expected_calibration_error(group)
        rows.append(item)


def make_prediction_frame(
    rows: pd.DataFrame,
    p_hat: np.ndarray,
    *,
    fold: Fold,
    representation: str,
    model: str,
    window: int = 0,
    placebo_type: str = "none",
) -> pd.DataFrame:
    pred = rows.copy()
    pred["p_hat"] = p_hat
    pred["fold"] = fold.fold
    pred["representation"] = representation
    pred["model"] = model
    pred["window_hours"] = window
    pred["placebo_type"] = placebo_type
    return pred


def transform_topology_frame(topo: pd.DataFrame, transform: str, fold: Fold, window: int) -> pd.DataFrame:
    df = topo.copy()
    if transform == "shuffle":
        rng = np.random.default_rng(RANDOM_SEED + fold.fold + window)
        feature_cols = SCALAR_COLS + ["h1_image_10x10", "h1_image_mass"]
        shuffled = df[feature_cols].sample(frac=1.0, random_state=RANDOM_SEED + fold.fold + window).reset_index(drop=True)
        df.loc[:, feature_cols] = shuffled.to_numpy()
    elif transform == "future_shift":
        df["timestamp"] = df["timestamp"] - pd.Timedelta(hours=window)
    elif transform != "real":
        raise ValueError(transform)
    return df


def run_fold(
    supervised: pd.DataFrame,
    family_state: pd.DataFrame,
    fold: Fold,
) -> tuple[list[pd.DataFrame], list[dict[str, object]], pd.DataFrame]:
    train_rows = fold_rows(supervised, fold, "train")
    test_rows = fold_rows(supervised, fold, "test")
    pca_frame, residuals, pca_cols = pca_scores_and_residuals(family_state, fold)
    train_pca = train_rows.merge(pca_frame, on="timestamp", how="left")
    test_pca = test_rows.merge(pca_frame, on="timestamp", how="left")
    base_cols = ["p_i_t", "active_market_count_t", *pca_cols]

    predictions: list[pd.DataFrame] = []
    result_rows: list[dict[str, object]] = []

    baseline = make_prediction_frame(test_rows, probability_clip(test_rows["p_i_t"]), fold=fold, representation="market_probability", model="market_probability")
    predictions.append(baseline)
    append_result_slices(result_rows, baseline, {"fold": fold.fold, "representation": "market_probability", "model": "market_probability", "window_hours": 0, "placebo_type": "none"})

    pca_hat = fit_predict_logistic(train_pca, test_pca, base_cols)
    pca_pred = make_prediction_frame(test_pca, pca_hat, fold=fold, representation="pca_only", model=f"pca_only_{PCA_SETTING}_C{LOGISTIC_C:g}")
    predictions.append(pca_pred)
    append_result_slices(result_rows, pca_pred, {"fold": fold.fold, "representation": "pca_only", "model": f"pca_only_{PCA_SETTING}_C{LOGISTIC_C:g}", "window_hours": 0, "placebo_type": "none"})

    topology = compute_residual_topology_features(family_state, residuals, fold)
    topology_summary = (
        topology.groupby("window_hours", as_index=False)
        .agg(
            timestamps=("timestamp", "nunique"),
            h1_nontrivial_rate=("h1_num_loops", lambda x: float((x > 0).mean())),
            avg_h0_total_persistence=("h0_total_persistence", "mean"),
            avg_h1_total_persistence=("h1_total_persistence", "mean"),
            avg_h1_image_mass=("h1_image_mass", "mean"),
        )
    )
    topology_summary["fold"] = fold.fold

    for window in WINDOW_HOURS:
        topo_window = topology[topology["window_hours"].eq(window)].copy()
        for transform in ("real", "shuffle", "future_shift"):
            transformed = transform_topology_frame(topo_window, transform, fold, window)
            train_t = train_pca[["timestamp"]].drop_duplicates().merge(transformed, on="timestamp", how="inner")
            test_t = test_pca[["timestamp"]].drop_duplicates().merge(transformed, on="timestamp", how="inner")
            if train_t.empty or test_t.empty:
                continue
            train_scalar_t, test_scalar_t, scaled_cols = scale_timestamp_features(train_t, test_t, SCALAR_COLS)
            scalar_train = train_pca.merge(train_scalar_t, on="timestamp", how="inner")
            scalar_test = test_pca.merge(test_scalar_t, on="timestamp", how="inner")
            scalar_cols = [*base_cols, *scaled_cols]
            scalar_hat = fit_predict_logistic(scalar_train, scalar_test, scalar_cols)
            scalar_repr = "pca_residual_scalar_ph" if transform == "real" else f"pca_residual_scalar_ph_{transform}"
            scalar_model = f"{scalar_repr}_{window}h"
            scalar_pred = make_prediction_frame(scalar_test, scalar_hat, fold=fold, representation=scalar_repr, model=scalar_model, window=window, placebo_type=transform)
            predictions.append(scalar_pred)
            append_result_slices(result_rows, scalar_pred, {"fold": fold.fold, "representation": scalar_repr, "model": scalar_model, "window_hours": window, "placebo_type": transform})

            image_train_t = train_pca[["timestamp"]].drop_duplicates().merge(transformed[["timestamp", "h1_image_10x10"]], on="timestamp", how="inner")
            image_test_t = test_pca[["timestamp"]].drop_duplicates().merge(transformed[["timestamp", "h1_image_10x10"]], on="timestamp", how="inner")
            image_train = train_pca.merge(image_train_t, on="timestamp", how="inner")
            image_test = test_pca.merge(image_test_t, on="timestamp", how="inner")
            image_hat = fit_predict_image_logistic(image_train, image_test, base_cols)
            image_repr = "pca_residual_image_ph" if transform == "real" else f"pca_residual_image_ph_{transform}"
            image_model = f"{image_repr}_{window}h"
            image_pred = make_prediction_frame(image_test, image_hat, fold=fold, representation=image_repr, model=image_model, window=window, placebo_type=transform)
            predictions.append(image_pred)
            append_result_slices(result_rows, image_pred, {"fold": fold.fold, "representation": image_repr, "model": image_model, "window_hours": window, "placebo_type": transform})

    return predictions, result_rows, topology_summary


def calibration_by_decile(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    df = predictions.copy()
    df["prob_decile"] = pd.cut(df["p_hat"], bins=np.linspace(0, 1, 11), include_lowest=True, labels=False, duplicates="drop")
    return (
        df.groupby(["fold", "representation", "model", "window_hours", "placebo_type", "prob_decile"], observed=True)
        .agg(n_obs=("Y_i", "size"), avg_pred=("p_hat", "mean"), avg_actual=("Y_i", "mean"))
        .reset_index()
    )


def overall_summary(results: pd.DataFrame) -> pd.DataFrame:
    df = results[results["eval_group_type"].eq("overall") & results["status"].eq("ok")].copy()
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["representation", "model", "window_hours", "placebo_type"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            n_obs=("n_obs", "sum"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
            ece=("ece", "mean"),
            avg_pred=("avg_pred", "mean"),
            avg_actual=("avg_actual", "mean"),
        )
        .sort_values(["brier", "log_loss"])
    )


def write_summary(supervised: pd.DataFrame, results: pd.DataFrame, topology_summary: pd.DataFrame) -> str:
    summary = overall_summary(results)
    pca = summary[summary["representation"].eq("pca_only")].iloc[0]
    real_topology = summary[summary["representation"].isin(["pca_residual_scalar_ph", "pca_residual_image_ph"])].copy()
    best_topology = real_topology.sort_values(["log_loss", "brier"]).iloc[0] if not real_topology.empty else pd.Series(dtype=object)
    placebo = summary[summary["placebo_type"].isin(["shuffle", "future_shift"])].copy()
    best_placebo_log = float(placebo["log_loss"].min()) if not placebo.empty else np.nan
    brier_delta = float(best_topology["brier"] - pca["brier"]) if not best_topology.empty else np.nan
    log_delta = float(best_topology["log_loss"] - pca["log_loss"]) if not best_topology.empty else np.nan
    beats_pca = bool(np.isfinite(brier_delta) and np.isfinite(log_delta) and brier_delta < 0 and log_delta < 0)
    meaningful = bool(beats_pca and max(abs(brier_delta), abs(log_delta)) > MARGINAL_THRESHOLD)
    beats_placebo = bool(not best_topology.empty and np.isfinite(best_placebo_log) and float(best_topology["log_loss"]) < best_placebo_log)

    fold_deltas = []
    pca_fold = results[results["eval_group_type"].eq("overall") & results["representation"].eq("pca_only")][["fold", "brier", "log_loss"]]
    if not best_topology.empty:
        top_fold = results[
            results["eval_group_type"].eq("overall")
            & results["model"].eq(best_topology["model"])
            & results["placebo_type"].eq("real")
        ][["fold", "brier", "log_loss"]]
        merged = pca_fold.merge(top_fold, on="fold", suffixes=("_pca", "_topology"))
        if not merged.empty:
            fold_deltas = (merged["log_loss_topology"] - merged["log_loss_pca"]).tolist()
    consistent = bool(fold_deltas and np.mean(np.array(fold_deltas) < 0) >= 0.60)
    h1_nontrivial = float(topology_summary["h1_nontrivial_rate"].mean()) if not topology_summary.empty else np.nan
    avg_h1_persistence = float(topology_summary["avg_h1_total_persistence"].mean()) if not topology_summary.empty else np.nan

    if meaningful and consistent and beats_placebo:
        recommendation = "Frame topology as residual nonlinear structure after linear PCA compression."
    elif beats_pca and beats_placebo:
        recommendation = "Report residual topology as marginal supplemental evidence, not a main result."
    else:
        recommendation = "Abandon this topology path for the current paper and focus on PCA/market-implied forecasts."

    def fmt(label: str, row: pd.Series) -> str:
        if row.empty:
            return f"- {label}: unavailable"
        return f"- {label}: {row['model']} Brier {row['brier']:.4f}, log loss {row['log_loss']:.4f}, ECE {row['ece']:.4f}"

    lines = [
        "PCA RESIDUAL TOPOLOGY SUMMARY",
        "",
        "Dataset:",
        f"- markets: {supervised['market_id'].nunique()}",
        f"- supervised rows: {len(supervised):,}",
        f"- YES rate by unique market: {supervised.drop_duplicates('market_id')['Y_i'].mean():.3f}",
        "",
        "Model comparison:",
        fmt("PCA-only", pca),
        fmt("best residual PH", best_topology),
        f"- best placebo log loss: {best_placebo_log:.4f}" if np.isfinite(best_placebo_log) else "- best placebo log loss: unavailable",
        "",
        "Answers:",
        f"- Does residual PH improve over PCA-only? {'yes' if beats_pca else 'no'} (Brier delta {brier_delta:+.6f}, log-loss delta {log_delta:+.6f})",
        f"- Is the improvement larger than {MARGINAL_THRESHOLD}? {'yes' if meaningful else 'no'}",
        f"- Is improvement consistent across folds? {'yes' if consistent else 'no'}",
        f"- Are residual topological features nontrivial? {'yes' if h1_nontrivial > 0 else 'no'} (mean H1 nontrivial rate {h1_nontrivial:.3f}, mean H1 total persistence {avg_h1_persistence:.4f})",
        f"- Is any improvement robust to placebo checks? {'yes' if beats_placebo else 'no'}",
        "",
        "Recommendation:",
        f"- {recommendation}",
    ]
    return "\n".join(lines) + "\n"


def run_all(
    candidate_markets_path: Path,
    prices_path: Path,
    panel_path: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    ensure_dirs([output_dir])
    markets, panel, _raw, mask = load_inputs(candidate_markets_path, prices_path, panel_path)
    supervised = clean_supervised(load_or_build_supervised(output_dir, panel, mask, markets))
    family_state = build_family_state(panel, mask, markets)
    folds = make_chronological_folds(panel)

    prediction_parts: list[pd.DataFrame] = []
    result_rows: list[dict[str, object]] = []
    topology_parts: list[pd.DataFrame] = []

    for fold in folds:
        logging.info("Running PCA residual topology fold=%s", fold.fold)
        preds, rows, topo_summary = run_fold(supervised, family_state, fold)
        prediction_parts.extend(preds)
        result_rows.extend(rows)
        topology_parts.append(topo_summary)

    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    results = pd.DataFrame(result_rows)
    topology_summary = pd.concat(topology_parts, ignore_index=True) if topology_parts else pd.DataFrame()
    calibration = calibration_by_decile(predictions)
    summary = write_summary(supervised, results, topology_summary)

    predictions.to_parquet(output_dir / "pca_residual_topology_predictions.parquet", index=False)
    results.to_csv(output_dir / "pca_residual_topology_results.csv", index=False)
    calibration.to_csv(output_dir / "pca_residual_topology_calibration.csv", index=False)
    topology_summary.to_csv(output_dir / "pca_residual_topology_feature_summary.csv", index=False)
    (output_dir / "pca_residual_topology_summary.md").write_text(summary, encoding="utf-8")
    return predictions, results, topology_summary, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Test persistent homology on PCA residual family-state geometry.")
    parser.add_argument("--candidate-markets", default="data/processed/candidate_universe_markets.parquet")
    parser.add_argument("--prices", default="data/processed/prices_long.parquet")
    parser.add_argument("--panel", default="data/processed/universe_b_macro_crypto_panel.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    root = project_root()
    *_frames, summary = run_all(
        resolve_path(root, args.candidate_markets),
        resolve_path(root, args.prices),
        resolve_path(root, args.panel),
        resolve_path(root, args.output_dir),
    )
    print(summary)


if __name__ == "__main__":
    main()
