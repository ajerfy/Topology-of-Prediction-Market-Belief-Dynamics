from __future__ import annotations

import argparse
import logging
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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
from persistent_homology_forecasting import TDA_FEATURE_COLS, compute_tda_features_for_fold
from supervised_forecasting import Fold, make_chronological_folds, probability_clip
from utils import ensure_dirs, project_root, setup_logging


PCA_SETTINGS = ("fixed_2", "fixed_5", "var_85")
TDA_WINDOWS = (24, 72, 168)
C_VALUES = (0.01, 0.1, 1.0)
H0_COLS = [col for col in TDA_FEATURE_COLS if col.startswith("h0_")]
H1_COLS = [col for col in TDA_FEATURE_COLS if col.startswith("h1_")]
TDA_GROUPS = {
    "h0": H0_COLS,
    "h1": H1_COLS,
    "h0_h1": TDA_FEATURE_COLS,
}
MARGINAL_THRESHOLD = 0.0005
RANDOM_SEED = 7


@dataclass(frozen=True)
class ModelConfig:
    representation: str
    pca_setting: str = "none"
    tda_window: int = 0
    tda_group: str = "none"
    c_value: float = 1.0

    @property
    def model_name(self) -> str:
        if self.representation == "market_probability":
            return "market_probability"
        parts = [self.representation]
        if self.pca_setting != "none":
            parts.append(self.pca_setting)
        if self.tda_window:
            parts.append(f"{self.tda_window}h")
        if self.tda_group != "none":
            parts.append(self.tda_group)
        parts.append(f"C{self.c_value:g}")
        return "_".join(parts)


def clean_supervised(supervised: pd.DataFrame) -> pd.DataFrame:
    df = supervised.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["market_id"] = df["market_id"].astype(str)
    df["p_i_t"] = probability_clip(df["p_i_t"])
    df["Y_i"] = df["Y_i"].astype(int)
    return df.sort_values(["timestamp", "market_id"]).reset_index(drop=True)


def locked_holdout_fold(panel: pd.DataFrame, holdout_fraction: float = 0.20) -> tuple[pd.DataFrame, Fold]:
    cutoff = int(len(panel.index) * (1.0 - holdout_fraction))
    if cutoff <= 0 or cutoff >= len(panel.index):
        raise ValueError("Invalid holdout split.")
    design_panel = panel.iloc[:cutoff].copy()
    fold = Fold(
        fold=999,
        train_start=panel.index[0],
        train_end=panel.index[cutoff - 1],
        test_start=panel.index[cutoff],
        test_end=panel.index[-1],
    )
    return design_panel, fold


def inner_validation_fold(panel: pd.DataFrame, outer: Fold, validation_fraction: float = 0.20) -> Fold:
    train_index = panel.loc[outer.train_start : outer.train_end].index
    split = int(len(train_index) * (1.0 - validation_fraction))
    min_train = 24 * 30
    min_val = 24 * 7
    split = max(min_train, min(split, len(train_index) - min_val))
    if split <= 0 or split >= len(train_index):
        raise ValueError(f"Cannot make inner validation fold for outer fold {outer.fold}.")
    return Fold(
        fold=outer.fold,
        train_start=train_index[0],
        train_end=train_index[split - 1],
        test_start=train_index[split],
        test_end=train_index[-1],
    )


def n_components_for_setting(train_scaled: np.ndarray, setting: str) -> int:
    max_components = min(train_scaled.shape[0], train_scaled.shape[1])
    if setting.startswith("fixed_"):
        return min(int(setting.split("_")[1]), max_components)
    if setting == "var_85":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pca = PCA(n_components=max_components, svd_solver="full").fit(train_scaled)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        return min(int(np.searchsorted(cumulative, 0.85) + 1), max_components)
    raise ValueError(f"Unknown PCA setting: {setting}")


def pca_features_for_fold(family_state: pd.DataFrame, fold: Fold, setting: str) -> tuple[pd.DataFrame, list[str], int]:
    train_family = family_state.loc[fold.train_start : fold.train_end].replace([np.inf, -np.inf], np.nan)
    cols, imputer, scaler, train_scaled = fit_family_preprocess(train_family)
    all_scaled = transform_family_state(family_state.replace([np.inf, -np.inf], np.nan), cols, imputer, scaler)
    n_components = n_components_for_setting(train_scaled, setting)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pca = PCA(n_components=n_components, svd_solver="full").fit(train_scaled)
        scores = pca.transform(all_scaled)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite PCA scores.")
    pca_cols = [f"pca_{idx + 1}" for idx in range(n_components)]
    frame = pd.DataFrame(scores, columns=pca_cols, index=family_state.index)
    return frame.reset_index().rename(columns={"index": "timestamp"}), pca_cols, n_components


def scale_tda_features(train_tda: pd.DataFrame, test_tda: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    out_cols = [f"{col}__scaled" for col in cols]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = scaler.fit_transform(imputer.fit_transform(train_tda[cols]))
    test_values = scaler.transform(imputer.transform(test_tda[cols]))
    train_out = train_tda[["timestamp"]].copy()
    test_out = test_tda[["timestamp"]].copy()
    train_out[out_cols] = train_values
    test_out[out_cols] = test_values
    if not np.isfinite(train_out[out_cols].to_numpy()).all() or not np.isfinite(test_out[out_cols].to_numpy()).all():
        raise ValueError("Non-finite scaled TDA features.")
    return train_out, test_out, out_cols


def build_model_frames(
    supervised: pd.DataFrame,
    family_state: pd.DataFrame,
    tda_features: pd.DataFrame | None,
    fold: Fold,
    config: ModelConfig,
    *,
    tda_transform: str = "real",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int]:
    train = fold_rows(supervised, fold, "train")
    test = fold_rows(supervised, fold, "test")
    feature_cols = ["p_i_t", "active_market_count_t"]
    n_components = 0

    if "pca" in config.representation:
        pca_frame, pca_cols, n_components = pca_features_for_fold(family_state, fold, config.pca_setting)
        train = train.merge(pca_frame, on="timestamp", how="left")
        test = test.merge(pca_frame, on="timestamp", how="left")
        feature_cols.extend(pca_cols)

    if "tda" in config.representation:
        if tda_features is None:
            raise ValueError("TDA features are required for TDA model frames.")
        tda_cols = TDA_GROUPS[config.tda_group]
        tda_window = tda_features[tda_features["window_hours"].eq(config.tda_window)].copy()
        if tda_window.empty:
            raise ValueError(f"No TDA features for window {config.tda_window}.")
        tda_frame = tda_window[["timestamp", *tda_cols]].sort_values("timestamp").reset_index(drop=True)
        if tda_transform == "shuffle":
            rng = np.random.default_rng(RANDOM_SEED + fold.fold + config.tda_window)
            shuffled = tda_frame[tda_cols].to_numpy().copy()
            rng.shuffle(shuffled, axis=0)
            tda_frame.loc[:, tda_cols] = shuffled
        elif tda_transform == "future_shift":
            shifted = tda_frame.copy()
            shifted["timestamp"] = shifted["timestamp"] - pd.Timedelta(hours=config.tda_window)
            tda_frame = shifted
        elif tda_transform != "real":
            raise ValueError(f"Unknown TDA transform: {tda_transform}")

        train_t = train[["timestamp"]].drop_duplicates().merge(tda_frame, on="timestamp", how="inner")
        test_t = test[["timestamp"]].drop_duplicates().merge(tda_frame, on="timestamp", how="inner")
        train_t, test_t, scaled_cols = scale_tda_features(train_t, test_t, tda_cols)
        train = train.merge(train_t, on="timestamp", how="inner")
        test = test.merge(test_t, on="timestamp", how="inner")
        feature_cols.extend(scaled_cols)

    return train, test, feature_cols, n_components


def fit_predict_logistic(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], c_value: float) -> np.ndarray:
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


def expected_calibration_error(predictions: pd.DataFrame) -> float:
    if predictions.empty:
        return np.nan
    df = predictions.copy()
    df["prob_decile"] = pd.cut(df["p_hat"], bins=np.linspace(0, 1, 11), include_lowest=True, labels=False, duplicates="drop")
    total = len(df)
    ece = 0.0
    for _, group in df.groupby("prob_decile", observed=True):
        ece += len(group) / total * abs(group["p_hat"].mean() - group["Y_i"].mean())
    return float(ece)


def result_slices(pred: pd.DataFrame, *, fold: Fold, split: str, config: ModelConfig, n_components: int, status: str) -> list[dict[str, object]]:
    base = {
        "fold": fold.fold,
        "split": split,
        "representation": config.representation,
        "model": config.model_name,
        "pca_setting": config.pca_setting,
        "tda_window": config.tda_window,
        "tda_group": config.tda_group,
        "c_value": config.c_value,
        "n_components": n_components,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "status": status,
    }
    if pred.empty or status != "ok":
        return [{**base, "eval_group_type": "overall", "eval_group": "overall", "n_obs": 0}]
    rows = [{**base, "eval_group_type": "overall", "eval_group": "overall", **evaluate(pred["Y_i"], pred["p_hat"])}]
    rows[-1]["ece"] = expected_calibration_error(pred)
    for family, group in pred.groupby("broad_family", dropna=False):
        item = {**base, "eval_group_type": "broad_family", "eval_group": str(family), **evaluate(group["Y_i"], group["p_hat"])}
        item["ece"] = expected_calibration_error(group)
        rows.append(item)
    bucketed = pred.assign(active_market_count_bucket=active_count_bucket(pred["active_market_count_t"]))
    for bucket, group in bucketed.groupby("active_market_count_bucket", observed=True):
        item = {**base, "eval_group_type": "active_market_count_bucket", "eval_group": str(bucket), **evaluate(group["Y_i"], group["p_hat"])}
        item["ece"] = expected_calibration_error(group)
        rows.append(item)
    return rows


def predict_config(
    supervised: pd.DataFrame,
    family_state: pd.DataFrame,
    tda_features: pd.DataFrame | None,
    fold: Fold,
    config: ModelConfig,
    *,
    split: str,
    tda_transform: str = "real",
) -> tuple[pd.DataFrame, list[dict[str, object]], int]:
    if config.representation == "market_probability":
        test = fold_rows(supervised, fold, "test")
        pred = test.copy()
        pred["p_hat"] = probability_clip(pred["p_i_t"])
        n_components = 0
    else:
        train, test, feature_cols, n_components = build_model_frames(
            supervised,
            family_state,
            tda_features,
            fold,
            config,
            tda_transform=tda_transform,
        )
        p_hat = fit_predict_logistic(train, test, feature_cols, config.c_value)
        pred = test.copy()
        pred["p_hat"] = p_hat

    pred["fold"] = fold.fold
    pred["split"] = split
    pred["representation"] = config.representation
    pred["model"] = config.model_name
    pred["pca_setting"] = config.pca_setting
    pred["tda_window"] = config.tda_window
    pred["tda_group"] = config.tda_group
    pred["c_value"] = config.c_value
    pred["n_components"] = n_components
    rows = result_slices(pred, fold=fold, split=split, config=config, n_components=n_components, status="ok")
    return pred, rows, n_components


def candidate_configs(representation: str) -> list[ModelConfig]:
    if representation == "pca_only":
        return [ModelConfig(representation, pca_setting=pca, c_value=c) for pca in PCA_SETTINGS for c in C_VALUES]
    if representation == "tda_only":
        return [
            ModelConfig(representation, tda_window=window, tda_group=group, c_value=c)
            for window in TDA_WINDOWS
            for group in ("h0_h1",)
            for c in C_VALUES
        ]
    if representation == "pca_plus_tda":
        return [
            ModelConfig(representation, pca_setting=pca, tda_window=window, tda_group=group, c_value=c)
            for pca in PCA_SETTINGS
            for window in TDA_WINDOWS
            for group in ("h0_h1",)
            for c in C_VALUES
        ]
    raise ValueError(representation)


def select_config(
    supervised: pd.DataFrame,
    family_state: pd.DataFrame,
    tda_features: pd.DataFrame | None,
    inner_fold: Fold,
    representation: str,
) -> tuple[ModelConfig, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for config in candidate_configs(representation):
        try:
            pred, _result_rows, n_components = predict_config(
                supervised,
                family_state,
                tda_features,
                inner_fold,
                config,
                split="inner_validation",
            )
            metrics = evaluate(pred["Y_i"], pred["p_hat"])
            rows.append(
                {
                    "representation": representation,
                    "model": config.model_name,
                    "pca_setting": config.pca_setting,
                    "tda_window": config.tda_window,
                    "tda_group": config.tda_group,
                    "c_value": config.c_value,
                    "n_components": n_components,
                    "status": "ok",
                    **metrics,
                }
            )
        except ValueError as exc:
            rows.append(
                {
                    "representation": representation,
                    "model": config.model_name,
                    "pca_setting": config.pca_setting,
                    "tda_window": config.tda_window,
                    "tda_group": config.tda_group,
                    "c_value": config.c_value,
                    "n_components": 0,
                    "status": f"skipped:{exc}",
                    "n_obs": 0,
                    "brier": np.nan,
                    "log_loss": np.nan,
                    "avg_pred": np.nan,
                    "avg_actual": np.nan,
                }
            )
    selection = pd.DataFrame(rows)
    valid = selection[selection["status"].eq("ok")].copy()
    if valid.empty:
        raise ValueError(f"No valid configs for {representation}.")
    best = valid.sort_values(["log_loss", "brier", "n_components"]).iloc[0]
    return (
        ModelConfig(
            representation=representation,
            pca_setting=str(best["pca_setting"]),
            tda_window=int(best["tda_window"]),
            tda_group=str(best["tda_group"]),
            c_value=float(best["c_value"]),
        ),
        selection,
    )


def choose_locked_config(selection_history: pd.DataFrame, representation: str) -> ModelConfig:
    valid = selection_history[
        selection_history["representation"].eq(representation) & selection_history["selected"].eq(True)
    ].copy()
    if valid.empty:
        raise ValueError(f"No selected configs for {representation}.")
    key_cols = ["pca_setting", "tda_window", "tda_group", "c_value"]
    grouped = (
        valid.groupby(key_cols, dropna=False, as_index=False)
        .agg(
            selected_folds=("outer_fold", "nunique"),
            mean_validation_log_loss=("selected_validation_log_loss", "mean"),
            mean_validation_brier=("selected_validation_brier", "mean"),
        )
        .sort_values(["selected_folds", "mean_validation_log_loss", "mean_validation_brier"], ascending=[False, True, True])
    )
    best = grouped.iloc[0]
    return ModelConfig(
        representation=representation,
        pca_setting=str(best["pca_setting"]),
        tda_window=int(best["tda_window"]),
        tda_group=str(best["tda_group"]),
        c_value=float(best["c_value"]),
    )


def calibration_by_decile(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    df = predictions.copy()
    df["prob_decile"] = pd.cut(df["p_hat"], bins=np.linspace(0, 1, 11), include_lowest=True, labels=False, duplicates="drop")
    return (
        df.groupby(["split", "fold", "representation", "model", "prob_decile"], observed=True)
        .agg(n_obs=("Y_i", "size"), avg_pred=("p_hat", "mean"), avg_actual=("Y_i", "mean"))
        .reset_index()
    )


def paired_loss_tests(predictions: pd.DataFrame, split: str = "locked_holdout") -> dict[str, float]:
    df = predictions[predictions["split"].eq(split)].copy()
    pca_model = df[df["representation"].eq("pca_only")]["model"].drop_duplicates()
    combo_model = df[df["representation"].eq("pca_plus_tda")]["model"].drop_duplicates()
    if len(pca_model) != 1 or len(combo_model) != 1:
        return {}
    key_cols = ["timestamp", "market_id", "Y_i"]
    pca = df[df["model"].eq(pca_model.iloc[0])][key_cols + ["p_hat"]].rename(columns={"p_hat": "p_pca"})
    combo = df[df["model"].eq(combo_model.iloc[0])][key_cols + ["p_hat"]].rename(columns={"p_hat": "p_combo"})
    merged = pca.merge(combo, on=key_cols, how="inner")
    if merged.empty:
        return {}
    y = merged["Y_i"].astype(int).to_numpy()
    pca_p = probability_clip(merged["p_pca"])
    combo_p = probability_clip(merged["p_combo"])
    brier_diff = (y - combo_p) ** 2 - (y - pca_p) ** 2
    log_diff = -(y * np.log(combo_p) + (1 - y) * np.log(1 - combo_p)) + (y * np.log(pca_p) + (1 - y) * np.log(1 - pca_p))
    rng = np.random.default_rng(RANDOM_SEED)
    boot_brier = []
    boot_log = []
    n = len(merged)
    for _ in range(500):
        idx = rng.integers(0, n, n)
        boot_brier.append(float(brier_diff[idx].mean()))
        boot_log.append(float(log_diff[idx].mean()))

    def normal_pvalue(diff: np.ndarray) -> float:
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        if se == 0 or not np.isfinite(se):
            return np.nan
        z = diff.mean() / se
        return float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))

    return {
        "paired_rows": int(n),
        "brier_delta_mean": float(brier_diff.mean()),
        "brier_delta_ci_low": float(np.percentile(boot_brier, 2.5)),
        "brier_delta_ci_high": float(np.percentile(boot_brier, 97.5)),
        "brier_delta_pvalue_normal": normal_pvalue(brier_diff),
        "log_loss_delta_mean": float(log_diff.mean()),
        "log_loss_delta_ci_low": float(np.percentile(boot_log, 2.5)),
        "log_loss_delta_ci_high": float(np.percentile(boot_log, 97.5)),
        "log_loss_delta_pvalue_normal": normal_pvalue(log_diff),
    }


def overall_summary(results: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    df = results[results["eval_group_type"].eq("overall") & results["status"].eq("ok")].copy()
    if split is not None:
        df = df[df["split"].eq(split)]
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["split", "representation", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            n_obs=("n_obs", "sum"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
            ece=("ece", "mean"),
            avg_pred=("avg_pred", "mean"),
            avg_actual=("avg_actual", "mean"),
        )
        .sort_values(["split", "brier", "log_loss"])
    )


def evaluate_placebos_and_ablations(
    supervised: pd.DataFrame,
    family_state: pd.DataFrame,
    tda_features: pd.DataFrame,
    fold: Fold,
    split: str,
    selected_combo: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    placebo_preds: list[pd.DataFrame] = []
    placebo_rows: list[dict[str, object]] = []
    ablation_preds: list[pd.DataFrame] = []
    ablation_rows: list[dict[str, object]] = []

    for transform in ("shuffle", "future_shift"):
        try:
            pred, rows, _ = predict_config(
                supervised,
                family_state,
                tda_features,
                fold,
                selected_combo,
                split=split,
                tda_transform=transform,
            )
            pred["placebo_type"] = transform
            for row in rows:
                row["placebo_type"] = transform
            placebo_preds.append(pred)
            placebo_rows.extend(rows)
        except ValueError as exc:
            logging.info("Skipping placebo=%s fold=%s: %s", transform, fold.fold, exc)

    for group in ("h0", "h1", "h0_h1"):
        config = ModelConfig(
            "pca_plus_tda",
            pca_setting=selected_combo.pca_setting,
            tda_window=selected_combo.tda_window,
            tda_group=group,
            c_value=selected_combo.c_value,
        )
        try:
            pred, rows, _ = predict_config(supervised, family_state, tda_features, fold, config, split=split)
            for row in rows:
                row["ablation_type"] = group
            pred["ablation_type"] = group
            ablation_preds.append(pred)
            ablation_rows.extend(rows)
        except ValueError as exc:
            logging.info("Skipping ablation=%s fold=%s: %s", group, fold.fold, exc)

    return (
        pd.concat(placebo_preds, ignore_index=True) if placebo_preds else pd.DataFrame(),
        pd.concat(ablation_preds, ignore_index=True) if ablation_preds else pd.DataFrame(),
        placebo_rows,
        ablation_rows,
    )


def run_benchmark(
    supervised: pd.DataFrame,
    family_state: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    design_panel, holdout = locked_holdout_fold(panel)
    outer_folds = make_chronological_folds(design_panel)
    logging.info("Using %s design folds and one locked holdout fold.", len(outer_folds))

    all_predictions: list[pd.DataFrame] = []
    all_results: list[dict[str, object]] = []
    selection_rows: list[pd.DataFrame] = []
    placebo_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []

    market_config = ModelConfig("market_probability")

    for outer in outer_folds:
        logging.info("Robust benchmark design fold=%s", outer.fold)
        inner = inner_validation_fold(panel, outer)
        inner_tda, _examples = compute_tda_features_for_fold(family_state, inner, TDA_WINDOWS)
        selected: dict[str, ModelConfig] = {}

        for representation in ("pca_only", "tda_only", "pca_plus_tda"):
            tda_arg = inner_tda if "tda" in representation else None
            best, selection = select_config(supervised, family_state, tda_arg, inner, representation)
            selection["outer_fold"] = outer.fold
            selection["inner_train_start"] = inner.train_start
            selection["inner_train_end"] = inner.train_end
            selection["inner_validation_start"] = inner.test_start
            selection["inner_validation_end"] = inner.test_end
            selection["selected"] = selection["model"].eq(best.model_name)
            best_row = selection[selection["selected"]].iloc[0]
            selection["selected_validation_log_loss"] = float(best_row["log_loss"])
            selection["selected_validation_brier"] = float(best_row["brier"])
            selection_rows.append(selection)
            selected[representation] = best
            logging.info("Selected fold=%s representation=%s model=%s", outer.fold, representation, best.model_name)

        outer_tda, _examples = compute_tda_features_for_fold(family_state, outer, TDA_WINDOWS)
        for config in [market_config, selected["pca_only"], selected["tda_only"], selected["pca_plus_tda"]]:
            tda_arg = outer_tda if "tda" in config.representation else None
            pred, rows, _ = predict_config(supervised, family_state, tda_arg, outer, config, split="design_cv")
            all_predictions.append(pred)
            all_results.extend(rows)

        placebo_pred, ablation_pred, p_rows, a_rows = evaluate_placebos_and_ablations(
            supervised,
            family_state,
            outer_tda,
            outer,
            "design_cv",
            selected["pca_plus_tda"],
        )
        if not placebo_pred.empty:
            placebo_rows.extend(p_rows)
        if not ablation_pred.empty:
            ablation_rows.extend(a_rows)

    selection_history = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    locked_configs = {
        representation: choose_locked_config(selection_history, representation)
        for representation in ("pca_only", "tda_only", "pca_plus_tda")
    }
    logging.info("Locked configs: %s", {key: value.model_name for key, value in locked_configs.items()})

    holdout_tda, _examples = compute_tda_features_for_fold(family_state, holdout, TDA_WINDOWS)
    for config in [market_config, locked_configs["pca_only"], locked_configs["tda_only"], locked_configs["pca_plus_tda"]]:
        tda_arg = holdout_tda if "tda" in config.representation else None
        pred, rows, _ = predict_config(supervised, family_state, tda_arg, holdout, config, split="locked_holdout")
        all_predictions.append(pred)
        all_results.extend(rows)

    placebo_pred, ablation_pred, p_rows, a_rows = evaluate_placebos_and_ablations(
        supervised,
        family_state,
        holdout_tda,
        holdout,
        "locked_holdout",
        locked_configs["pca_plus_tda"],
    )
    if not placebo_pred.empty:
        placebo_rows.extend(p_rows)
    if not ablation_pred.empty:
        ablation_rows.extend(a_rows)

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    results = pd.DataFrame(all_results)
    calibration = calibration_by_decile(predictions)
    placebo = pd.DataFrame(placebo_rows)
    ablation = pd.DataFrame(ablation_rows)
    summary = write_summary(supervised, panel, results, placebo, ablation, predictions, selection_history, locked_configs)
    return predictions, results, calibration, placebo, ablation, summary


def metric_row(summary: pd.DataFrame, representation: str, split: str = "locked_holdout") -> pd.Series:
    rows = summary[summary["split"].eq(split) & summary["representation"].eq(representation)].copy()
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.sort_values(["brier", "log_loss"]).iloc[0]


def write_summary(
    supervised: pd.DataFrame,
    panel: pd.DataFrame,
    results: pd.DataFrame,
    placebo: pd.DataFrame,
    ablation: pd.DataFrame,
    predictions: pd.DataFrame,
    selection_history: pd.DataFrame,
    locked_configs: dict[str, ModelConfig],
) -> str:
    overall = overall_summary(results)
    holdout_summary = overall_summary(results, "locked_holdout")
    design_summary = overall_summary(results, "design_cv")
    pca = metric_row(holdout_summary, "pca_only")
    combo = metric_row(holdout_summary, "pca_plus_tda")
    market = metric_row(holdout_summary, "market_probability")
    tda = metric_row(holdout_summary, "tda_only")
    paired = paired_loss_tests(predictions)

    brier_delta = float(combo["brier"] - pca["brier"]) if not combo.empty and not pca.empty else np.nan
    log_delta = float(combo["log_loss"] - pca["log_loss"]) if not combo.empty and not pca.empty else np.nan
    beats_brier = bool(np.isfinite(brier_delta) and brier_delta < 0)
    beats_log = bool(np.isfinite(log_delta) and log_delta < 0)
    meaningful_gain = bool(
        beats_brier
        and beats_log
        and max(abs(brier_delta), abs(log_delta)) > MARGINAL_THRESHOLD
    ) if np.isfinite(brier_delta) and np.isfinite(log_delta) else False

    design_deltas = []
    if not design_summary.empty:
        for fold in sorted(results.loc[results["split"].eq("design_cv"), "fold"].dropna().unique()):
            fold_overall = results[
                results["split"].eq("design_cv")
                & results["eval_group_type"].eq("overall")
                & results["status"].eq("ok")
                & results["fold"].eq(fold)
            ]
            pca_fold = fold_overall[fold_overall["representation"].eq("pca_only")]
            combo_fold = fold_overall[fold_overall["representation"].eq("pca_plus_tda")]
            if not pca_fold.empty and not combo_fold.empty:
                design_deltas.append(float(combo_fold.iloc[0]["log_loss"] - pca_fold.iloc[0]["log_loss"]))
    consistent = bool(design_deltas and np.mean(np.array(design_deltas) < 0) >= 0.60)

    placebo_holdout = placebo[
        placebo.get("split", pd.Series(dtype=object)).eq("locked_holdout")
        & placebo.get("eval_group_type", pd.Series(dtype=object)).eq("overall")
    ].copy()
    placebo_best_log = float(placebo_holdout["log_loss"].min()) if not placebo_holdout.empty else np.nan
    real_beats_placebo = bool(np.isfinite(placebo_best_log) and not combo.empty and float(combo["log_loss"]) < placebo_best_log)

    ablation_holdout = ablation[
        ablation.get("split", pd.Series(dtype=object)).eq("locked_holdout")
        & ablation.get("eval_group_type", pd.Series(dtype=object)).eq("overall")
    ].copy()
    h0 = ablation_holdout[ablation_holdout.get("ablation_type", pd.Series(dtype=object)).eq("h0")]
    h1 = ablation_holdout[ablation_holdout.get("ablation_type", pd.Series(dtype=object)).eq("h1")]
    h0h1 = ablation_holdout[ablation_holdout.get("ablation_type", pd.Series(dtype=object)).eq("h0_h1")]
    h1_adds = bool(not h0.empty and not h0h1.empty and float(h0h1.iloc[0]["log_loss"]) < float(h0.iloc[0]["log_loss"]) - MARGINAL_THRESHOLD)

    ece_delta = float(combo["ece"] - pca["ece"]) if not combo.empty and not pca.empty else np.nan
    calibration_text = "improves calibration" if np.isfinite(ece_delta) and ece_delta < -MARGINAL_THRESHOLD else "mostly shifts probabilities without clear calibration improvement"

    if beats_brier and beats_log and meaningful_gain and consistent and real_beats_placebo:
        rec = "A) Use PCA+TDA as the paper's main enhanced model"
    elif beats_brier and beats_log and real_beats_placebo:
        rec = "B) Report PCA as main result and TDA as marginal supplemental evidence"
    elif real_beats_placebo or h1_adds:
        rec = "C) Try richer topology with stricter regularization"
    else:
        rec = "D) Stop topology development and focus paper on PCA"

    def fmt(row: pd.Series, label: str) -> str:
        if row.empty:
            return f"- {label}: unavailable"
        return f"- {label}: {row['model']} Brier {row['brier']:.4f}, log loss {row['log_loss']:.4f}, ECE {row['ece']:.4f}"

    selected_counts = {}
    if not selection_history.empty:
        for rep in ("pca_only", "tda_only", "pca_plus_tda"):
            selected = selection_history[selection_history["representation"].eq(rep) & selection_history["selected"].eq(True)]
            selected_counts[rep] = Counter(selected["model"]).most_common(3)

    lines = [
        "ROBUST PCA + TOPOLOGY BENCHMARK SUMMARY",
        "",
        "Dataset:",
        f"- markets: {supervised['market_id'].nunique()}",
        f"- supervised rows: {len(supervised):,}",
        f"- YES rate by unique market: {supervised.drop_duplicates('market_id')['Y_i'].mean():.3f}",
        f"- timestamp range: {panel.index.min()} to {panel.index.max()}",
        "",
        "Locked holdout results:",
        fmt(market, "market probability"),
        fmt(pca, "PCA-only"),
        fmt(tda, "TDA-only"),
        fmt(combo, "PCA+TDA"),
        "",
        "Locked configs selected before holdout:",
        f"- PCA-only: {locked_configs['pca_only'].model_name}",
        f"- TDA-only: {locked_configs['tda_only'].model_name}",
        f"- PCA+TDA: {locked_configs['pca_plus_tda'].model_name}",
        "",
        "Success criteria:",
        f"- Does PCA+TDA beat PCA-only on locked holdout Brier? {'yes' if beats_brier else 'no'} ({brier_delta:+.6f})",
        f"- Does PCA+TDA beat PCA-only on locked holdout log loss? {'yes' if beats_log else 'no'} ({log_delta:+.6f})",
        f"- Is the gain larger than {MARGINAL_THRESHOLD}? {'yes' if meaningful_gain else 'no'}",
        f"- Is the gain consistent across design folds? {'yes' if consistent else 'no'}",
        f"- Does real TDA beat shuffled/future-shift placebo? {'yes' if real_beats_placebo else 'no'}",
        f"- Are H1 features adding beyond H0? {'yes' if h1_adds else 'no'}",
        f"- Is topology improving calibration? {calibration_text} (ECE delta {ece_delta:+.6f})",
        "",
        "Paired holdout loss test:",
        f"- rows: {paired.get('paired_rows', 0)}",
        f"- Brier delta CI: [{paired.get('brier_delta_ci_low', np.nan):+.6f}, {paired.get('brier_delta_ci_high', np.nan):+.6f}]",
        f"- Log-loss delta CI: [{paired.get('log_loss_delta_ci_low', np.nan):+.6f}, {paired.get('log_loss_delta_ci_high', np.nan):+.6f}]",
        "",
        "Design-fold selected models:",
        f"- PCA-only: {selected_counts.get('pca_only', [])}",
        f"- TDA-only: {selected_counts.get('tda_only', [])}",
        f"- PCA+TDA: {selected_counts.get('pca_plus_tda', [])}",
        "",
        "Recommendation:",
        f"- {rec}",
    ]
    return "\n".join(lines) + "\n"


def run_all(
    candidate_markets_path: Path,
    prices_path: Path,
    panel_path: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    ensure_dirs([output_dir])
    markets, panel, _raw, mask = load_inputs(candidate_markets_path, prices_path, panel_path)
    supervised = clean_supervised(load_or_build_supervised(output_dir, panel, mask, markets))
    family_state = build_family_state(panel, mask, markets)
    predictions, results, calibration, placebo, ablation, summary = run_benchmark(supervised, family_state, panel)

    predictions.to_parquet(output_dir / "robust_pca_topology_predictions.parquet", index=False)
    results.to_csv(output_dir / "robust_pca_topology_results.csv", index=False)
    calibration.to_csv(output_dir / "robust_pca_topology_calibration.csv", index=False)
    ablation.to_csv(output_dir / "robust_pca_topology_ablation.csv", index=False)
    placebo.to_csv(output_dir / "robust_pca_topology_placebo.csv", index=False)
    (output_dir / "robust_pca_topology_summary.md").write_text(summary, encoding="utf-8")
    return predictions, results, calibration, placebo, ablation, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robust nested PCA + topology enhancement benchmark.")
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
