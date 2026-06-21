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

from utils import ensure_dirs, project_root, setup_logging


EPS = 1e-6
VARIANCE_THRESHOLDS = (0.85, 0.90, 0.95)
FIXED_COMPONENTS = (2, 5, 10)


@dataclass(frozen=True)
class Fold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def clean_panel(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel.index = pd.to_datetime(panel.index, utc=True)
    panel.columns = panel.columns.astype(str)
    panel = panel.sort_index()
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.clip(lower=0, upper=1)
    return panel


def clean_market_universe(markets: pd.DataFrame) -> pd.DataFrame:
    markets = markets.copy()
    markets["market_id"] = markets["market_id"].astype(str)
    for col in ["start_date", "end_date", "close_date"]:
        if col in markets.columns:
            markets[col] = pd.to_datetime(markets[col], utc=True, errors="coerce")
    markets["Y_i"] = markets["resolved_outcome"].map({"Yes": 1, "No": 0})
    return markets


def make_chronological_folds(
    panel: pd.DataFrame,
    *,
    initial_train_fraction: float = 0.60,
    test_window_hours: int = 24 * 14,
    step_hours: int = 24 * 14,
    min_test_hours: int = 24 * 7,
) -> list[Fold]:
    index = panel.index
    n = len(index)
    initial_train = int(n * initial_train_fraction)
    folds: list[Fold] = []
    start = initial_train
    fold_id = 1
    while start < n:
        stop = min(start + test_window_hours, n)
        if stop - start >= min_test_hours:
            folds.append(
                Fold(
                    fold=fold_id,
                    train_start=index[0],
                    train_end=index[start - 1],
                    test_start=index[start],
                    test_end=index[stop - 1],
                )
            )
            fold_id += 1
        start += step_hours
    return folds


def training_feature_columns(train_panel: pd.DataFrame) -> list[str]:
    observed = train_panel.notna().sum(axis=0)
    varying = train_panel.nunique(dropna=True) > 1
    keep = observed[observed > 0].index.intersection(varying[varying].index)
    return keep.astype(str).tolist()


def fit_panel_transformers(train_panel: pd.DataFrame) -> tuple[SimpleImputer, StandardScaler, np.ndarray, list[str]]:
    feature_cols = training_feature_columns(train_panel)
    if not feature_cols:
        raise ValueError("No non-empty varying columns available for PCA training.")
    dropped = train_panel.shape[1] - len(feature_cols)
    if dropped:
        logging.info("Dropping %s all-missing or zero-variance training columns before PCA", dropped)
    train_panel = train_panel[feature_cols]
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(train_panel)
    train_scaled = scaler.fit_transform(train_imputed)
    if not np.isfinite(train_scaled).all():
        raise ValueError("Non-finite values found after imputation/scaling.")
    return imputer, scaler, train_scaled, feature_cols


def choose_component_settings(train_scaled: np.ndarray, n_features: int) -> dict[str, int]:
    full = PCA(n_components=None, svd_solver="full")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full.fit(train_scaled)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    settings: dict[str, int] = {}
    for fixed in FIXED_COMPONENTS:
        if fixed <= n_features:
            settings[f"fixed_{fixed}"] = fixed
    for threshold in VARIANCE_THRESHOLDS:
        k = int(np.searchsorted(cumulative, threshold) + 1)
        settings[f"var_{int(threshold * 100)}"] = min(k, n_features)
    return settings


def pca_scores_for_setting(
    train_scaled: np.ndarray,
    all_scaled: np.ndarray,
    *,
    n_components: int,
) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, svd_solver="full")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pca.fit(train_scaled)
        scores = pca.transform(all_scaled)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite values found in PCA scores.")
    return scores, pca


def panel_to_supervised_rows(
    panel_with_pca: pd.DataFrame,
    markets: pd.DataFrame,
    fold: Fold,
    *,
    split: str,
    known_labels_only: bool,
) -> pd.DataFrame:
    if split == "train":
        time_panel = panel_with_pca.loc[fold.train_start : fold.train_end]
        if known_labels_only:
            label_markets = markets[
                markets["Y_i"].notna()
                & markets["close_date"].notna()
                & (markets["close_date"] <= fold.train_end)
            ].copy()
        else:
            label_markets = markets[markets["Y_i"].notna()].copy()
    elif split == "test":
        time_panel = panel_with_pca.loc[fold.test_start : fold.test_end]
        label_markets = markets[markets["Y_i"].notna()].copy()
    else:
        raise ValueError(f"Unknown split: {split}")

    market_ids = [col for col in time_panel.columns if col in set(label_markets["market_id"])]
    if not market_ids:
        return pd.DataFrame()
    pca_cols = [col for col in time_panel.columns if col.startswith("pca_")]
    base = time_panel[market_ids]
    rows = base.stack(future_stack=True).dropna().rename("p_i_t").reset_index()
    rows.columns = ["timestamp", "market_id", "p_i_t"]
    if rows.empty:
        return rows
    pca_frame = time_panel[pca_cols].reset_index().rename(columns={"index": "timestamp"})
    rows = rows.merge(pca_frame, on="timestamp", how="left")
    meta_cols = [
        "market_id",
        "Y_i",
        "market_family",
        "asset",
        "is_core",
        "is_satellite",
        "close_date",
    ]
    rows = rows.merge(label_markets[[col for col in meta_cols if col in label_markets.columns]], on="market_id", how="left")
    rows["split"] = split
    return rows


def probability_clip(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPS, 1 - EPS)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | int]:
    y_pred = probability_clip(y_pred)
    y = y_true.astype(int).to_numpy()
    return {
        "n_obs": int(len(y)),
        "brier": float(brier_score_loss(y, y_pred)),
        "log_loss": float(log_loss(y, y_pred, labels=[0, 1])),
        "avg_pred": float(np.mean(y_pred)),
        "avg_actual": float(np.mean(y)),
    }


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
    grouped = (
        df.groupby(["panel", "fold", "model", "prob_decile"], observed=True)
        .agg(
            n_obs=("Y_i", "size"),
            avg_pred=("p_hat", "mean"),
            avg_actual=("Y_i", "mean"),
        )
        .reset_index()
    )
    return grouped


def run_panel_forecast(
    panel: pd.DataFrame,
    markets: pd.DataFrame,
    *,
    panel_name: str,
    known_labels_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = clean_panel(panel)
    markets = clean_market_universe(markets)
    markets = markets[markets["market_id"].isin(panel.columns)].copy()
    folds = make_chronological_folds(panel)
    all_results: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    supervised_samples: list[pd.DataFrame] = []

    for fold in folds:
        train_panel = panel.loc[fold.train_start : fold.train_end]
        test_panel = panel.loc[fold.test_start : fold.test_end]
        imputer, scaler, train_scaled, pca_feature_cols = fit_panel_transformers(train_panel)
        pca_input_panel = panel[pca_feature_cols]
        all_scaled = scaler.transform(imputer.transform(pca_input_panel))
        if not np.isfinite(all_scaled).all():
            raise ValueError("Non-finite values found in transformed panel.")
        settings = choose_component_settings(train_scaled, len(pca_feature_cols))

        baseline_test = panel_to_supervised_rows(
            panel,
            markets,
            fold,
            split="test",
            known_labels_only=known_labels_only,
        )
        if not baseline_test.empty:
            baseline_pred = baseline_test.copy()
            baseline_pred["panel"] = panel_name
            baseline_pred["fold"] = fold.fold
            baseline_pred["model"] = "market_probability"
            baseline_pred["n_components"] = 0
            baseline_pred["p_hat"] = probability_clip(baseline_pred["p_i_t"])
            all_predictions.append(baseline_pred)
            metrics = evaluate_predictions(baseline_pred["Y_i"], baseline_pred["p_hat"])
            all_results.append(
                {
                    "panel": panel_name,
                    "fold": fold.fold,
                    "model": "market_probability",
                    "n_components": 0,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "status": "ok",
                    **metrics,
                }
            )

        for model_name, n_components in sorted(settings.items(), key=lambda item: (item[1], item[0])):
            scores, pca = pca_scores_for_setting(train_scaled, all_scaled, n_components=n_components)
            pca_cols = [f"pca_{idx + 1}" for idx in range(n_components)]
            pca_frame = pd.DataFrame(scores, index=panel.index, columns=pca_cols)
            panel_with_pca = panel.join(pca_frame)
            train_rows = panel_to_supervised_rows(
                panel_with_pca,
                markets,
                fold,
                split="train",
                known_labels_only=known_labels_only,
            )
            test_rows = panel_to_supervised_rows(
                panel_with_pca,
                markets,
                fold,
                split="test",
                known_labels_only=known_labels_only,
            )
            if train_rows.empty or test_rows.empty or train_rows["Y_i"].nunique() < 2:
                reason = (
                    "empty_train_or_test"
                    if train_rows.empty or test_rows.empty
                    else "single_class_train_labels"
                )
                logging.info(
                    "Skipping %s fold=%s model=%s due to %s",
                    panel_name,
                    fold.fold,
                    model_name,
                    reason,
                )
                all_results.append(
                    {
                        "panel": panel_name,
                        "fold": fold.fold,
                        "model": model_name,
                        "n_components": n_components,
                        "train_start": fold.train_start,
                        "train_end": fold.train_end,
                        "test_start": fold.test_start,
                        "test_end": fold.test_end,
                        "train_rows": int(len(train_rows)),
                        "test_rows": int(len(test_rows)),
                        "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
                        "status": f"skipped:{reason}",
                        "n_obs": 0,
                        "brier": np.nan,
                        "log_loss": np.nan,
                        "avg_pred": np.nan,
                        "avg_actual": np.nan,
                    }
                )
                continue
            feature_cols = ["p_i_t", *pca_cols]
            train_x = train_rows[feature_cols].astype(float)
            test_x = test_rows[feature_cols].astype(float)
            if not np.isfinite(train_x.to_numpy()).all() or not np.isfinite(test_x.to_numpy()).all():
                raise ValueError(f"Non-finite supervised features in {panel_name} fold={fold.fold} model={model_name}")
            model = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                class_weight=None,
                random_state=0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                model.fit(train_x, train_rows["Y_i"].astype(int))
                p_hat = model.predict_proba(test_x)[:, 1]
            if not np.isfinite(p_hat).all():
                raise ValueError(f"Non-finite predictions in {panel_name} fold={fold.fold} model={model_name}")
            pred = test_rows.copy()
            pred["panel"] = panel_name
            pred["fold"] = fold.fold
            pred["model"] = model_name
            pred["n_components"] = n_components
            pred["p_hat"] = probability_clip(p_hat)
            all_predictions.append(pred)
            if model_name in {"var_85", "var_90", "var_95"}:
                supervised_sample = pd.concat(
                    [
                        train_rows.assign(panel=panel_name, fold=fold.fold, model=model_name, n_components=n_components),
                        test_rows.assign(panel=panel_name, fold=fold.fold, model=model_name, n_components=n_components),
                    ],
                    ignore_index=True,
                )
                supervised_samples.append(supervised_sample)
            metrics = evaluate_predictions(pred["Y_i"], pred["p_hat"])
            all_results.append(
                {
                    "panel": panel_name,
                    "fold": fold.fold,
                    "model": model_name,
                    "n_components": n_components,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "train_rows": int(len(train_rows)),
                    "test_rows": int(len(test_rows)),
                    "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
                    "status": "ok",
                    **metrics,
                }
            )

    results = pd.DataFrame(all_results)
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    supervised = pd.concat(supervised_samples, ignore_index=True) if supervised_samples else pd.DataFrame()
    return supervised, predictions, results


def run_all(
    *,
    core_panel_path: Path,
    core_plus_panel_path: Path,
    market_universe_path: Path,
    output_dir: Path,
    known_labels_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dirs([output_dir])
    markets = pd.read_parquet(market_universe_path)
    core = pd.read_parquet(core_panel_path)
    core_plus = pd.read_parquet(core_plus_panel_path)

    supervised_parts = []
    prediction_parts = []
    result_parts = []
    for name, panel in [("core", core), ("core_plus_satellites", core_plus)]:
        supervised, predictions, results = run_panel_forecast(
            panel,
            markets,
            panel_name=name,
            known_labels_only=known_labels_only,
        )
        supervised_parts.append(supervised)
        prediction_parts.append(predictions)
        result_parts.append(results)

    supervised_df = pd.concat(supervised_parts, ignore_index=True) if supervised_parts else pd.DataFrame()
    predictions_df = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    results_df = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()
    calibration_df = calibration_by_decile(predictions_df)

    supervised_df.to_parquet(output_dir / "pca_supervised_dataset.parquet", index=False)
    predictions_df.to_parquet(output_dir / "pca_supervised_predictions.parquet", index=False)
    results_df.to_csv(output_dir / "pca_supervised_forecast_results.csv", index=False)
    calibration_df.to_csv(output_dir / "pca_supervised_calibration_by_decile.csv", index=False)
    return supervised_df, predictions_df, results_df, calibration_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supervised PCA logistic forecasting backtest.")
    parser.add_argument("--core-panel", default="data/processed/panel_hourly_core.parquet")
    parser.add_argument("--core-plus-panel", default="data/processed/panel_hourly_core_plus_satellites.parquet")
    parser.add_argument("--markets", default="data/processed/market_universe.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--known-labels-only",
        action="store_true",
        help="Only train on rows for markets resolved by the fold training cutoff. This is stricter but may leave too few labels.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    root = project_root()
    supervised, predictions, results, calibration = run_all(
        core_panel_path=resolve_path(root, args.core_panel),
        core_plus_panel_path=resolve_path(root, args.core_plus_panel),
        market_universe_path=resolve_path(root, args.markets),
        output_dir=resolve_path(root, args.output_dir),
        known_labels_only=args.known_labels_only,
    )
    logging.info("Saved supervised rows=%s predictions=%s results=%s calibration=%s", len(supervised), len(predictions), len(results), len(calibration))
    if not results.empty:
        print(results.groupby(["panel", "model"], as_index=False).agg(folds=("fold", "nunique"), brier=("brier", "mean"), log_loss=("log_loss", "mean")).to_string(index=False))


if __name__ == "__main__":
    main()
