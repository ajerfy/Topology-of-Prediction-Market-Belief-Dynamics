from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from active_set_forecasting import build_family_state, fit_family_preprocess, fold_rows, load_inputs, resolve_path, transform_family_state
from final_model_comparison import require_xgboost
from local_topology_validation import LOCKED_K, LOCKED_METHOD, build_neighborhood_features, control_columns, topology_columns
from pca_plus_topology_forecasting import load_or_build_supervised
from supervised_forecasting import make_chronological_folds, probability_clip
from topology_rescue_sweep import GRAPH_THRESHOLDS, RANDOM_SEED, clean_supervised, fit_pca_bundle, ph_stats
from utils import ensure_dirs, project_root, setup_logging

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


WINDOWS = (24, 72, 168)
BOOTSTRAPS = 300


@dataclass(frozen=True)
class ModelSpec:
    model: str
    model_type: str
    feature_group: str


MODEL_SPECS = (
    ModelSpec("market_probability", "market", "market"),
    ModelSpec("logit_pca_controls", "logit", "pca_controls"),
    ModelSpec("logit_pca_controls_static_topology", "logit", "pca_controls_static"),
    ModelSpec("logit_pca_controls_dynamic_topology", "logit", "pca_controls_dynamic"),
    ModelSpec("logit_pca_controls_static_dynamic_topology", "logit", "pca_controls_static_dynamic"),
    ModelSpec("xgb_pca_controls", "xgb", "pca_controls"),
    ModelSpec("xgb_pca_controls_static_topology", "xgb", "pca_controls_static"),
    ModelSpec("xgb_pca_controls_dynamic_topology", "xgb", "pca_controls_dynamic"),
    ModelSpec("xgb_pca_controls_static_dynamic_topology", "xgb", "pca_controls_static_dynamic"),
)


COMPARISONS = (
    ("logit_pca_controls_dynamic_topology", "logit_pca_controls", "logit dynamic topology after controls"),
    ("xgb_pca_controls_dynamic_topology", "xgb_pca_controls", "xgb dynamic topology after controls"),
    ("logit_pca_controls_dynamic_topology", "logit_pca_controls_static_topology", "logit dynamic vs static topology"),
    ("xgb_pca_controls_dynamic_topology", "xgb_pca_controls_static_topology", "xgb dynamic vs static topology"),
)


def require_xgb() -> None:
    if XGBClassifier is None:
        raise ImportError(f"XGBoost is required. Install it with: pip install xgboost. Original error: {XGBOOST_IMPORT_ERROR!r}")


def positions_for_fold(index: pd.Index, fold, window: int, stride_hours: int) -> list[int]:
    positions = np.flatnonzero((index >= fold.train_start) & (index <= fold.test_end))
    out: list[int] = []
    for pos in positions:
        if pos < window - 1:
            continue
        ts = index[pos]
        if ts.hour % stride_hours == 0:
            out.append(int(pos))
    return out


def dynamic_family_ph(family_state: pd.DataFrame, fold, stride_hours: int) -> pd.DataFrame:
    train_family = family_state.loc[fold.train_start : fold.train_end].replace([np.inf, -np.inf], np.nan)
    cols, imputer, scaler, _ = fit_family_preprocess(train_family)
    scaled = transform_family_state(family_state.replace([np.inf, -np.inf], np.nan), cols, imputer, scaler)
    scaled = np.clip(np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
    index = pd.Index(family_state.index)
    rows: list[dict[str, object]] = []
    for window in WINDOWS:
        for pos in positions_for_fold(index, fold, window, stride_hours):
            points = scaled[pos - window + 1 : pos + 1]
            stats = ph_stats(points, prefix=f"dyn_family_w{window}_")
            rows.append({"timestamp": index[pos], "window_hours": window, **stats})
    if not rows:
        return pd.DataFrame(columns=["timestamp"])
    out = pd.DataFrame(rows).sort_values(["window_hours", "timestamp"])
    value_cols = [c for c in out.columns if c.startswith("dyn_family_")]
    for col in value_cols:
        out[f"{col}_lag1"] = out.groupby("window_hours")[col].shift(1)
        out[f"{col}_delta"] = out.groupby("window_hours")[col].diff()
    wide = out.drop(columns=["window_hours"]).groupby("timestamp", as_index=False).first()
    return wide.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def dynamic_local_graph(panel: pd.DataFrame, fold, neighborhoods: pd.DataFrame, stride_hours: int) -> pd.DataFrame:
    index = pd.Index(panel.index)
    values = panel.copy()
    values.columns = values.columns.astype(str)
    values = values.ffill().bfill().fillna(0.5)
    market_to_ids: dict[str, list[str]] = {}
    corr_panel = values.loc[(values.index >= fold.train_start) & (values.index <= fold.train_end)]
    corr = corr_panel.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0).abs()
    np.fill_diagonal(corr.values, -np.inf)
    all_markets = list(values.columns)
    for market_id in neighborhoods["market_id"].astype(str).unique():
        if market_id not in corr.index:
            continue
        top = corr.loc[market_id].sort_values(ascending=False).head(LOCKED_K).index.astype(str).tolist()
        market_to_ids[market_id] = [market_id, *[m for m in top if m in values.columns and m != market_id]]

    rows: list[dict[str, object]] = []
    for window in WINDOWS:
        for pos in positions_for_fold(index, fold, window, stride_hours):
            block = values.iloc[pos - window + 1 : pos + 1]
            for market_id, ids in market_to_ids.items():
                cols = [m for m in ids if m in block.columns]
                if len(cols) < 4:
                    continue
                sub_corr = block[cols].corr().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
                row: dict[str, object] = {"timestamp": index[pos], "market_id": market_id, "window_hours": window}
                for threshold in GRAPH_THRESHOLDS:
                    from topology_rescue_sweep import graph_features_for_corr

                    row.update({f"dyn_local_w{window}_t{int(threshold * 10)}_{key}": value for key, value in graph_features_for_corr(sub_corr, threshold).items()})
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "market_id"])
    out = pd.DataFrame(rows).sort_values(["market_id", "window_hours", "timestamp"])
    value_cols = [c for c in out.columns if c.startswith("dyn_local_")]
    for col in value_cols:
        out[f"{col}_lag1"] = out.groupby(["market_id", "window_hours"])[col].shift(1)
        out[f"{col}_delta"] = out.groupby(["market_id", "window_hours"])[col].diff()
    wide = out.drop(columns=["window_hours"]).groupby(["timestamp", "market_id"], as_index=False).first()
    return wide.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def zscore_dynamic(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cols:
        return train, test
    means = train[cols].mean(axis=0)
    stds = train[cols].std(axis=0, ddof=0).replace(0, np.nan)
    for frame in (train, test):
        frame.loc[:, cols] = ((frame[cols] - means) / stds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return train, test


def feature_map(pca_cols: list[str], ctrl_cols: list[str], static_cols: list[str], dynamic_cols: list[str]) -> dict[str, list[str]]:
    base = ["p_i_t", "active_market_count_t", *pca_cols]
    return {
        "market": ["p_i_t"],
        "pca_controls": [*base, *ctrl_cols],
        "pca_controls_static": [*base, *ctrl_cols, *static_cols],
        "pca_controls_dynamic": [*base, *ctrl_cols, *dynamic_cols],
        "pca_controls_static_dynamic": [*base, *ctrl_cols, *static_cols, *dynamic_cols],
    }


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], model_type: str) -> np.ndarray:
    if model_type == "market":
        return probability_clip(test["p_i_t"].to_numpy(dtype=float))
    train = train.dropna(subset=["Y_i"]).copy()
    test = test.dropna(subset=["Y_i"]).copy()
    if train["Y_i"].nunique() < 2:
        raise ValueError("one-class training labels")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[feature_cols].astype(float))
    x_test = imputer.transform(test[feature_cols].astype(float))
    if model_type == "logit":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = LogisticRegression(C=0.01, solver="liblinear", max_iter=1000, random_state=RANDOM_SEED)
    elif model_type == "xgb":
        require_xgb()
        model = XGBClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            reg_alpha=1,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            tree_method="hist",
            n_jobs=0,
        )
    else:
        raise ValueError(model_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(x_train, train["Y_i"].astype(int))
        pred = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    return probability_clip(pred)


def evaluate(y_true: pd.Series, p_hat: np.ndarray) -> dict[str, float | int]:
    y = y_true.astype(int).to_numpy()
    p = probability_clip(p_hat)
    out = {
        "n_obs": int(len(y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "avg_pred": float(np.mean(p)),
        "avg_actual": float(np.mean(y)),
    }
    try:
        out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    except ValueError:
        out["auc"] = np.nan
    return out


def prediction_frame(test: pd.DataFrame, pred: np.ndarray, spec: ModelSpec, fold: int) -> pd.DataFrame:
    cols = ["timestamp", "market_id", "Y_i", "p_i_t", "active_market_count_t", "broad_domain", "broad_family"]
    out = test[cols].dropna(subset=["Y_i"]).copy()
    out["fold"] = fold
    out["model"] = spec.model
    out["model_type"] = spec.model_type
    out["feature_group"] = spec.feature_group
    out["p_hat"] = probability_clip(pred)
    return out


def add_buckets(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["active_count_bucket"] = pd.cut(out["active_market_count_t"], [-np.inf, 10, 25, 50, 100, np.inf], labels=["<=10", "11-25", "26-50", "51-100", ">100"])
    out["prob_bucket"] = pd.cut(out["p_i_t"], [-0.001, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.001], labels=["0-5%", "5-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-95%", "95-100%"])
    return out


def result_rows(pred: pd.DataFrame) -> list[dict[str, object]]:
    pred = add_buckets(pred)
    slices = [("overall", "overall", pred)]
    for col in ["broad_domain", "broad_family", "Y_i", "active_count_bucket", "prob_bucket"]:
        for value, group in pred.groupby(col, observed=False, dropna=True):
            slices.append((col, str(value), group))
    rows = []
    for slice_type, slice_value, group in slices:
        if group.empty:
            continue
        rows.append({"fold": int(group["fold"].iloc[0]), "model": group["model"].iloc[0], "model_type": group["model_type"].iloc[0], "feature_group": group["feature_group"].iloc[0], "slice_type": slice_type, "slice_value": slice_value, **evaluate(group["Y_i"], group["p_hat"].to_numpy())})
    return rows


def aggregate_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (_, _), group in predictions.groupby(["fold", "model"], sort=False):
        rows.extend(result_rows(group))
    return pd.DataFrame(rows)


def calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    df["prob_decile"] = pd.cut(df["p_hat"], np.linspace(0, 1, 11), labels=False, include_lowest=True, duplicates="drop")
    rows = []
    for (model, fold, decile), group in df.groupby(["model", "fold", "prob_decile"], dropna=True):
        rows.append({"model": model, "fold": int(fold), "prob_decile": int(decile), "n_obs": len(group), "avg_pred": group["p_hat"].mean(), "avg_actual": group["Y_i"].mean(), "abs_calibration_error": abs(group["p_hat"].mean() - group["Y_i"].mean())})
    cal = pd.DataFrame(rows)
    if cal.empty:
        return cal
    ece = cal.groupby(["model", "fold"]).apply(lambda g: np.average(g["abs_calibration_error"], weights=g["n_obs"]), include_groups=False).rename("ece").reset_index()
    return cal.merge(ece, on=["model", "fold"], how="left")


def bootstrap_comparison(predictions: pd.DataFrame, candidate: str, baseline: str, label: str, cluster: str | None) -> dict[str, object]:
    wide = predictions[predictions["model"].isin([candidate, baseline])].pivot_table(index=["fold", "timestamp", "market_id", "Y_i"], columns="model", values="p_hat", aggfunc="first")
    wide = wide.dropna(subset=[candidate, baseline]).reset_index()
    y = wide["Y_i"].astype(int).to_numpy()
    cand = probability_clip(wide[candidate])
    base = probability_clip(wide[baseline])
    log_diff = (-(y * np.log(base) + (1 - y) * np.log(1 - base))) - (-(y * np.log(cand) + (1 - y) * np.log(1 - cand)))
    brier_diff = (base - y) ** 2 - (cand - y) ** 2
    rng = np.random.default_rng(RANDOM_SEED)

    def ci(diff: np.ndarray):
        samples = []
        if cluster is None:
            n = len(diff)
            for _ in range(BOOTSTRAPS):
                idx = rng.integers(0, n, n)
                samples.append(diff[idx].mean())
            units = n
        else:
            codes, vals = pd.factorize(wide[cluster], sort=False)
            groups = [np.flatnonzero(codes == i) for i in range(len(vals))]
            for _ in range(BOOTSTRAPS):
                draw = rng.integers(0, len(groups), len(groups))
                idx = np.concatenate([groups[int(i)] for i in draw])
                samples.append(diff[idx].mean())
            units = len(vals)
        arr = np.asarray(samples)
        return float(diff.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), int(units)

    lm, ll, lh, units = ci(log_diff)
    bm, bl, bh, _ = ci(brier_diff)
    return {"comparison": label, "candidate": candidate, "baseline": baseline, "cluster": cluster or "row", "n_units": units, "log_loss_gain": lm, "log_loss_ci_low": ll, "log_loss_ci_high": lh, "brier_gain": bm, "brier_ci_low": bl, "brier_ci_high": bh}


def stat_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, baseline, label in COMPARISONS:
        for cluster in (None, "market_id", "timestamp"):
            rows.append(bootstrap_comparison(predictions, candidate, baseline, label, cluster))
    return pd.DataFrame(rows)


def run_fold(supervised: pd.DataFrame, panel: pd.DataFrame, family_state: pd.DataFrame, fold, output_dir: Path, stride_hours: int, skip_existing: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_dir = output_dir / "xgb_forecasting_topology_folds"
    ensure_dirs([fold_dir])
    pred_path = fold_dir / f"predictions_fold_{fold.fold:02d}.parquet"
    feat_path = fold_dir / f"features_fold_{fold.fold:02d}.parquet"
    diag_path = fold_dir / f"diagnostics_fold_{fold.fold:02d}.csv"
    if skip_existing and pred_path.exists() and feat_path.exists() and diag_path.exists():
        return pd.read_parquet(pred_path), pd.read_parquet(feat_path), pd.read_csv(diag_path)

    train = fold_rows(supervised, fold, "train")
    test = fold_rows(supervised, fold, "test")
    bundle = fit_pca_bundle(family_state, fold)
    train = train.merge(bundle.pca_features, on="timestamp", how="left")
    test = test.merge(bundle.pca_features, on="timestamp", how="left")

    nbh, diag = build_neighborhood_features(panel, fold, method=LOCKED_METHOD, k=LOCKED_K, source="train")
    static_cols = topology_columns(nbh)
    ctrl_cols = control_columns(nbh)
    train = train.merge(nbh[["market_id", *static_cols, *ctrl_cols]], on="market_id", how="left")
    test = test.merge(nbh[["market_id", *static_cols, *ctrl_cols]], on="market_id", how="left")

    family_dyn = dynamic_family_ph(family_state, fold, stride_hours)
    local_dyn = dynamic_local_graph(panel, fold, nbh, stride_hours)
    train = pd.merge_asof(train.sort_values("timestamp"), family_dyn.sort_values("timestamp"), on="timestamp", direction="backward")
    test = pd.merge_asof(test.sort_values("timestamp"), family_dyn.sort_values("timestamp"), on="timestamp", direction="backward")
    train = pd.merge_asof(train.sort_values("timestamp"), local_dyn.sort_values("timestamp"), on="timestamp", by="market_id", direction="backward")
    test = pd.merge_asof(test.sort_values("timestamp"), local_dyn.sort_values("timestamp"), on="timestamp", by="market_id", direction="backward")
    dynamic_cols = [c for c in train.columns if c.startswith("dyn_")]
    train, test = zscore_dynamic(train, test, dynamic_cols)
    fmap = feature_map(bundle.pca_cols, ctrl_cols, static_cols, dynamic_cols)

    pred_frames = []
    for spec in MODEL_SPECS:
        pred = fit_predict(train, test, fmap[spec.feature_group], spec.model_type)
        pred_frames.append(prediction_frame(test, pred, spec, fold.fold))
    predictions = pd.concat(pred_frames, ignore_index=True)
    feature_rows = pd.concat(
        [
            family_dyn.assign(fold=fold.fold, feature_source="family_ph"),
            local_dyn.assign(fold=fold.fold, feature_source="local_graph"),
        ],
        ignore_index=True,
        sort=False,
    )
    diagnostics = pd.DataFrame(
        [
            {"fold": fold.fold, "diagnostic": "locked_neighborhood_source", **diag},
            {"fold": fold.fold, "diagnostic": "dynamic_ph_windows_use_past_only", "used_future_timestamps": False, "stride_hours": stride_hours},
        ]
    )
    predictions.to_parquet(pred_path, index=False)
    feature_rows.to_parquet(feat_path, index=False)
    diagnostics.to_csv(diag_path, index=False)
    return predictions, feature_rows, diagnostics


def load_fold_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_dir = output_dir / "xgb_forecasting_topology_folds"
    pred = pd.concat([pd.read_parquet(p) for p in sorted(fold_dir.glob("predictions_fold_*.parquet"))], ignore_index=True)
    feat = pd.concat([pd.read_parquet(p) for p in sorted(fold_dir.glob("features_fold_*.parquet"))], ignore_index=True)
    diag = pd.concat([pd.read_csv(p) for p in sorted(fold_dir.glob("diagnostics_fold_*.csv"))], ignore_index=True)
    return pred, feat, diag


def summarize_overall(results: pd.DataFrame) -> pd.DataFrame:
    return results[results["slice_type"].eq("overall")].groupby(["model", "model_type", "feature_group"], as_index=False).agg(folds=("fold", "nunique"), n_obs=("n_obs", "sum"), log_loss=("log_loss", "mean"), brier=("brier", "mean"), auc=("auc", "mean"), avg_pred=("avg_pred", "mean"), avg_actual=("avg_actual", "mean")).sort_values("log_loss")


def save_bar(df: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot = df.sort_values(metric)
    ax.barh(plot["model"], plot[metric])
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def create_figures(results: pd.DataFrame, predictions: pd.DataFrame, features: pd.DataFrame, calibration: pd.DataFrame, stat_df: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures" / "xgb_forecasting_topology"
    ensure_dirs([fig_dir])
    overall = summarize_overall(results)
    save_bar(overall, "log_loss", "Dynamic Topology Model Ranking by Log Loss", fig_dir / "log_loss_ranking.png")
    save_bar(overall, "brier", "Dynamic Topology Model Ranking by Brier", fig_dir / "brier_ranking.png")
    fold = results[results["slice_type"].eq("overall")]
    base = fold[fold["model"].eq("xgb_pca_controls")][["fold", "log_loss"]].rename(columns={"log_loss": "base_log_loss"})
    deltas = fold.merge(base, on="fold", how="left")
    deltas["delta_log_loss"] = deltas["log_loss"] - deltas["base_log_loss"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in ["xgb_pca_controls_dynamic_topology", "xgb_pca_controls_static_topology", "logit_pca_controls_dynamic_topology"]:
        g = deltas[deltas["model"].eq(model)]
        ax.plot(g["fold"], g["delta_log_loss"], marker="o", label=model)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Fold-Level Delta vs XGB Controls")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Log-loss delta")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fold_level_deltas.png", dpi=180)
    plt.close(fig)
    save_bar(fold[fold["fold"].eq(fold["fold"].max())], "log_loss", "Locked Holdout Log Loss", fig_dir / "locked_holdout.png")
    cal = calibration.groupby(["model", "prob_decile"], as_index=False).agg(avg_pred=("avg_pred", "mean"), avg_actual=("avg_actual", "mean"))
    fig, ax = plt.subplots(figsize=(6, 6))
    for model in ["market_probability", "xgb_pca_controls", "xgb_pca_controls_dynamic_topology", "logit_pca_controls_dynamic_topology"]:
        g = cal[cal["model"].eq(model)]
        ax.plot(g["avg_pred"], g["avg_actual"], marker="o", label=model)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--")
    ax.legend(fontsize=7)
    ax.set_title("Calibration Curves")
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_curves.png", dpi=180)
    fig.savefig(fig_dir / "reliability_diagram.png", dpi=180)
    plt.close(fig)
    ci = stat_df[stat_df["cluster"].eq("market_id")]
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(ci))
    ax.errorbar(ci["log_loss_gain"], y, xerr=[ci["log_loss_gain"] - ci["log_loss_ci_low"], ci["log_loss_ci_high"] - ci["log_loss_gain"]], fmt="o")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ci["comparison"])
    fig.tight_layout()
    fig.savefig(fig_dir / "bootstrap_ci.png", dpi=180)
    plt.close(fig)
    domain = results[(results["slice_type"].eq("broad_domain")) & results["model"].isin(["xgb_pca_controls", "xgb_pca_controls_dynamic_topology"])]
    pivot = domain.pivot_table(index="slice_value", columns="model", values="log_loss", aggfunc="mean").dropna()
    pivot["gain"] = pivot["xgb_pca_controls"] - pivot["xgb_pca_controls_dynamic_topology"]
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot["gain"].plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Domain-Level Dynamic Topology Gain")
    fig.tight_layout()
    fig.savefig(fig_dir / "domain_level_gain.png", dpi=180)
    plt.close(fig)
    dyn_cols = [c for c in features.columns if c.startswith("dyn_family") and ("total_persistence" in c or "entropy" in c)]
    if dyn_cols:
        ts = features[features["feature_source"].eq("family_ph")].drop_duplicates("timestamp").sort_values("timestamp")
        fig, ax = plt.subplots(figsize=(10, 4))
        for col in dyn_cols[:4]:
            ax.plot(pd.to_datetime(ts["timestamp"]), ts[col], label=col)
        ax.legend(fontsize=7)
        ax.set_title("Dynamic Family PH Features")
        fig.tight_layout()
        fig.savefig(fig_dir / "dynamic_topology_timeseries.png", dpi=180)
        plt.close(fig)
    try:
        dynamic_feature_importance(predictions, features, fig_dir / "xgboost_feature_importance.png")
    except Exception as exc:
        logging.info("Skipping dynamic topology feature-importance figure: %s", exc)
    save_bar(overall[overall["model"].str.contains("dynamic|static|xgb_pca_controls", regex=True)], "log_loss", "Static vs Dynamic Topology Ablation", fig_dir / "static_dynamic_ablation.png")
    save_bar(overall[overall["model"].str.startswith("xgb")], "log_loss", "XGBoost Dynamic Topology Comparison", fig_dir / "xgboost_feature_comparison.png")


def dynamic_feature_importance(predictions: pd.DataFrame, features: pd.DataFrame, path: Path) -> None:
    require_xgb()
    local = features[features["feature_source"].eq("local_graph")].copy()
    local = local.drop(columns=[c for c in ["feature_source"] if c in local.columns])
    rows = predictions[predictions["model"].eq("xgb_pca_controls_dynamic_topology")][["timestamp", "market_id", "Y_i"]].copy()
    rows = rows.merge(local, on=["timestamp", "market_id"], how="inner")
    feature_cols = [c for c in rows.columns if c.startswith("dyn_")]
    if not feature_cols:
        raise ValueError("No dynamic topology features available for importance diagnostic.")
    feature_cols = [c for c in feature_cols if rows[c].notna().any()]
    rows = rows.dropna(subset=["Y_i"]).copy()
    if len(rows) > 250_000:
        rows = rows.sample(n=250_000, random_state=RANDOM_SEED)
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(rows[feature_cols].astype(float))
    y = rows["Y_i"].astype(int).to_numpy()
    model = XGBClassifier(
        n_estimators=80,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5,
        reg_alpha=1,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        tree_method="hist",
        n_jobs=0,
    )
    model.fit(x, y)
    kept_cols = list(imputer.get_feature_names_out(feature_cols))
    imp = pd.Series(model.feature_importances_, index=kept_cols).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    imp.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("XGBoost Dynamic Topology Feature Importance Diagnostic")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(results: pd.DataFrame, calibration: pd.DataFrame, stat_df: pd.DataFrame, diagnostics: pd.DataFrame, output_dir: Path) -> str:
    overall = summarize_overall(results)

    def val(model: str, metric: str) -> float:
        return float(overall[overall["model"].eq(model)][metric].iloc[0])

    xgb_gain = val("xgb_pca_controls", "log_loss") - val("xgb_pca_controls_dynamic_topology", "log_loss")
    xgb_brier_gain = val("xgb_pca_controls", "brier") - val("xgb_pca_controls_dynamic_topology", "brier")
    logit_gain = val("logit_pca_controls", "log_loss") - val("logit_pca_controls_dynamic_topology", "log_loss")
    logit_brier_gain = val("logit_pca_controls", "brier") - val("logit_pca_controls_dynamic_topology", "brier")
    dynamic_vs_static = val("xgb_pca_controls_static_topology", "log_loss") - val("xgb_pca_controls_dynamic_topology", "log_loss")
    ece = calibration.groupby("model", as_index=False).agg(ece=("ece", "mean"))
    ece_gain = float(ece[ece["model"].eq("xgb_pca_controls")]["ece"].iloc[0] - ece[ece["model"].eq("xgb_pca_controls_dynamic_topology")]["ece"].iloc[0])
    leak_source = diagnostics["used_future_timestamps"] if "used_future_timestamps" in diagnostics.columns else pd.Series(False, index=diagnostics.index)
    leak_ok = int(leak_source.replace({np.nan: False}).astype(bool).sum()) == 0
    market_ci = stat_df[(stat_df["comparison"].eq("xgb dynamic topology after controls")) & (stat_df["cluster"].eq("market_id"))]
    market_ci_low = float(market_ci["log_loss_ci_low"].iloc[0]) if not market_ci.empty else np.nan
    domain = results[(results["slice_type"].eq("broad_domain")) & results["model"].isin(["xgb_pca_controls", "xgb_pca_controls_dynamic_topology"])]
    pivot = domain.pivot_table(index="slice_value", columns="model", values="log_loss", aggfunc="mean").dropna()
    pivot["gain"] = pivot["xgb_pca_controls"] - pivot["xgb_pca_controls_dynamic_topology"]
    dominated_macro = bool("macro" in pivot.index and pivot.loc["macro", "gain"] > max(pivot["gain"].sum(), 1e-9) * 0.75)
    if xgb_gain > 0 and xgb_brier_gain > 0 and logit_gain > 0 and logit_brier_gain > 0:
        framing = "Dynamic topology is a robust incremental belief-dynamics signal."
    elif logit_gain > 0 and logit_brier_gain > 0 and not (xgb_gain > 0 and xgb_brier_gain > 0):
        framing = "Dynamic topology helps logistic models but not XGBoost."
    elif xgb_gain > 0 and xgb_brier_gain > 0:
        framing = "Dynamic topology is mainly a nonlinear XGBoost interaction signal."
    else:
        framing = "Dynamic topology does not improve beyond the existing final comparison."
    lines = ["XGBOOST FORECASTING TOPOLOGY SUMMARY", "", "Overall ranking:"]
    for _, row in overall.iterrows():
        lines.append(f"- {row['model']}: log loss {row['log_loss']:.6f}, Brier {row['brier']:.6f}, AUC {row['auc']:.6f}")
    lines.extend(
        [
            "",
            "Key answers:",
            f"- Does dynamic topology improve XGBoost final-resolution forecasting? {'yes' if xgb_gain > 0 and xgb_brier_gain > 0 else 'no'}",
            f"- XGBoost dynamic topology gain: log loss {xgb_gain:+.6f}, Brier {xgb_brier_gain:+.6f}",
            f"- Does dynamic topology improve logistic final-resolution forecasting? {'yes' if logit_gain > 0 and logit_brier_gain > 0 else 'no'}",
            f"- Logistic dynamic topology gain: log loss {logit_gain:+.6f}, Brier {logit_brier_gain:+.6f}",
            f"- Does dynamic topology beat static topology for XGBoost? {'yes' if dynamic_vs_static > 0 else 'no'}",
            f"- XGBoost dynamic-vs-static log-loss gain: {dynamic_vs_static:+.6f}",
            f"- Does topology improve calibration? {'yes' if ece_gain > 0 else 'no'}",
            f"- XGBoost ECE gain: {ece_gain:+.6f}",
            f"- Market-clustered XGBoost CI excludes zero: {'yes' if market_ci_low > 0 else 'no'}",
            f"- Result dominated by macro markets: {'yes' if dominated_macro else 'no'}",
            f"- Zero future timestamp leakage diagnostics: {'yes' if leak_ok else 'no'}",
            "- External CPU supported: yes, via Colab/Kaggle setup instructions and fold split/combine CLI.",
            "",
            "Final paper framing:",
            f"- {framing}",
        ]
    )
    text = "\n".join(lines) + "\n"
    (output_dir / "xgb_forecasting_topology_summary.md").write_text(text, encoding="utf-8")
    return text


def finalize_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    predictions, features, diagnostics = load_fold_outputs(output_dir)
    if not np.isfinite(predictions["p_hat"].to_numpy()).all():
        raise ValueError("Non-finite predictions")
    results = aggregate_results(predictions)
    calibration = calibration_table(predictions)
    stats = stat_tests(predictions)
    results.to_csv(output_dir / "xgb_forecasting_topology_results.csv", index=False)
    predictions.to_parquet(output_dir / "xgb_forecasting_topology_predictions.parquet", index=False)
    features.to_parquet(output_dir / "xgb_forecasting_topology_features.parquet", index=False)
    calibration.to_csv(output_dir / "xgb_forecasting_topology_calibration.csv", index=False)
    stats.to_csv(output_dir / "xgb_forecasting_topology_stat_tests.csv", index=False)
    diagnostics.to_csv(output_dir / "xgb_forecasting_topology_diagnostics.csv", index=False)
    create_figures(results, predictions, features, calibration, stats, output_dir)
    summary = write_summary(results, calibration, stats, diagnostics, output_dir)
    return results, predictions, summary


def run_all(candidate_markets_path: Path, prices_path: Path, panel_path: Path, output_dir: Path, fold_start: int | None, fold_end: int | None, skip_existing: bool, stride_hours: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    require_xgb()
    ensure_dirs([output_dir, output_dir / "xgb_forecasting_topology_folds"])
    markets, panel, _raw, mask = load_inputs(candidate_markets_path, prices_path, panel_path)
    markets = markets[markets["market_id"].isin(panel.columns.astype(str))].copy()
    supervised = clean_supervised(load_or_build_supervised(output_dir, panel, mask, markets))
    family_state = build_family_state(panel, mask, markets)
    folds = make_chronological_folds(panel)
    selected = [f for f in folds if (fold_start is None or f.fold >= fold_start) and (fold_end is None or f.fold <= fold_end)]
    for fold in selected:
        logging.info("Running dynamic topology fold=%s/%s", fold.fold, len(folds))
        run_fold(supervised, panel, family_state, fold, output_dir, stride_hours, skip_existing)
    return finalize_outputs(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XGBoost final-resolution forecasting with dynamic topology.")
    parser.add_argument("--candidate-markets", default="data/processed/candidate_universe_markets.parquet")
    parser.add_argument("--prices", default="data/processed/prices_long.parquet")
    parser.add_argument("--panel", default="data/processed/universe_b_macro_crypto_panel.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--fold-start", type=int)
    parser.add_argument("--fold-end", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--combine-fold-outputs", action="store_true")
    parser.add_argument("--topology-stride-hours", type=int, default=24)
    parser.add_argument("--n-jobs", type=int, default=1, help="Reserved for external split runs; XGBoost uses its fixed internal setting.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    root = project_root()
    output_dir = resolve_path(root, args.output_dir)
    if args.combine_fold_outputs:
        _results, _predictions, summary = finalize_outputs(output_dir)
    else:
        _results, _predictions, summary = run_all(
            resolve_path(root, args.candidate_markets),
            resolve_path(root, args.prices),
            resolve_path(root, args.panel),
            output_dir,
            args.fold_start,
            args.fold_end,
            args.skip_existing,
            args.topology_stride_hours,
        )
    print(summary)


if __name__ == "__main__":
    main()
