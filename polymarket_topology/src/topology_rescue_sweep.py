from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from active_set_forecasting import build_family_state, fit_family_preprocess, fold_rows, load_inputs, resolve_path, transform_family_state
from pca_plus_topology_forecasting import load_or_build_supervised
from persistent_homology_forecasting import compute_window_diagrams, diagram_stats
from supervised_forecasting import Fold, make_chronological_folds, probability_clip
from utils import ensure_dirs, project_root, setup_logging


PCA_COMPONENTS = 5
WINDOWS = (24, 72)
STRIDE_HOURS = 6
RANDOM_SEED = 23
MIN_GROUP_MARKETS = 3
TOPO_COLS = [
    "h0_num_components",
    "h0_total_persistence",
    "h0_max_persistence",
    "h0_entropy",
    "h1_num_loops",
    "h1_total_persistence",
    "h1_max_persistence",
    "h1_entropy",
]
GRAPH_THRESHOLDS = (0.3, 0.5, 0.7)


@dataclass(frozen=True)
class FeatureBundle:
    pca_features: pd.DataFrame
    residuals: np.ndarray
    pca_cols: list[str]


def clean_supervised(supervised: pd.DataFrame) -> pd.DataFrame:
    df = supervised.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["market_id"] = df["market_id"].astype(str)
    df["p_i_t"] = probability_clip(df["p_i_t"])
    df["Y_i"] = df["Y_i"].astype(int)
    return df.sort_values(["timestamp", "market_id"]).reset_index(drop=True)


def fit_pca_bundle(family_state: pd.DataFrame, fold: Fold) -> FeatureBundle:
    train_family = family_state.loc[fold.train_start : fold.train_end].replace([np.inf, -np.inf], np.nan)
    cols, imputer, scaler, train_scaled = fit_family_preprocess(train_family)
    all_scaled = transform_family_state(family_state.replace([np.inf, -np.inf], np.nan), cols, imputer, scaler)
    n_components = min(PCA_COMPONENTS, train_scaled.shape[0], train_scaled.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pca = PCA(n_components=n_components, svd_solver="full").fit(train_scaled)
        scores = pca.transform(all_scaled)
        reconstructed = pca.inverse_transform(scores)
    residuals = np.nan_to_num(all_scaled - reconstructed, nan=0.0, posinf=0.0, neginf=0.0)
    residuals = np.clip(residuals, -10.0, 10.0)
    pca_cols = [f"pca_{idx + 1}" for idx in range(n_components)]
    pca_features = pd.DataFrame(scores, columns=pca_cols, index=family_state.index).reset_index().rename(columns={"index": "timestamp"})
    return FeatureBundle(pca_features, residuals, pca_cols)


def timestamp_positions(index: pd.Index, fold: Fold, window: int, stride_hours: int = STRIDE_HOURS) -> list[int]:
    positions = np.flatnonzero((index >= fold.train_start) & (index <= fold.test_end))
    out = []
    for pos in positions:
        if pos < window - 1:
            continue
        ts = index[pos]
        if ts.hour % stride_hours == 0:
            out.append(int(pos))
    return out


def ph_stats(points: np.ndarray, prefix: str = "") -> dict[str, float]:
    if points.shape[0] < 3 or points.shape[1] < 1:
        return {f"{prefix}{col}": 0.0 for col in TOPO_COLS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        diagrams = compute_window_diagrams(points)
    stats = {**diagram_stats(diagrams[0], "h0"), **diagram_stats(diagrams[1], "h1")}
    return {f"{prefix}{key}": float(value) for key, value in stats.items()}


def build_global_residual_topology(family_state: pd.DataFrame, residuals: np.ndarray, fold: Fold) -> pd.DataFrame:
    index = pd.Index(family_state.index)
    rows = []
    for window in WINDOWS:
        for pos in timestamp_positions(index, fold, window):
            points = residuals[pos - window + 1 : pos + 1]
            rows.append({"timestamp": index[pos], "window_hours": window, **ph_stats(points, "resid_")})
    return pd.DataFrame(rows)


def build_local_topology(panel: pd.DataFrame, markets: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    index = pd.Index(panel.index)
    meta = markets.set_index("market_id")
    panel = panel.copy()
    panel.columns = panel.columns.astype(str)
    rows: list[dict[str, object]] = []

    group_specs = []
    for col in ["broad_family", "broad_domain"]:
        for group, ids in meta.groupby(col).groups.items():
            market_ids = [mid for mid in ids if mid in panel.columns]
            if len(market_ids) >= MIN_GROUP_MARKETS:
                group_specs.append((col, str(group), market_ids))

    for window in (24,):
        positions = timestamp_positions(index, fold, window)
        for group_type, group_name, ids in group_specs:
            values = panel[ids].ffill().bfill()
            for pos in positions:
                block = values.iloc[pos - window + 1 : pos + 1].dropna(axis=1, how="all")
                if block.shape[1] < MIN_GROUP_MARKETS:
                    continue
                arr = block.fillna(block.mean()).fillna(0.5).to_numpy(dtype=float)
                rows.append(
                    {
                        "timestamp": index[pos],
                        "local_group_type": group_type,
                        "local_group": group_name,
                        "window_hours": window,
                        **ph_stats(arr, "local_"),
                    }
                )
    return pd.DataFrame(rows)


def graph_features_for_corr(corr: np.ndarray, threshold: float) -> dict[str, float]:
    n = corr.shape[0]
    if n == 0:
        return {"components": 0.0, "cycle_rank": 0.0, "edge_density": 0.0, "avg_degree": 0.0, "clustering": 0.0}
    adj = (np.abs(corr) >= threshold).astype(int)
    np.fill_diagonal(adj, 0)
    edges = int(adj.sum() // 2)
    seen = np.zeros(n, dtype=bool)
    components = 0
    for start in range(n):
        if seen[start]:
            continue
        components += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            for nxt in np.flatnonzero(adj[node]):
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(int(nxt))
    degrees = adj.sum(axis=1)
    triangles = float(np.trace(adj @ adj @ adj) / 6.0)
    triples = float(np.sum(degrees * (degrees - 1) / 2.0))
    clustering = 0.0 if triples <= 0 else float(3.0 * triangles / triples)
    return {
        "components": float(components),
        "cycle_rank": float(max(edges - n + components, 0)),
        "edge_density": float(edges / max(n * (n - 1) / 2.0, 1.0)),
        "avg_degree": float(degrees.mean()) if n else 0.0,
        "clustering": clustering,
    }


def build_graph_topology(panel: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    index = pd.Index(panel.index)
    rows = []
    values = panel.ffill().bfill()
    for window in (72,):
        for pos in timestamp_positions(index, fold, window):
            block = values.iloc[pos - window + 1 : pos + 1].dropna(axis=1, how="all")
            block = block.loc[:, block.nunique(dropna=True) > 1]
            if block.shape[1] < 4:
                continue
            corr = block.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            row = {"timestamp": index[pos], "window_hours": window}
            for threshold in GRAPH_THRESHOLDS:
                stats = graph_features_for_corr(corr, threshold)
                row.update({f"graph_t{int(threshold * 10)}_{key}": value for key, value in stats.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def asof_merge(left: pd.DataFrame, right: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    left = left.sort_values("timestamp")
    right = right.sort_values("timestamp")
    if by:
        return pd.merge_asof(left, right, on="timestamp", by=by, direction="backward")
    return pd.merge_asof(left, right, on="timestamp", direction="backward")


def scale_fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str,
    task: str,
    model_type: str = "logistic",
) -> np.ndarray:
    train = train.dropna(subset=[target_col]).copy()
    test = test.dropna(subset=[target_col]).copy()
    if train.empty or test.empty:
        raise ValueError("empty train/test")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[feature_cols].astype(float)))
    x_test = scaler.transform(imputer.transform(test[feature_cols].astype(float)))
    if task == "regression":
        model = Ridge(alpha=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(x_train, train[target_col].astype(float))
            pred = np.asarray(model.predict(x_test), dtype=float)
        if not np.isfinite(pred).all():
            raise ValueError("non-finite regression predictions")
        return pred
    if train[target_col].nunique() < 2:
        raise ValueError("one-class training labels")
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=80,
                max_depth=2,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
            )
        except Exception:
            model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=8, learning_rate=0.04, random_state=RANDOM_SEED)
    else:
        model = LogisticRegression(C=0.01, solver="liblinear", max_iter=1000, random_state=RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(x_train, train[target_col].astype(int))
        pred = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("non-finite classification predictions")
    return probability_clip(pred)


def evaluate_predictions(y_true: pd.Series, pred: np.ndarray, task: str) -> dict[str, float | int]:
    y = y_true.to_numpy()
    if task == "regression":
        return {
            "n_obs": int(len(y)),
            "metric_primary": float(mean_squared_error(y, pred)),
            "mse": float(mean_squared_error(y, pred)),
            "mae": float(mean_absolute_error(y, pred)),
            "brier": np.nan,
            "log_loss": np.nan,
            "auc": np.nan,
        }
    p = probability_clip(pred)
    out = {
        "n_obs": int(len(y)),
        "metric_primary": float(log_loss(y.astype(int), p, labels=[0, 1])),
        "brier": float(brier_score_loss(y.astype(int), p)),
        "log_loss": float(log_loss(y.astype(int), p, labels=[0, 1])),
        "mse": np.nan,
        "mae": np.nan,
    }
    try:
        out["auc"] = float(roc_auc_score(y.astype(int), p)) if len(np.unique(y)) > 1 else np.nan
    except ValueError:
        out["auc"] = np.nan
    return out


def result_row(
    *,
    hypothesis: str,
    target: str,
    task: str,
    fold: int,
    model: str,
    feature_set: str,
    placebo_type: str,
    y_true: pd.Series,
    pred: np.ndarray,
) -> dict[str, object]:
    return {
        "hypothesis": hypothesis,
        "target": target,
        "task": task,
        "fold": fold,
        "model": model,
        "feature_set": feature_set,
        "placebo_type": placebo_type,
        **evaluate_predictions(y_true, pred, task),
    }


def add_future_targets(rows: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    long = panel.stack(future_stack=True).rename("price").reset_index()
    long.columns = ["timestamp", "market_id", "price"]
    long["market_id"] = long["market_id"].astype(str)
    for horizon in (24, 72):
        shifted = long.copy()
        shifted["timestamp"] = shifted["timestamp"] - pd.Timedelta(hours=horizon)
        shifted = shifted.rename(columns={"price": f"future_price_{horizon}h"})
        out = out.merge(shifted[["timestamp", "market_id", f"future_price_{horizon}h"]], on=["timestamp", "market_id"], how="left")
        out[f"abs_move_{horizon}h"] = (out[f"future_price_{horizon}h"] - out["p_i_t"]).abs()
        out[f"large_move_{horizon}h"] = (out[f"abs_move_{horizon}h"] >= 0.10).astype(float)
    out["market_abs_error"] = (out["p_i_t"] - out["Y_i"]).abs()
    out["market_squared_error"] = (out["p_i_t"] - out["Y_i"]) ** 2
    out["miscalibration_indicator"] = (out["market_abs_error"] >= 0.25).astype(float)
    return out


def make_placebo_features(frame: pd.DataFrame, cols: list[str], fold: Fold, kind: str) -> pd.DataFrame:
    out = frame.copy()
    if kind == "shuffle":
        shuffled = out[cols].sample(frac=1.0, random_state=RANDOM_SEED + fold.fold).reset_index(drop=True)
        out.loc[:, cols] = shuffled.to_numpy()
    elif kind == "future_shift":
        out["timestamp"] = out["timestamp"] - pd.Timedelta(hours=72)
    return out


def fold_feature_frames(
    panel: pd.DataFrame,
    markets: pd.DataFrame,
    family_state: pd.DataFrame,
    fold: Fold,
) -> tuple[FeatureBundle, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bundle = fit_pca_bundle(family_state, fold)
    residual_topo = build_global_residual_topology(family_state, bundle.residuals, fold)
    graph_topo = build_graph_topology(panel, fold)
    local_topo = build_local_topology(panel, markets, fold)
    change_topo = residual_topo.sort_values(["window_hours", "timestamp"]).copy()
    for col in [c for c in change_topo.columns if c.startswith("resid_")]:
        change_topo[f"delta_{col}"] = change_topo.groupby("window_hours")[col].diff().fillna(0.0)
    return bundle, residual_topo, graph_topo, local_topo, change_topo


def run_fold(
    supervised: pd.DataFrame,
    panel: pd.DataFrame,
    markets: pd.DataFrame,
    family_state: pd.DataFrame,
    fold: Fold,
) -> list[dict[str, object]]:
    rows = add_future_targets(supervised, panel)
    train = fold_rows(rows, fold, "train")
    test = fold_rows(rows, fold, "test")
    bundle, residual_topo, graph_topo, local_topo, change_topo = fold_feature_frames(panel, markets, family_state, fold)

    train = train.merge(bundle.pca_features, on="timestamp", how="left")
    test = test.merge(bundle.pca_features, on="timestamp", how="left")
    base_cols = ["p_i_t", "active_market_count_t"]
    pca_cols = [*base_cols, *bundle.pca_cols]
    result_rows: list[dict[str, object]] = []

    def run_binary(hypothesis: str, target: str, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], model: str, feature_set: str, placebo_type: str = "real", model_type: str = "logistic") -> None:
        try:
            pred = scale_fit_predict(train_df, test_df, feature_cols, target_col=target, task="classification", model_type=model_type)
            y = test_df.dropna(subset=[target])[target]
            result_rows.append(result_row(hypothesis=hypothesis, target=target, task="classification", fold=fold.fold, model=model, feature_set=feature_set, placebo_type=placebo_type, y_true=y, pred=pred))
        except ValueError as exc:
            logging.info("Skipping %s/%s fold=%s: %s", hypothesis, model, fold.fold, exc)

    def run_regression(hypothesis: str, target: str, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], model: str, feature_set: str, placebo_type: str = "real") -> None:
        try:
            pred = scale_fit_predict(train_df, test_df, feature_cols, target_col=target, task="regression")
            y = test_df.dropna(subset=[target])[target]
            result_rows.append(result_row(hypothesis=hypothesis, target=target, task="regression", fold=fold.fold, model=model, feature_set=feature_set, placebo_type=placebo_type, y_true=y, pred=pred))
        except ValueError as exc:
            logging.info("Skipping %s/%s fold=%s: %s", hypothesis, model, fold.fold, exc)

    run_binary("baseline", "Y_i", train, test, base_cols, "market_only", "market_probability", "none")
    run_binary("baseline", "Y_i", train, test, pca_cols, "pca_only", "pca", "none")

    # Hypothesis 1: local family/domain topology attached to matching market rows.
    for group_type, meta_col in [("broad_family", "broad_family"), ("broad_domain", "broad_domain")]:
        local = local_topo[local_topo["local_group_type"].eq(group_type)].copy()
        if local.empty:
            continue
        local_cols = [c for c in local.columns if c.startswith("local_") and c not in {"local_group", "local_group_type"}]
        train_l = asof_merge(train.sort_values("timestamp"), local.rename(columns={"local_group": meta_col}).sort_values("timestamp"), by=meta_col)
        test_l = asof_merge(test.sort_values("timestamp"), local.rename(columns={"local_group": meta_col}).sort_values("timestamp"), by=meta_col)
        features = [*pca_cols, *local_cols]
        run_binary("H1_local_topology", "Y_i", train_l, test_l, features, f"pca_plus_local_{group_type}_ph", f"local_{group_type}")
        for placebo in ("shuffle", "future_shift"):
            pframe = make_placebo_features(local.rename(columns={"local_group": meta_col}), local_cols, fold, placebo)
            train_p = asof_merge(train.sort_values("timestamp"), pframe.sort_values("timestamp"), by=meta_col)
            test_p = asof_merge(test.sort_values("timestamp"), pframe.sort_values("timestamp"), by=meta_col)
            run_binary("H1_local_topology", "Y_i", train_p, test_p, features, f"pca_plus_local_{group_type}_ph_{placebo}", f"local_{group_type}", placebo)

    # Hypothesis 2: topology/proxies of rolling correlation graphs.
    graph_cols = [c for c in graph_topo.columns if c.startswith("graph_")]
    train_g = asof_merge(train, graph_topo)
    test_g = asof_merge(test, graph_topo)
    run_binary("H2_graph_topology", "Y_i", train_g, test_g, [*pca_cols, *graph_cols], "pca_plus_graph_topology", "graph_topology")
    for placebo in ("shuffle", "future_shift"):
        pframe = make_placebo_features(graph_topo, graph_cols, fold, placebo)
        run_binary("H2_graph_topology", "Y_i", asof_merge(train, pframe), asof_merge(test, pframe), [*pca_cols, *graph_cols], f"pca_plus_graph_topology_{placebo}", "graph_topology", placebo)

    # Hypothesis 3: topology as regime/dynamics predictor.
    resid_cols = [c for c in residual_topo.columns if c.startswith("resid_")]
    train_r = asof_merge(train, residual_topo)
    test_r = asof_merge(test, residual_topo)
    for target in ("large_move_24h", "large_move_72h"):
        run_binary("H3_regime_volatility", target, train, test, pca_cols, f"pca_{target}", "pca", "none")
        run_binary("H3_regime_volatility", target, train_r, test_r, [*pca_cols, *resid_cols], f"pca_plus_topology_{target}", "residual_topology")
    for target in ("abs_move_24h", "abs_move_72h"):
        run_regression("H3_regime_volatility", target, train, test, pca_cols, f"pca_{target}", "pca", "none")
        run_regression("H3_regime_volatility", target, train_r, test_r, [*pca_cols, *resid_cols], f"pca_plus_topology_{target}", "residual_topology")

    # Hypothesis 4: topology change features.
    delta_cols = [c for c in change_topo.columns if c.startswith("delta_resid_")]
    train_c = asof_merge(train, change_topo[["timestamp", *delta_cols]])
    test_c = asof_merge(test, change_topo[["timestamp", *delta_cols]])
    run_binary("H4_topology_change", "Y_i", train_c, test_c, [*pca_cols, *delta_cols], "pca_plus_topology_change", "topology_change")
    for placebo in ("shuffle", "future_shift"):
        pframe = make_placebo_features(change_topo[["timestamp", *delta_cols]], delta_cols, fold, placebo)
        run_binary("H4_topology_change", "Y_i", asof_merge(train, pframe), asof_merge(test, pframe), [*pca_cols, *delta_cols], f"pca_plus_topology_change_{placebo}", "topology_change", placebo)

    # Hypothesis 5: interactions and nonlinear classifier.
    inter = train_r.copy()
    inter_test = test_r.copy()
    for col in resid_cols[:4]:
        for base in ["p_i_t", "active_market_count_t", *bundle.pca_cols[:2]]:
            inter[f"{base}_x_{col}"] = inter[base] * inter[col]
            inter_test[f"{base}_x_{col}"] = inter_test[base] * inter_test[col]
    inter_cols = [c for c in inter.columns if "_x_resid_" in c]
    run_binary("H5_interactions", "Y_i", inter, inter_test, [*pca_cols, *resid_cols[:4], *inter_cols], "logistic_interactions", "topology_interactions")
    run_binary("H5_interactions", "Y_i", inter, inter_test, [*pca_cols, *resid_cols[:4], *inter_cols], "xgboost_interactions", "topology_interactions", model_type="xgboost")

    # Hypothesis 6: topology as uncertainty/error signal.
    for target, task in [("miscalibration_indicator", "classification"), ("market_abs_error", "regression"), ("market_squared_error", "regression")]:
        if task == "classification":
            run_binary("H6_uncertainty_signal", target, train, test, pca_cols, f"pca_error_{target}", "pca", "none")
            run_binary("H6_uncertainty_signal", target, train_r, test_r, [*pca_cols, *resid_cols], f"pca_plus_topology_error_{target}", "residual_topology")
        else:
            run_regression("H6_uncertainty_signal", target, train, test, pca_cols, f"pca_error_{target}", "pca", "none")
            run_regression("H6_uncertainty_signal", target, train_r, test_r, [*pca_cols, *resid_cols], f"pca_plus_topology_error_{target}", "residual_topology")

    return result_rows


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    return (
        results.groupby(["hypothesis", "target", "task", "model", "feature_set", "placebo_type"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            n_obs=("n_obs", "sum"),
            metric_primary=("metric_primary", "mean"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
            mse=("mse", "mean"),
            mae=("mae", "mean"),
            auc=("auc", "mean"),
        )
        .sort_values(["hypothesis", "target", "metric_primary"])
    )


def write_summary(results: pd.DataFrame, output_dir: Path) -> str:
    summary = summarize_results(results)
    if summary.empty:
        text = "TOPOLOGY RESCUE SWEEP SUMMARY\n\nNo valid results.\n"
        (output_dir / "topology_rescue_summary.md").write_text(text, encoding="utf-8")
        return text

    lines = ["TOPOLOGY RESCUE SWEEP SUMMARY", ""]
    best_by_hypothesis = []
    for hypothesis, group in summary[~summary["hypothesis"].eq("baseline")].groupby("hypothesis"):
        topology_group = group[~group["feature_set"].isin(["pca", "market_probability"])].copy()
        real = topology_group[~topology_group["placebo_type"].isin(["shuffle", "future_shift"])].copy()
        if real.empty:
            continue
        best = real.sort_values("metric_primary").iloc[0]
        best_by_hypothesis.append(best)
        lines.append(f"{hypothesis}:")
        metric_name = "log loss" if best["task"] == "classification" else "MSE"
        lines.append(f"- best: {best['model']} on {best['target']} ({metric_name} {best['metric_primary']:.6f}, folds {int(best['folds'])})")
        same_target = summary["target"].eq(best["target"])
        base = summary[summary["hypothesis"].eq("baseline") & same_target]
        pca = summary[summary["feature_set"].eq("pca") & same_target & summary["placebo_type"].eq("none")]
        comparator = pca.sort_values("metric_primary").iloc[0] if not pca.empty else (base.sort_values("metric_primary").iloc[0] if not base.empty else None)
        if comparator is not None:
            lines.append(f"- delta vs best PCA comparator: {best['metric_primary'] - comparator['metric_primary']:+.6f}")
        placebos = topology_group[topology_group["placebo_type"].isin(["shuffle", "future_shift"])]
        if not placebos.empty:
            best_placebo = placebos.sort_values("metric_primary").iloc[0]
            lines.append(f"- beats best placebo: {'yes' if best['metric_primary'] < best_placebo['metric_primary'] else 'no'}")
        else:
            lines.append("- beats best placebo: not tested")
        lines.append("")

    best_frame = pd.DataFrame(best_by_hypothesis)
    primary_best = (
        best_frame[best_frame["target"].eq("Y_i")].sort_values("metric_primary").iloc[0]
        if not best_frame.empty and best_frame["target"].eq("Y_i").any()
        else pd.Series(dtype=object)
    )
    auxiliary_best = (
        best_frame[~best_frame["target"].eq("Y_i")].sort_values("metric_primary").iloc[0]
        if not best_frame.empty and (~best_frame["target"].eq("Y_i")).any()
        else pd.Series(dtype=object)
    )
    best_overall = primary_best if not primary_best.empty else (auxiliary_best if not auxiliary_best.empty else pd.Series(dtype=object))
    role_map = {
        "H1_local_topology": "B) local topology",
        "H2_graph_topology": "C) graph topology",
        "H3_regime_volatility": "D) volatility/regime forecasting",
        "H4_topology_change": "A) direct forecasting",
        "H5_interactions": "A) direct forecasting",
        "H6_uncertainty_signal": "E) uncertainty/error signal",
    }
    role = role_map.get(str(best_overall.get("hypothesis", "")), "F) none") if not best_overall.empty else "F) none"
    robust = False
    if not best_overall.empty:
        group = summary[(summary["hypothesis"].eq(best_overall["hypothesis"])) & (summary["target"].eq(best_overall["target"]))]
        placebos = group[group["placebo_type"].isin(["shuffle", "future_shift"])]
        robust = placebos.empty or float(best_overall["metric_primary"]) < float(placebos["metric_primary"].min())
    if robust and not best_overall.empty and float(best_overall.get("metric_primary", np.inf)) < np.inf:
        recommendation = "B) refine topology"
    else:
        recommendation = "C) use topology as diagnostic" if role != "F) none" else "D) stop topology"

    lines.extend(
        [
            "Best primary-outcome topology result:",
            f"- {primary_best.get('hypothesis', 'none')} / {primary_best.get('model', 'none')} / target {primary_best.get('target', 'none')} / metric {primary_best.get('metric_primary', np.nan):.6f}" if not primary_best.empty else "- none",
            "",
            "Best auxiliary topology result:",
            f"- {auxiliary_best.get('hypothesis', 'none')} / {auxiliary_best.get('model', 'none')} / target {auxiliary_best.get('target', 'none')} / metric {auxiliary_best.get('metric_primary', np.nan):.6f}" if not auxiliary_best.empty else "- none",
            f"- survives placebo checks: {'yes' if robust else 'no'}",
            "",
            "Most promising topology role:",
            f"- {role}",
            "",
            "Final recommendation:",
            f"- {recommendation}",
        ]
    )
    text = "\n".join(lines) + "\n"
    (output_dir / "topology_rescue_summary.md").write_text(text, encoding="utf-8")
    return text


def run_all(candidate_markets_path: Path, prices_path: Path, panel_path: Path, output_dir: Path) -> tuple[pd.DataFrame, str]:
    ensure_dirs([output_dir])
    markets, panel, _raw, mask = load_inputs(candidate_markets_path, prices_path, panel_path)
    markets = markets[markets["market_id"].isin(panel.columns.astype(str))].copy()
    supervised = clean_supervised(load_or_build_supervised(output_dir, panel, mask, markets))
    family_state = build_family_state(panel, mask, markets)
    folds = make_chronological_folds(panel)
    rows: list[dict[str, object]] = []
    for fold in folds:
        logging.info("Running topology rescue sweep fold=%s/%s", fold.fold, len(folds))
        rows.extend(run_fold(supervised, panel, markets, family_state, fold))
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "topology_rescue_results.csv", index=False)
    summary = write_summary(results, output_dir)
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded topology rescue sweep.")
    parser.add_argument("--candidate-markets", default="data/processed/candidate_universe_markets.parquet")
    parser.add_argument("--prices", default="data/processed/prices_long.parquet")
    parser.add_argument("--panel", default="data/processed/universe_b_macro_crypto_panel.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    root = project_root()
    _results, summary = run_all(
        resolve_path(root, args.candidate_markets),
        resolve_path(root, args.prices),
        resolve_path(root, args.panel),
        resolve_path(root, args.output_dir),
    )
    print(summary)


if __name__ == "__main__":
    main()
