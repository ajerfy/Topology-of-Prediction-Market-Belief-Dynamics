from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from active_set_forecasting import build_family_state, fold_rows, load_inputs, resolve_path
from pca_plus_topology_forecasting import load_or_build_supervised
from supervised_forecasting import Fold, make_chronological_folds, probability_clip
from topology_rescue_sweep import GRAPH_THRESHOLDS, RANDOM_SEED, clean_supervised, fit_pca_bundle, graph_features_for_corr
from utils import ensure_dirs, project_root, setup_logging


LOCKED_METHOD = "top_corr"
LOCKED_K = 20
SENSITIVITY_K = (10, 40)
MIN_LOG_LOSS_GAIN = 0.001
MIN_BRIER_GAIN = 0.0005
BOOTSTRAPS = 300


@dataclass(frozen=True)
class ModelSpec:
    model: str
    feature_set: str
    feature_cols: list[str]
    category: str
    is_confirmatory: bool = False
    is_valid_placebo: bool = False
    is_leakage_control: bool = False


def offdiag_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return np.array([], dtype=float)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def build_neighborhood_features(
    panel: pd.DataFrame,
    fold: Fold,
    *,
    method: str,
    k: int,
    source: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    panel = panel.copy()
    panel.columns = panel.columns.astype(str)
    if source == "train":
        source_panel = panel.loc[(panel.index >= fold.train_start) & (panel.index <= fold.train_end)]
        used_test = False
    elif source == "test":
        source_panel = panel.loc[(panel.index >= fold.test_start) & (panel.index <= fold.test_end)]
        used_test = True
    else:
        raise ValueError(source)

    values = source_panel.ffill().bfill().fillna(0.5)
    usable = values.loc[:, values.nunique(dropna=True) > 1]
    if usable.shape[1] < 5:
        return pd.DataFrame(), {"used_test_timestamps": used_test, "n_usable_markets": int(usable.shape[1])}

    corr = usable.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    abs_corr = corr.abs()
    np.fill_diagonal(abs_corr.values, -np.inf)
    markets = list(usable.columns)
    rng = np.random.default_rng(RANDOM_SEED + fold.fold + k)
    rows: list[dict[str, object]] = []

    for idx, market_id in enumerate(markets):
        k_eff = min(k, max(len(markets) - 1, 1))
        if method == "top_corr":
            neighbor_idx = np.argsort(-abs_corr.iloc[idx].to_numpy())[:k_eff]
        elif method == "random":
            candidates = [pos for pos in range(len(markets)) if pos != idx]
            neighbor_idx = rng.choice(candidates, size=k_eff, replace=False)
        else:
            raise ValueError(method)

        ids = [market_id, *[markets[int(j)] for j in neighbor_idx if markets[int(j)] != market_id]]
        sub_corr = corr.loc[ids, ids].to_numpy(dtype=float)
        offdiag = np.abs(offdiag_values(sub_corr))
        nbh_values = usable[ids]
        row: dict[str, object] = {
            "market_id": market_id,
            "neighborhood_method": method,
            "k": k,
            "source": source,
            "used_test_timestamps": used_test,
            "nbh_corr_mean_abs": float(np.nanmean(offdiag)) if offdiag.size else 0.0,
            "nbh_corr_std_abs": float(np.nanstd(offdiag)) if offdiag.size else 0.0,
            "nbh_prob_mean": float(nbh_values.mean(axis=1).mean()),
            "nbh_prob_std": float(nbh_values.std(axis=1, ddof=0).mean()),
        }
        for threshold in GRAPH_THRESHOLDS:
            row.update({f"nbh_t{int(threshold * 10)}_{key}": value for key, value in graph_features_for_corr(sub_corr, threshold).items()})
        rows.append(row)

    diagnostics = {
        "used_test_timestamps": used_test,
        "n_usable_markets": int(usable.shape[1]),
        "source_start": str(source_panel.index.min()),
        "source_end": str(source_panel.index.max()),
    }
    return pd.DataFrame(rows), diagnostics


def topology_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("nbh_t")]


def control_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in ["nbh_corr_mean_abs", "nbh_corr_std_abs", "nbh_prob_mean", "nbh_prob_std"] if c in frame.columns]
    cols += [c for c in frame.columns if c.endswith("_edge_density") or c.endswith("_clustering")]
    return cols


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    train = train.dropna(subset=["Y_i"]).copy()
    test = test.dropna(subset=["Y_i"]).copy()
    if train.empty or test.empty:
        raise ValueError("empty train/test")
    if train["Y_i"].nunique() < 2:
        raise ValueError("one-class training labels")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[feature_cols].astype(float)))
    x_test = scaler.transform(imputer.transform(test[feature_cols].astype(float)))
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("non-finite model features")
    model = LogisticRegression(C=0.01, solver="liblinear", max_iter=1000, random_state=RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(x_train, train["Y_i"].astype(int))
        pred = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("non-finite predictions")
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


def model_prediction_frame(test: pd.DataFrame, p_hat: np.ndarray, spec: ModelSpec, fold: Fold) -> pd.DataFrame:
    cols = [
        "timestamp",
        "market_id",
        "Y_i",
        "p_i_t",
        "active_market_count_t",
        "broad_domain",
        "broad_family",
    ]
    out = test[cols].dropna(subset=["Y_i"]).copy()
    out["fold"] = fold.fold
    out["model"] = spec.model
    out["feature_set"] = spec.feature_set
    out["category"] = spec.category
    out["is_confirmatory"] = spec.is_confirmatory
    out["is_valid_placebo"] = spec.is_valid_placebo
    out["is_leakage_control"] = spec.is_leakage_control
    out["p_hat"] = probability_clip(p_hat)
    return out


def result_rows_from_predictions(pred: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    slices = [("overall", "overall", pred)]
    for col in ["broad_domain", "broad_family", "Y_i"]:
        for value, group in pred.groupby(col, dropna=False):
            slices.append((col, str(value), group))
    pred = pred.copy()
    pred["active_count_bucket"] = pd.cut(
        pred["active_market_count_t"],
        bins=[-np.inf, 10, 25, 50, 100, np.inf],
        labels=["<=10", "11-25", "26-50", "51-100", ">100"],
    )
    pred["prob_bucket"] = pd.cut(
        pred["p_i_t"],
        bins=[-0.001, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.001],
        labels=["0-5%", "5-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-95%", "95-100%"],
    )
    for col in ["active_count_bucket", "prob_bucket"]:
        for value, group in pred.groupby(col, observed=False, dropna=True):
            slices.append((col, str(value), group))

    for slice_type, slice_value, group in slices:
        if group.empty:
            continue
        metrics = evaluate(group["Y_i"], group["p_hat"].to_numpy())
        rows.append(
            {
                "fold": int(group["fold"].iloc[0]),
                "model": str(group["model"].iloc[0]),
                "feature_set": str(group["feature_set"].iloc[0]),
                "category": str(group["category"].iloc[0]),
                "is_confirmatory": bool(group["is_confirmatory"].iloc[0]),
                "is_valid_placebo": bool(group["is_valid_placebo"].iloc[0]),
                "is_leakage_control": bool(group["is_leakage_control"].iloc[0]),
                "slice_type": slice_type,
                "slice_value": slice_value,
                **metrics,
            }
        )
    return rows


def shuffle_across_markets_within_timestamp(frame: pd.DataFrame, feature_cols: list[str], fold: Fold) -> pd.DataFrame:
    out = frame.copy()
    rng = np.random.default_rng(RANDOM_SEED + 101 * fold.fold)
    for _, idx in out.groupby("timestamp").groups.items():
        idx = list(idx)
        if len(idx) <= 1:
            continue
        shuffled = out.loc[idx, feature_cols].to_numpy()
        rng.shuffle(shuffled, axis=0)
        out.loc[idx, feature_cols] = shuffled
    return out


def bootstrap_delta(predictions: pd.DataFrame, candidate: str, baseline: str, *, cluster: str | None) -> dict[str, object]:
    wide = predictions[predictions["model"].isin([candidate, baseline])].pivot_table(
        index=["fold", "timestamp", "market_id", "Y_i", "broad_domain", "broad_family"],
        columns="model",
        values="p_hat",
        aggfunc="first",
    )
    wide = wide.dropna(subset=[candidate, baseline]).reset_index()
    if wide.empty:
        return {"candidate": candidate, "baseline": baseline, "cluster": cluster or "row", "mean_log_loss_gain": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_units": 0}
    y = wide["Y_i"].astype(int).to_numpy()
    cand_loss = -(y * np.log(probability_clip(wide[candidate])) + (1 - y) * np.log(1 - probability_clip(wide[candidate])))
    base_loss = -(y * np.log(probability_clip(wide[baseline])) + (1 - y) * np.log(1 - probability_clip(wide[baseline])))
    diff = base_loss - cand_loss
    rng = np.random.default_rng(RANDOM_SEED)
    samples = []
    if cluster is None:
        n = len(diff)
        for _ in range(BOOTSTRAPS):
            idx = rng.integers(0, n, size=n)
            samples.append(float(np.mean(diff[idx])))
        n_units = n
    else:
        codes, units = pd.factorize(wide[cluster], sort=False)
        grouped = [np.flatnonzero(codes == code) for code in range(len(units))]
        for _ in range(BOOTSTRAPS):
            draw = rng.integers(0, len(grouped), size=len(grouped))
            idx = np.concatenate([grouped[int(unit)] for unit in draw])
            samples.append(float(np.mean(diff[idx])))
        n_units = len(units)
    arr = np.asarray(samples)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "cluster": cluster or "row",
        "mean_log_loss_gain": float(np.mean(diff)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "n_units": int(n_units),
    }


def run_fold(
    supervised: pd.DataFrame,
    panel: pd.DataFrame,
    family_state: pd.DataFrame,
    fold: Fold,
) -> tuple[list[pd.DataFrame], list[dict[str, object]], list[dict[str, object]]]:
    train = fold_rows(supervised, fold, "train")
    test = fold_rows(supervised, fold, "test")
    bundle = fit_pca_bundle(family_state, fold)
    train = train.merge(bundle.pca_features, on="timestamp", how="left")
    test = test.merge(bundle.pca_features, on="timestamp", how="left")

    base_cols = ["p_i_t", "active_market_count_t"]
    pca_cols = [*base_cols, *bundle.pca_cols]
    pred_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    def run_model(spec: ModelSpec, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        try:
            pred = test_df["p_i_t"].to_numpy(dtype=float) if spec.model == "market_probability" else fit_predict(train_df, test_df, spec.feature_cols)
            pred_frames.append(model_prediction_frame(test_df, pred, spec, fold))
        except ValueError as exc:
            skipped.append({"fold": fold.fold, "model": spec.model, "reason": str(exc)})
            logging.info("Skipping fold=%s model=%s: %s", fold.fold, spec.model, exc)

    run_model(ModelSpec("market_probability", "market_probability", ["p_i_t"], "baseline"), train, test)
    run_model(ModelSpec("pca_only", "pca", pca_cols, "baseline"), train, test)

    train_nbh, diag = build_neighborhood_features(panel, fold, method=LOCKED_METHOD, k=LOCKED_K, source="train")
    diagnostics.append({"fold": fold.fold, "diagnostic": "locked_neighborhood_source", **diag})
    topo_cols = topology_columns(train_nbh)
    ctrl_cols = control_columns(train_nbh)
    train_locked = train.merge(train_nbh[["market_id", *topo_cols, *ctrl_cols]], on="market_id", how="left")
    test_locked = test.merge(train_nbh[["market_id", *topo_cols, *ctrl_cols]], on="market_id", how="left")

    run_model(ModelSpec("local_topology_locked", f"{LOCKED_METHOD}_k{LOCKED_K}", [*pca_cols, *topo_cols], "confirmatory", True), train_locked, test_locked)
    run_model(ModelSpec("controls_only", f"{LOCKED_METHOD}_k{LOCKED_K}_controls", [*pca_cols, *ctrl_cols], "ablation"), train_locked, test_locked)
    run_model(ModelSpec("topology_plus_controls", f"{LOCKED_METHOD}_k{LOCKED_K}_topology_controls", [*pca_cols, *topo_cols, *ctrl_cols], "ablation"), train_locked, test_locked)
    edge_cols = [c for c in topo_cols if c.endswith("_edge_density")]
    cluster_cols = [c for c in topo_cols if c.endswith("_clustering")]
    run_model(ModelSpec("edge_density_only", f"{LOCKED_METHOD}_k{LOCKED_K}_edge_density", [*pca_cols, *edge_cols], "ablation"), train_locked, test_locked)
    run_model(ModelSpec("clustering_only", f"{LOCKED_METHOD}_k{LOCKED_K}_clustering", [*pca_cols, *cluster_cols], "ablation"), train_locked, test_locked)

    train_shuffle = shuffle_across_markets_within_timestamp(train_locked, topo_cols, fold)
    test_shuffle = shuffle_across_markets_within_timestamp(test_locked, topo_cols, fold)
    run_model(ModelSpec("placebo_shuffle_market_within_timestamp", f"{LOCKED_METHOD}_k{LOCKED_K}", [*pca_cols, *topo_cols], "placebo", is_valid_placebo=True), train_shuffle, test_shuffle)

    random_nbh, random_diag = build_neighborhood_features(panel, fold, method="random", k=LOCKED_K, source="train")
    diagnostics.append({"fold": fold.fold, "diagnostic": "random_neighborhood_source", **random_diag})
    random_topo = topology_columns(random_nbh)
    train_random = train.merge(random_nbh[["market_id", *random_topo]], on="market_id", how="left")
    test_random = test.merge(random_nbh[["market_id", *random_topo]], on="market_id", how="left")
    run_model(ModelSpec("placebo_random_neighborhood", f"random_k{LOCKED_K}", [*pca_cols, *random_topo], "placebo", is_valid_placebo=True), train_random, test_random)

    leak_nbh, leak_diag = build_neighborhood_features(panel, fold, method=LOCKED_METHOD, k=LOCKED_K, source="test")
    diagnostics.append({"fold": fold.fold, "diagnostic": "invalid_test_period_neighborhood_source", **leak_diag})
    leak_topo = topology_columns(leak_nbh)
    train_leak = train.merge(leak_nbh[["market_id", *leak_topo]], on="market_id", how="left")
    test_leak = test.merge(leak_nbh[["market_id", *leak_topo]], on="market_id", how="left")
    run_model(ModelSpec("invalid_test_period_topology", f"{LOCKED_METHOD}_k{LOCKED_K}_test_corr", [*pca_cols, *leak_topo], "leakage_control", is_leakage_control=True), train_leak, test_leak)

    skipped.append({"fold": fold.fold, "model": "placebo_shuffle_timestamp_within_market", "reason": "not_applicable_fold_static_market_features"})
    skipped.append({"fold": fold.fold, "model": "placebo_future_shift", "reason": "not_applicable_fold_static_market_features"})

    for k in SENSITIVITY_K:
        sens_nbh, sens_diag = build_neighborhood_features(panel, fold, method=LOCKED_METHOD, k=k, source="train")
        diagnostics.append({"fold": fold.fold, "diagnostic": f"sensitivity_k{k}_source", **sens_diag})
        sens_topo = topology_columns(sens_nbh)
        train_sens = train.merge(sens_nbh[["market_id", *sens_topo]], on="market_id", how="left")
        test_sens = test.merge(sens_nbh[["market_id", *sens_topo]], on="market_id", how="left")
        run_model(ModelSpec(f"sensitivity_top_corr_k{k}", f"top_corr_k{k}", [*pca_cols, *sens_topo], "sensitivity"), train_sens, test_sens)

    return pred_frames, diagnostics, skipped


def aggregate_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (_, _), group in predictions.groupby(["fold", "model"], sort=False):
        rows.extend(result_rows_from_predictions(group))
    return pd.DataFrame(rows)


def summarize_overall(results: pd.DataFrame) -> pd.DataFrame:
    overall = results[results["slice_type"].eq("overall")].copy()
    return (
        overall.groupby(["model", "feature_set", "category", "is_confirmatory", "is_valid_placebo", "is_leakage_control"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            n_obs=("n_obs", "sum"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
            auc=("auc", "mean"),
            avg_pred=("avg_pred", "mean"),
            avg_actual=("avg_actual", "mean"),
        )
        .sort_values("log_loss")
    )


def write_summary(results: pd.DataFrame, predictions: pd.DataFrame, diagnostics: pd.DataFrame, skipped: pd.DataFrame, output_dir: Path) -> str:
    summary = summarize_overall(results)
    locked_fold = int(results["fold"].max()) if not results.empty else -1

    def metric(model: str, fold: int | None = None) -> pd.Series:
        frame = results[results["slice_type"].eq("overall")]
        frame = frame[frame["model"].eq(model)]
        if fold is not None:
            frame = frame[frame["fold"].eq(fold)]
        if frame.empty:
            return pd.Series(dtype=float)
        if fold is None:
            return frame.agg({"log_loss": "mean", "brier": "mean", "n_obs": "sum"})
        return frame.iloc[0]

    pca = metric("pca_only")
    local = metric("local_topology_locked")
    pca_holdout = metric("pca_only", locked_fold)
    local_holdout = metric("local_topology_locked", locked_fold)
    log_loss_gain = float(pca["log_loss"] - local["log_loss"])
    brier_gain = float(pca["brier"] - local["brier"])
    holdout_log_loss_gain = float(pca_holdout["log_loss"] - local_holdout["log_loss"])
    holdout_brier_gain = float(pca_holdout["brier"] - local_holdout["brier"])

    fold_wide = results[results["slice_type"].eq("overall") & results["model"].isin(["pca_only", "local_topology_locked"])].pivot(index="fold", columns="model", values=["log_loss", "brier"])
    log_folds = int((fold_wide[("log_loss", "pca_only")] > fold_wide[("log_loss", "local_topology_locked")]).sum())
    brier_folds = int((fold_wide[("brier", "pca_only")] > fold_wide[("brier", "local_topology_locked")]).sum())

    valid_placebos = summary[summary["is_valid_placebo"]]
    beats_placebos = bool(valid_placebos.empty or local["log_loss"] < valid_placebos["log_loss"].min())
    leakage_control = summary[summary["is_leakage_control"]]
    leakage_log_loss = float(leakage_control["log_loss"].iloc[0]) if not leakage_control.empty else np.nan
    leakage_outperforms_real = bool(not leakage_control.empty and leakage_log_loss < float(local["log_loss"]))
    controls = summary[summary["model"].isin(["controls_only", "topology_plus_controls"])]
    controls_only = summary[summary["model"].eq("controls_only")]
    beats_controls = bool(controls_only.empty or local["log_loss"] < controls_only["log_loss"].iloc[0])
    topo_plus_controls = summary[summary["model"].eq("topology_plus_controls")]
    topology_survives_controls = bool(not topo_plus_controls.empty and topo_plus_controls["log_loss"].iloc[0] < controls_only["log_loss"].iloc[0])
    leakage_rows = diagnostics[diagnostics["diagnostic"].eq("invalid_test_period_neighborhood_source")] if not diagnostics.empty else pd.DataFrame()
    leakage_control_used_test = bool(not leakage_rows.empty and leakage_rows["used_test_timestamps"].astype(bool).any())

    ci_rows = [
        bootstrap_delta(predictions, "local_topology_locked", "pca_only", cluster=None),
        bootstrap_delta(predictions, "local_topology_locked", "pca_only", cluster="market_id"),
        bootstrap_delta(predictions, "local_topology_locked", "pca_only", cluster="timestamp"),
    ]
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(output_dir / "local_topology_validation_bootstrap.csv", index=False)

    decision = "abandoned"
    if holdout_log_loss_gain >= MIN_LOG_LOSS_GAIN and holdout_brier_gain >= MIN_BRIER_GAIN and beats_placebos and topology_survives_controls:
        decision = "main topology-enhanced model"
    elif holdout_log_loss_gain >= MIN_LOG_LOSS_GAIN and holdout_brier_gain >= MIN_BRIER_GAIN:
        decision = "local-neighborhood supplemental signal"
    elif log_loss_gain > 0 and brier_gain > 0:
        decision = "diagnostic only"

    fragile = log_folds < 10
    domain_results = results[(results["slice_type"].eq("broad_domain")) & (results["model"].isin(["pca_only", "local_topology_locked"]))]
    if not domain_results.empty:
        domain_wide = domain_results.pivot_table(index="slice_value", columns="model", values="log_loss", aggfunc="mean").dropna()
        if not domain_wide.empty:
            domain_wide["gain"] = domain_wide["pca_only"] - domain_wide["local_topology_locked"]
            dominant_domain = str(domain_wide["gain"].idxmax())
            dominant_domain_gain = float(domain_wide["gain"].max())
        else:
            dominant_domain = "none"
            dominant_domain_gain = np.nan
    else:
        dominant_domain = "none"
        dominant_domain_gain = np.nan

    lines = [
        "LOCAL TOPOLOGY VALIDATION SUMMARY",
        "",
        "Locked candidate:",
        f"- method: {LOCKED_METHOD}",
        f"- k: {LOCKED_K}",
        f"- locked holdout fold: {locked_fold}",
        f"- graph thresholds: {', '.join(str(x) for x in GRAPH_THRESHOLDS)}",
        "",
        "Overall performance:",
        f"- PCA-only log loss: {pca['log_loss']:.6f}",
        f"- local topology log loss: {local['log_loss']:.6f}",
        f"- log loss gain vs PCA: {log_loss_gain:+.6f}",
        f"- PCA-only Brier: {pca['brier']:.6f}",
        f"- local topology Brier: {local['brier']:.6f}",
        f"- Brier gain vs PCA: {brier_gain:+.6f}",
        f"- folds improved: log loss {log_folds}/17, Brier {brier_folds}/17",
        "",
        "Locked holdout:",
        f"- log loss gain vs PCA: {holdout_log_loss_gain:+.6f}",
        f"- Brier gain vs PCA: {holdout_brier_gain:+.6f}",
        f"- clears meaningful thresholds: {'yes' if holdout_log_loss_gain >= MIN_LOG_LOSS_GAIN and holdout_brier_gain >= MIN_BRIER_GAIN else 'no'}",
        "",
        "Placebo and leakage checks:",
        f"- beats all valid placebos: {'yes' if beats_placebos else 'no'}",
        f"- invalid test-period topology explicitly used test timestamps: {'yes' if leakage_control_used_test else 'no'}",
        f"- invalid test-period topology log loss: {leakage_log_loss:.6f}",
        f"- invalid test-period topology outperforms valid locked topology: {'yes' if leakage_outperforms_real else 'no'}",
        "- timestamp-within-market and future-shift placebos: not applicable because locked topology features are fold-static market features",
        "",
        "Ablation and interpretation:",
        f"- beats simple neighborhood controls: {'yes' if beats_controls else 'no'}",
        f"- topology still helps after controls: {'yes' if topology_survives_controls else 'no'}",
        f"- strongest domain gain: {dominant_domain} ({dominant_domain_gain:+.6f} log-loss gain)",
        f"- fragility flag: {'yes' if fragile else 'no'}",
        "",
        "Bootstrap log-loss gain vs PCA:",
    ]
    for _, row in ci.iterrows():
        lines.append(f"- {row['cluster']} bootstrap: mean {row['mean_log_loss_gain']:+.6f}, 95% CI [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]")
    lines.extend(
        [
            "",
            "Answers:",
            f"- Does local topology beat PCA-only on log loss? {'yes' if log_loss_gain > 0 else 'no'}",
            f"- Does local topology beat PCA-only on Brier? {'yes' if brier_gain > 0 else 'no'}",
            f"- Is the locked-holdout improvement meaningful? {'yes' if holdout_log_loss_gain >= MIN_LOG_LOSS_GAIN and holdout_brier_gain >= MIN_BRIER_GAIN else 'no'}",
            f"- Does it beat all valid placebos? {'yes' if beats_placebos else 'no'}",
            f"- Is there evidence of leakage in the valid candidate? {'no' if diagnostics[diagnostics['diagnostic'].eq('locked_neighborhood_source')]['used_test_timestamps'].astype(bool).sum() == 0 else 'yes'}",
            "",
            "Paper framing decision:",
            f"- {decision}",
        ]
    )
    text = "\n".join(lines) + "\n"
    (output_dir / "local_topology_validation_summary.md").write_text(text, encoding="utf-8")
    return text


def run_all(candidate_markets_path: Path, prices_path: Path, panel_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    ensure_dirs([output_dir])
    markets, panel, _raw, mask = load_inputs(candidate_markets_path, prices_path, panel_path)
    markets = markets[markets["market_id"].isin(panel.columns.astype(str))].copy()
    supervised = clean_supervised(load_or_build_supervised(output_dir, panel, mask, markets))
    family_state = build_family_state(panel, mask, markets)
    folds = make_chronological_folds(panel)

    pred_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for fold in folds:
        logging.info("Running local topology validation fold=%s/%s", fold.fold, len(folds))
        fold_preds, fold_diag, fold_skipped = run_fold(supervised, panel, family_state, fold)
        pred_frames.extend(fold_preds)
        diagnostics.extend(fold_diag)
        skipped.extend(fold_skipped)

    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    results = aggregate_results(predictions)
    diagnostics_df = pd.DataFrame(diagnostics)
    skipped_df = pd.DataFrame(skipped)

    results.to_csv(output_dir / "local_topology_validation_results.csv", index=False)
    predictions.to_parquet(output_dir / "local_topology_validation_predictions.parquet", index=False)
    placebo = results[results["category"].isin(["placebo", "leakage_control"])].copy()
    if not skipped_df.empty:
        skipped_df.to_csv(output_dir / "local_topology_validation_skipped_placebos.csv", index=False)
    placebo.to_csv(output_dir / "local_topology_validation_placebos.csv", index=False)
    results[results["category"].eq("ablation")].to_csv(output_dir / "local_topology_validation_ablation.csv", index=False)
    diagnostics_df.to_csv(output_dir / "local_topology_validation_diagnostics.csv", index=False)
    summary = write_summary(results, predictions, diagnostics_df, skipped_df, output_dir)
    return results, predictions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run locked local-topology validation.")
    parser.add_argument("--candidate-markets", default="data/processed/candidate_universe_markets.parquet")
    parser.add_argument("--prices", default="data/processed/prices_long.parquet")
    parser.add_argument("--panel", default="data/processed/universe_b_macro_crypto_panel.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    root = project_root()
    _results, _predictions, summary = run_all(
        resolve_path(root, args.candidate_markets),
        resolve_path(root, args.prices),
        resolve_path(root, args.panel),
        resolve_path(root, args.output_dir),
    )
    print(summary)


if __name__ == "__main__":
    main()
