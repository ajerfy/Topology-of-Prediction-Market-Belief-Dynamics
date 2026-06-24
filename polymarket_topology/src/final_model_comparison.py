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

from active_set_forecasting import build_family_state, fold_rows, load_inputs, resolve_path
from local_topology_validation import (
    BOOTSTRAPS,
    LOCKED_K,
    LOCKED_METHOD,
    MIN_BRIER_GAIN,
    MIN_LOG_LOSS_GAIN,
    build_neighborhood_features,
    control_columns,
    topology_columns,
)
from pca_plus_topology_forecasting import load_or_build_supervised
from supervised_forecasting import make_chronological_folds, probability_clip
from topology_rescue_sweep import GRAPH_THRESHOLDS, RANDOM_SEED, clean_supervised, fit_pca_bundle
from utils import ensure_dirs, project_root, setup_logging


try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - exercised only when dependency is absent.
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


@dataclass(frozen=True)
class ModelSpec:
    model: str
    model_type: str
    feature_group: str
    category: str


MODEL_SPECS = (
    ModelSpec("market_probability", "market", "market", "baseline"),
    ModelSpec("logit_pca", "logit", "pca", "logistic"),
    ModelSpec("logit_pca_controls", "logit", "pca_controls", "logistic"),
    ModelSpec("logit_pca_topology", "logit", "pca_topology", "logistic"),
    ModelSpec("logit_pca_controls_topology", "logit", "pca_controls_topology", "logistic"),
    ModelSpec("xgb_pca", "xgb", "pca", "xgboost"),
    ModelSpec("xgb_pca_controls", "xgb", "pca_controls", "xgboost"),
    ModelSpec("xgb_pca_topology", "xgb", "pca_topology", "xgboost"),
    ModelSpec("xgb_pca_controls_topology", "xgb", "pca_controls_topology", "xgboost"),
)


COMPARISONS = (
    ("logit_pca_controls_topology", "logit_pca_controls", "logit topology after controls"),
    ("xgb_pca_controls_topology", "xgb_pca_controls", "xgb topology after controls"),
    ("logit_pca_controls_topology", "logit_pca", "logit controls+topology after PCA"),
    ("xgb_pca_controls_topology", "xgb_pca", "xgb controls+topology after PCA"),
)


def require_xgboost() -> None:
    if XGBClassifier is None:
        raise ImportError(f"XGBoost is required for the final comparison. Install it with: pip install xgboost. Original error: {XGBOOST_IMPORT_ERROR!r}")


def feature_map(pca_cols: list[str], topo_cols: list[str], ctrl_cols: list[str]) -> dict[str, list[str]]:
    base = ["p_i_t", "active_market_count_t"]
    pca = [*base, *pca_cols]
    return {
        "market": ["p_i_t"],
        "pca": pca,
        "pca_controls": [*pca, *ctrl_cols],
        "pca_topology": [*pca, *topo_cols],
        "pca_controls_topology": [*pca, *ctrl_cols, *topo_cols],
    }


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], model_type: str) -> np.ndarray:
    if model_type == "market":
        return probability_clip(test["p_i_t"].to_numpy(dtype=float))
    train = train.dropna(subset=["Y_i"]).copy()
    test = test.dropna(subset=["Y_i"]).copy()
    if train.empty or test.empty:
        raise ValueError("empty train/test")
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
        require_xgboost()
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
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("non-finite features")
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


def prediction_frame(test: pd.DataFrame, pred: np.ndarray, spec: ModelSpec, fold: int) -> pd.DataFrame:
    cols = ["timestamp", "market_id", "Y_i", "p_i_t", "active_market_count_t", "broad_domain", "broad_family"]
    out = test[cols].dropna(subset=["Y_i"]).copy()
    out["fold"] = fold
    out["model"] = spec.model
    out["model_type"] = spec.model_type
    out["feature_group"] = spec.feature_group
    out["category"] = spec.category
    out["p_hat"] = probability_clip(pred)
    return out


def add_prediction_buckets(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["active_count_bucket"] = pd.cut(
        out["active_market_count_t"],
        bins=[-np.inf, 10, 25, 50, 100, np.inf],
        labels=["<=10", "11-25", "26-50", "51-100", ">100"],
    )
    out["prob_bucket"] = pd.cut(
        out["p_i_t"],
        bins=[-0.001, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.001],
        labels=["0-5%", "5-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-95%", "95-100%"],
    )
    return out


def result_rows_for_group(pred: pd.DataFrame) -> list[dict[str, object]]:
    pred = add_prediction_buckets(pred)
    slices = [("overall", "overall", pred)]
    for col in ["broad_domain", "broad_family", "Y_i", "active_count_bucket", "prob_bucket"]:
        for value, group in pred.groupby(col, observed=False, dropna=True):
            slices.append((col, str(value), group))
    rows = []
    for slice_type, slice_value, group in slices:
        if group.empty:
            continue
        rows.append(
            {
                "fold": int(group["fold"].iloc[0]),
                "model": str(group["model"].iloc[0]),
                "model_type": str(group["model_type"].iloc[0]),
                "feature_group": str(group["feature_group"].iloc[0]),
                "category": str(group["category"].iloc[0]),
                "slice_type": slice_type,
                "slice_value": slice_value,
                **evaluate(group["Y_i"], group["p_hat"].to_numpy()),
            }
        )
    return rows


def aggregate_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, _), group in predictions.groupby(["fold", "model"], sort=False):
        rows.extend(result_rows_for_group(group))
    return pd.DataFrame(rows)


def calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    df["prob_decile"] = pd.cut(df["p_hat"], bins=np.linspace(0, 1, 11), include_lowest=True, labels=False, duplicates="drop")
    rows = []
    for (model, fold, decile), group in df.groupby(["model", "fold", "prob_decile"], dropna=True):
        if group.empty:
            continue
        rows.append(
            {
                "model": model,
                "fold": int(fold),
                "prob_decile": int(decile),
                "n_obs": int(len(group)),
                "avg_pred": float(group["p_hat"].mean()),
                "avg_actual": float(group["Y_i"].mean()),
                "abs_calibration_error": float(abs(group["p_hat"].mean() - group["Y_i"].mean())),
            }
        )
    cal = pd.DataFrame(rows)
    if cal.empty:
        return cal
    ece = cal.groupby(["model", "fold"]).apply(lambda g: np.average(g["abs_calibration_error"], weights=g["n_obs"]), include_groups=False).rename("ece").reset_index()
    return cal.merge(ece, on=["model", "fold"], how="left")


def loss_arrays(wide: pd.DataFrame, candidate: str, baseline: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = wide["Y_i"].astype(int).to_numpy()
    cand = probability_clip(wide[candidate])
    base = probability_clip(wide[baseline])
    cand_log = -(y * np.log(cand) + (1 - y) * np.log(1 - cand))
    base_log = -(y * np.log(base) + (1 - y) * np.log(1 - base))
    cand_brier = (cand - y) ** 2
    base_brier = (base - y) ** 2
    return base_log - cand_log, base_brier - cand_brier, y


def bootstrap_comparison(predictions: pd.DataFrame, candidate: str, baseline: str, label: str, cluster: str | None) -> dict[str, object]:
    index_cols = ["fold", "timestamp", "market_id", "Y_i", "broad_domain", "broad_family"]
    wide = predictions[predictions["model"].isin([candidate, baseline])].pivot_table(index=index_cols, columns="model", values="p_hat", aggfunc="first")
    wide = wide.dropna(subset=[candidate, baseline]).reset_index()
    log_gain, brier_gain, _y = loss_arrays(wide, candidate, baseline)
    rng = np.random.default_rng(RANDOM_SEED)

    def draw_metric(diff: np.ndarray) -> tuple[float, float, float]:
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
        return float(np.mean(diff)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), int(n_units)

    log_mean, log_low, log_high, n_units = draw_metric(log_gain)
    brier_mean, brier_low, brier_high, _ = draw_metric(brier_gain)
    return {
        "comparison": label,
        "candidate": candidate,
        "baseline": baseline,
        "cluster": cluster or "row",
        "n_units": n_units,
        "log_loss_gain": log_mean,
        "log_loss_ci_low": log_low,
        "log_loss_ci_high": log_high,
        "brier_gain": brier_mean,
        "brier_ci_low": brier_low,
        "brier_ci_high": brier_high,
    }


def stat_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    overall = predictions.groupby(["fold", "model"]).apply(lambda g: pd.Series(evaluate(g["Y_i"], g["p_hat"].to_numpy())), include_groups=False).reset_index()
    for candidate, baseline, label in COMPARISONS:
        for cluster in (None, "market_id", "timestamp"):
            rows.append(bootstrap_comparison(predictions, candidate, baseline, label, cluster))
        cand = overall[overall["model"].eq(candidate)].set_index("fold")
        base = overall[overall["model"].eq(baseline)].set_index("fold")
        joined = cand[["log_loss", "brier"]].join(base[["log_loss", "brier"]], lsuffix="_candidate", rsuffix="_baseline")
        rows.append(
            {
                "comparison": label,
                "candidate": candidate,
                "baseline": baseline,
                "cluster": "fold_count",
                "n_units": int(len(joined)),
                "log_loss_gain": float((joined["log_loss_baseline"] - joined["log_loss_candidate"]).mean()),
                "log_loss_ci_low": np.nan,
                "log_loss_ci_high": np.nan,
                "brier_gain": float((joined["brier_baseline"] - joined["brier_candidate"]).mean()),
                "brier_ci_low": np.nan,
                "brier_ci_high": np.nan,
                "log_loss_folds_improved": int((joined["log_loss_candidate"] < joined["log_loss_baseline"]).sum()),
                "brier_folds_improved": int((joined["brier_candidate"] < joined["brier_baseline"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def run_fold(supervised: pd.DataFrame, panel: pd.DataFrame, family_state: pd.DataFrame, fold) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    train = fold_rows(supervised, fold, "train")
    test = fold_rows(supervised, fold, "test")
    bundle = fit_pca_bundle(family_state, fold)
    train = train.merge(bundle.pca_features, on="timestamp", how="left")
    test = test.merge(bundle.pca_features, on="timestamp", how="left")
    nbh, diag = build_neighborhood_features(panel, fold, method=LOCKED_METHOD, k=LOCKED_K, source="train")
    topo_cols = topology_columns(nbh)
    ctrl_cols = control_columns(nbh)
    train = train.merge(nbh[["market_id", *topo_cols, *ctrl_cols]], on="market_id", how="left")
    test = test.merge(nbh[["market_id", *topo_cols, *ctrl_cols]], on="market_id", how="left")
    fmap = feature_map(bundle.pca_cols, topo_cols, ctrl_cols)
    pred_frames = []
    for spec in MODEL_SPECS:
        pred = fit_predict(train, test, fmap[spec.feature_group], spec.model_type)
        pred_frames.append(prediction_frame(test, pred, spec, fold.fold))
    diagnostics = [{"fold": fold.fold, "diagnostic": "locked_neighborhood_source", **diag}]
    return pred_frames, diagnostics


def summarize_overall(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results[results["slice_type"].eq("overall")]
        .groupby(["model", "model_type", "feature_group", "category"], as_index=False)
        .agg(folds=("fold", "nunique"), n_obs=("n_obs", "sum"), log_loss=("log_loss", "mean"), brier=("brier", "mean"), auc=("auc", "mean"), avg_pred=("avg_pred", "mean"), avg_actual=("avg_actual", "mean"))
        .sort_values("log_loss")
    )


def ensure_figure_dir(output_dir: Path) -> Path:
    fig_dir = output_dir / "figures" / "final_comparison"
    ensure_dirs([fig_dir])
    return fig_dir


def save_bar(df: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    plot_df = df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(plot_df["model"], plot_df[metric], color="#4C78A8")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def create_figures(results: pd.DataFrame, predictions: pd.DataFrame, calibration: pd.DataFrame, stat_df: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = ensure_figure_dir(output_dir)
    overall = summarize_overall(results)
    save_bar(overall, "log_loss", "Final Model Ranking by Log Loss", fig_dir / "log_loss_ranking.png")
    save_bar(overall, "brier", "Final Model Ranking by Brier Score", fig_dir / "brier_ranking.png")

    fold = results[results["slice_type"].eq("overall")].copy()
    pca = fold[fold["model"].eq("logit_pca")][["fold", "log_loss", "brier"]].rename(columns={"log_loss": "pca_log_loss", "brier": "pca_brier"})
    deltas = fold.merge(pca, on="fold", how="left")
    deltas["delta_log_loss_vs_pca"] = deltas["log_loss"] - deltas["pca_log_loss"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in ["logit_pca_controls_topology", "xgb_pca_controls_topology", "logit_pca_controls", "xgb_pca_controls"]:
        group = deltas[deltas["model"].eq(model)]
        ax.plot(group["fold"], group["delta_log_loss_vs_pca"], marker="o", label=model)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Log-Loss Delta vs Logistic PCA")
    ax.set_title("Fold-Level Deltas")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fold_level_deltas.png", dpi=180)
    plt.close(fig)

    holdout = fold[fold["fold"].eq(fold["fold"].max())]
    save_bar(holdout.sort_values("log_loss"), "log_loss", "Locked Holdout Log Loss", fig_dir / "locked_holdout_log_loss.png")

    cal_avg = calibration.groupby(["model", "prob_decile"], as_index=False).agg(avg_pred=("avg_pred", "mean"), avg_actual=("avg_actual", "mean"))
    fig, ax = plt.subplots(figsize=(6, 6))
    for model in ["market_probability", "logit_pca", "logit_pca_controls_topology", "xgb_pca_controls", "xgb_pca_controls_topology"]:
        group = cal_avg[cal_avg["model"].eq(model)]
        ax.plot(group["avg_pred"], group["avg_actual"], marker="o", label=model)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--")
    ax.set_xlabel("Average Predicted Probability")
    ax.set_ylabel("Average Actual Outcome")
    ax.set_title("Calibration Curves")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_curves.png", dpi=180)
    fig.savefig(fig_dir / "reliability_diagram.png", dpi=180)
    plt.close(fig)

    ci = stat_df[stat_df["cluster"].isin(["row", "market_id", "timestamp"])].copy()
    ci = ci[ci["comparison"].isin(["logit topology after controls", "xgb topology after controls"])]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ci["comparison"] + " / " + ci["cluster"]
    y = np.arange(len(ci))
    ax.errorbar(ci["log_loss_gain"], y, xerr=[ci["log_loss_gain"] - ci["log_loss_ci_low"], ci["log_loss_ci_high"] - ci["log_loss_gain"]], fmt="o")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Log-Loss Gain vs Baseline")
    ax.set_title("Bootstrap Confidence Intervals")
    fig.tight_layout()
    fig.savefig(fig_dir / "bootstrap_ci.png", dpi=180)
    plt.close(fig)

    domain = results[(results["slice_type"].eq("broad_domain")) & results["model"].isin(["logit_pca_controls", "logit_pca_controls_topology", "xgb_pca_controls", "xgb_pca_controls_topology"])]
    pivot = domain.pivot_table(index="slice_value", columns="model", values="log_loss", aggfunc="mean")
    pivot["logit_gain"] = pivot["logit_pca_controls"] - pivot["logit_pca_controls_topology"]
    pivot["xgb_gain"] = pivot["xgb_pca_controls"] - pivot["xgb_pca_controls_topology"]
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot[["logit_gain", "xgb_gain"]].plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Log-Loss Gain From Topology")
    ax.set_title("Domain-Level Topology Gains")
    fig.tight_layout()
    fig.savefig(fig_dir / "domain_level_delta.png", dpi=180)
    plt.close(fig)

    ablation_models = ["logit_pca", "logit_pca_controls", "logit_pca_topology", "logit_pca_controls_topology"]
    save_bar(overall[overall["model"].isin(ablation_models)], "log_loss", "Logistic Ablation", fig_dir / "ablation_chart.png")
    xgb_models = ["xgb_pca", "xgb_pca_controls", "xgb_pca_topology", "xgb_pca_controls_topology"]
    save_bar(overall[overall["model"].isin(xgb_models)], "log_loss", "XGBoost vs XGBoost + Topology", fig_dir / "xgboost_comparison.png")


def write_summary(results: pd.DataFrame, calibration: pd.DataFrame, stat_df: pd.DataFrame, diagnostics: pd.DataFrame, output_dir: Path) -> str:
    overall = summarize_overall(results)
    locked_fold = int(results["fold"].max())
    holdout = results[(results["slice_type"].eq("overall")) & (results["fold"].eq(locked_fold))]

    def row(model: str, frame: pd.DataFrame = overall) -> pd.Series:
        return frame[frame["model"].eq(model)].iloc[0]

    def gain(candidate: str, baseline: str, metric: str = "log_loss") -> float:
        return float(row(baseline)[metric] - row(candidate)[metric])

    logit_topology_gain = gain("logit_pca_controls_topology", "logit_pca_controls")
    logit_topology_brier_gain = gain("logit_pca_controls_topology", "logit_pca_controls", "brier")
    xgb_topology_gain = gain("xgb_pca_controls_topology", "xgb_pca_controls")
    xgb_topology_brier_gain = gain("xgb_pca_controls_topology", "xgb_pca_controls", "brier")
    xgb_vs_logit_gain = gain("xgb_pca_controls", "logit_pca_controls")
    best = overall.sort_values("log_loss").iloc[0]
    best_non_topology = overall[~overall["feature_group"].str.contains("topology")].sort_values("log_loss").iloc[0]
    best_topology = overall[overall["feature_group"].str.contains("topology")].sort_values("log_loss").iloc[0]
    valid_no_leak = int(diagnostics[diagnostics["diagnostic"].eq("locked_neighborhood_source")]["used_test_timestamps"].astype(bool).sum()) == 0

    ece = calibration.groupby("model", as_index=False).agg(ece=("ece", "mean"))
    ece_topology_gain = float(ece[ece["model"].eq("logit_pca_controls")]["ece"].iloc[0] - ece[ece["model"].eq("logit_pca_controls_topology")]["ece"].iloc[0])
    domain = results[(results["slice_type"].eq("broad_domain")) & results["model"].isin(["logit_pca_controls", "logit_pca_controls_topology"])]
    domain_pivot = domain.pivot_table(index="slice_value", columns="model", values="log_loss", aggfunc="mean").dropna()
    domain_pivot["gain"] = domain_pivot["logit_pca_controls"] - domain_pivot["logit_pca_controls_topology"]
    dominated_by_macro = bool("macro" in domain_pivot.index and domain_pivot.loc["macro", "gain"] > domain_pivot["gain"].sum() * 0.75)

    def stat_line(label: str) -> str:
        rows = stat_df[(stat_df["comparison"].eq(label)) & (stat_df["cluster"].eq("market_id"))]
        if rows.empty:
            return "not available"
        r = rows.iloc[0]
        return f"log-loss gain {r['log_loss_gain']:+.6f}, market-clustered 95% CI [{r['log_loss_ci_low']:+.6f}, {r['log_loss_ci_high']:+.6f}]"

    if logit_topology_gain > 0 and xgb_topology_gain > 0 and logit_topology_brier_gain > 0 and xgb_topology_brier_gain > 0:
        framing = "Topology is a robust incremental local-structure signal for both logistic and XGBoost models."
    elif logit_topology_gain > 0 and logit_topology_brier_gain > 0 and not (xgb_topology_gain > 0 and xgb_topology_brier_gain > 0):
        framing = "Topology helps linear models but is partly absorbed by nonlinear learners."
    elif best_non_topology["log_loss"] <= best_topology["log_loss"]:
        framing = "XGBoost or neighborhood controls dominate; topology should be reported as secondary."
    else:
        framing = "Topology is diagnostic/local-structure evidence only."

    lines = [
        "FINAL MODEL COMPARISON SUMMARY",
        "",
        "Dataset and validation:",
        f"- folds: {int(overall['folds'].max())}",
        f"- locked holdout fold: {locked_fold}",
        f"- topology construction used zero test timestamps: {'yes' if valid_no_leak else 'no'}",
        f"- locked topology: {LOCKED_METHOD}, k={LOCKED_K}, thresholds={', '.join(str(x) for x in GRAPH_THRESHOLDS)}",
        "",
        "Overall ranking:",
    ]
    for _, r in overall.iterrows():
        lines.append(f"- {r['model']}: log loss {r['log_loss']:.6f}, Brier {r['brier']:.6f}, AUC {r['auc']:.6f}")
    lines.extend(
        [
            "",
            "Key comparisons:",
            f"- Logistic topology after controls: log-loss gain {logit_topology_gain:+.6f}, Brier gain {logit_topology_brier_gain:+.6f}",
            f"- XGBoost topology after controls: log-loss gain {xgb_topology_gain:+.6f}, Brier gain {xgb_topology_brier_gain:+.6f}",
            f"- XGBoost controls vs logistic controls: log-loss gain {xgb_vs_logit_gain:+.6f}",
            f"- Best model: {best['model']}",
            f"- Best topology model: {best_topology['model']}",
            f"- Best non-topology model: {best_non_topology['model']}",
            "",
            "Statistical tests:",
            f"- Logistic topology after controls: {stat_line('logit topology after controls')}",
            f"- XGBoost topology after controls: {stat_line('xgb topology after controls')}",
            "",
            "Calibration:",
            f"- Logistic topology ECE gain vs controls: {ece_topology_gain:+.6f}",
            f"- Topology improves calibration: {'yes' if ece_topology_gain > 0 else 'no'}",
            "",
            "Robustness:",
            f"- Result dominated by macro markets: {'yes' if dominated_by_macro else 'no'}",
            f"- Locked holdout best model: {holdout.sort_values('log_loss').iloc[0]['model']}",
            "",
            "Answers:",
            f"- Is the logistic topology result paper-ready? {'yes' if logit_topology_gain > MIN_LOG_LOSS_GAIN and logit_topology_brier_gain > MIN_BRIER_GAIN and valid_no_leak else 'no'}",
            f"- Does topology improve over PCA + controls? {'yes' if logit_topology_gain > 0 and logit_topology_brier_gain > 0 else 'no'}",
            f"- Does XGBoost beat logistic regression? {'yes' if xgb_vs_logit_gain > 0 else 'no'}",
            f"- Does topology improve XGBoost? {'yes' if xgb_topology_gain > 0 and xgb_topology_brier_gain > 0 else 'no'}",
            f"- Does topology improve both log loss and Brier? {'yes' if logit_topology_gain > 0 and logit_topology_brier_gain > 0 else 'no'}",
            f"- Are gains statistically meaningful? {'yes' if stat_df[(stat_df['comparison'].eq('logit topology after controls')) & (stat_df['cluster'].eq('market_id'))]['log_loss_ci_low'].iloc[0] > 0 else 'mixed'}",
            "",
            "Final paper framing:",
            f"- {framing}",
        ]
    )
    text = "\n".join(lines) + "\n"
    (output_dir / "final_model_comparison_summary.md").write_text(text, encoding="utf-8")
    return text


def run_all(candidate_markets_path: Path, prices_path: Path, panel_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    require_xgboost()
    ensure_dirs([output_dir])
    markets, panel, _raw, mask = load_inputs(candidate_markets_path, prices_path, panel_path)
    markets = markets[markets["market_id"].isin(panel.columns.astype(str))].copy()
    supervised = clean_supervised(load_or_build_supervised(output_dir, panel, mask, markets))
    family_state = build_family_state(panel, mask, markets)
    folds = make_chronological_folds(panel)

    pred_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for fold in folds:
        logging.info("Running final model comparison fold=%s/%s", fold.fold, len(folds))
        fold_preds, fold_diag = run_fold(supervised, panel, family_state, fold)
        pred_frames.extend(fold_preds)
        diagnostics.extend(fold_diag)

    predictions = pd.concat(pred_frames, ignore_index=True)
    if not np.isfinite(predictions["p_hat"].to_numpy()).all():
        raise ValueError("Non-finite predictions generated.")
    results = aggregate_results(predictions)
    calibration = calibration_table(predictions)
    stat_df = stat_tests(predictions)
    diagnostics_df = pd.DataFrame(diagnostics)

    results.to_csv(output_dir / "final_model_comparison_results.csv", index=False)
    predictions.to_parquet(output_dir / "final_model_comparison_predictions.parquet", index=False)
    calibration.to_csv(output_dir / "final_model_comparison_calibration.csv", index=False)
    stat_df.to_csv(output_dir / "final_model_comparison_stat_tests.csv", index=False)
    diagnostics_df.to_csv(output_dir / "final_model_comparison_diagnostics.csv", index=False)
    create_figures(results, predictions, calibration, stat_df, output_dir)
    summary = write_summary(results, calibration, stat_df, diagnostics_df, output_dir)
    return results, predictions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final paper-ready model comparison.")
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
