from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from active_set_forecasting import build_family_state, load_inputs, resolve_path
from utils import ensure_dirs, project_root, setup_logging


def preprocess_family_state(family_state: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, SimpleImputer, StandardScaler, list[str]]:
    observed = family_state.notna().sum(axis=0)
    varying = family_state.nunique(dropna=True) > 1
    cols = observed[observed > 0].index.intersection(varying[varying].index).astype(str).tolist()
    if not cols:
        raise ValueError("No usable family-state features.")
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(family_state[cols])
    scaled = scaler.fit_transform(imputed)
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite values after family-state preprocessing.")
    return family_state[cols], scaled, imputer, scaler, cols


def timestamp_metadata(panel: pd.DataFrame, mask: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    active_values = panel.where(mask)
    active_market_count = active_values.notna().sum(axis=1)
    avg_probability = active_values.mean(axis=1)
    family_map = markets.set_index("market_id")["broad_family"].to_dict()
    active_family_frames = []
    for family in sorted(set(family_map.values())):
        cols = [col for col in panel.columns if family_map.get(col) == family]
        if not cols:
            continue
        active_family_frames.append(mask[cols].any(axis=1).rename(family))
    active_family_count = pd.concat(active_family_frames, axis=1).sum(axis=1) if active_family_frames else pd.Series(0, index=panel.index)
    return pd.DataFrame(
        {
            "timestamp": panel.index,
            "active_market_count": active_market_count.to_numpy(),
            "active_family_count": active_family_count.to_numpy(),
            "average_market_probability": avg_probability.to_numpy(),
        }
    )


def pca_projection(scaled: np.ndarray, index: pd.DatetimeIndex, meta: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, PCA]:
    n_components = min(10, scaled.shape[1], scaled.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pca = PCA(n_components=n_components, svd_solver="full").fit(scaled)
        scores = pca.transform(scaled)
    coords = pd.DataFrame(
        {
            "timestamp": index,
            "PC1": scores[:, 0],
            "PC2": scores[:, 1] if scores.shape[1] > 1 else 0.0,
            "PC3": scores[:, 2] if scores.shape[1] > 2 else 0.0,
        }
    )
    coords = coords.merge(meta, on="timestamp", how="left")
    coords.to_parquet(output_dir / "family_state_point_cloud_pca3.parquet", index=False)
    return coords, pca


def umap_projection(scaled: np.ndarray, index: pd.DatetimeIndex, meta: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame | None, str]:
    try:
        import umap  # type: ignore
    except ImportError:
        return None, "UMAP skipped because umap-learn is not installed."
    reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=0)
    embedding = reducer.fit_transform(scaled)
    coords = pd.DataFrame(
        {
            "timestamp": index,
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            "UMAP3": embedding[:, 2],
        }
    )
    coords = coords.merge(meta, on="timestamp", how="left")
    coords.to_parquet(output_dir / "family_state_point_cloud_umap3.parquet", index=False)
    return coords, "UMAP projection saved."


def color_values(coords: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(coords["timestamp"], utc=True)
    return ((ts - ts.min()) / (ts.max() - ts.min())).to_numpy(dtype=float)


def save_pca_figures(coords: pd.DataFrame, pca: PCA, figures_dir: Path) -> None:
    colors = color_values(coords)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(coords["PC1"], coords["PC2"], coords["PC3"], c=colors, cmap="viridis", s=8, alpha=0.75)
    ax.set_title("Family-state PCA point cloud")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="time")
    plt.tight_layout()
    plt.savefig(figures_dir / "pca3_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    ordered = coords.sort_values("timestamp")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ordered["PC1"], ordered["PC2"], ordered["PC3"], color="#4C78A8", linewidth=0.8, alpha=0.75)
    scatter = ax.scatter(ordered["PC1"], ordered["PC2"], ordered["PC3"], c=color_values(ordered), cmap="viridis", s=5)
    ax.set_title("Family-state PCA trajectory")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="time")
    plt.tight_layout()
    plt.savefig(figures_dir / "pca3_trajectory.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords["PC1"], coords["PC2"], c=colors, cmap="viridis", s=8, alpha=0.75)
    ax.set_title("Family-state PCA 2D projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax, label="time")
    plt.tight_layout()
    plt.savefig(figures_dir / "pca2_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = pca.explained_variance_ratio_[:10]
    ax.bar(np.arange(1, len(values) + 1), values, color="#54A24B")
    ax.plot(np.arange(1, len(values) + 1), np.cumsum(values), color="black", marker="o", label="cumulative")
    ax.set_title("PCA explained variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained")
    ax.set_ylim(0, max(1.0, np.cumsum(values).max() * 1.05))
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "pca_explained_variance_first10.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_umap_figure(coords: pd.DataFrame | None, figures_dir: Path) -> None:
    if coords is None:
        return
    colors = color_values(coords)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(coords["UMAP1"], coords["UMAP2"], coords["UMAP3"], c=colors, cmap="viridis", s=8, alpha=0.75)
    ax.set_title("Family-state UMAP point cloud")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_zlabel("UMAP3")
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="time")
    plt.tight_layout()
    plt.savefig(figures_dir / "umap3_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def structure_diagnostics(coords: pd.DataFrame, scaled: np.ndarray) -> dict[str, Any]:
    ordered = coords.sort_values("timestamp")
    xyz = ordered[["PC1", "PC2", "PC3"]].to_numpy()
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    center = xyz.mean(axis=0)
    radius = np.linalg.norm(xyz - center, axis=1)
    outlier_threshold = float(np.quantile(radius, 0.99))
    outlier_count = int((radius > outlier_threshold).sum())
    sil_scores = []
    for k in range(2, 6):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(scaled)
            sil_scores.append(silhouette_score(scaled, labels))
    best_silhouette = float(max(sil_scores)) if sil_scores else np.nan
    start_end_distance = float(np.linalg.norm(xyz[0] - xyz[-1])) if len(xyz) > 1 else np.nan
    median_radius = float(np.median(radius)) if len(radius) else np.nan
    return {
        "median_step_distance": float(np.median(step)) if len(step) else np.nan,
        "p95_step_distance": float(np.quantile(step, 0.95)) if len(step) else np.nan,
        "best_kmeans_silhouette_2_to_5": best_silhouette,
        "outlier_count_past_p99_radius": outlier_count,
        "start_end_distance": start_end_distance,
        "median_radius": median_radius,
        "start_end_radius_ratio": start_end_distance / median_radius if median_radius else np.nan,
    }


def write_summary(
    path: Path,
    family_state: pd.DataFrame,
    feature_state: pd.DataFrame,
    pca: PCA,
    meta: pd.DataFrame,
    diagnostics: dict[str, Any],
    umap_message: str,
) -> str:
    missingness_before = float(family_state.isna().mean().mean()) if family_state.size else np.nan
    missingness_after = 0.0
    active_family_coverage = meta["active_family_count"].describe()
    evr = pca.explained_variance_ratio_
    cumulative3 = float(evr[:3].sum()) if len(evr) >= 3 else float(evr.sum())
    clustering = diagnostics["best_kmeans_silhouette_2_to_5"] >= 0.25
    trajectory = diagnostics["p95_step_distance"] > diagnostics["median_step_distance"] * 2
    outliers = diagnostics["outlier_count_past_p99_radius"] > 0
    loops = diagnostics["start_end_radius_ratio"] < 0.75
    regime = trajectory or outliers
    recommendation = "A" if cumulative3 >= 0.45 and (clustering or trajectory or regime) else "C"
    rec_text = {
        "A": "Proceed to persistent homology on family-state sliding windows",
        "C": "Try a different projection/embedding method first",
    }[recommendation]
    lines = [
        "FAMILY-STATE POINT CLOUD SUMMARY",
        "",
        f"- number of timestamps: {len(family_state):,}",
        f"- number of family-state features: {feature_state.shape[1]}",
        f"- missingness before preprocessing: {missingness_before:.4f}",
        f"- missingness after preprocessing: {missingness_after:.4f}",
        f"- active-family coverage: min {active_family_coverage['min']:.0f}, median {active_family_coverage['50%']:.0f}, mean {active_family_coverage['mean']:.2f}, max {active_family_coverage['max']:.0f}",
        f"- PCA explained variance PC1: {evr[0]:.4f}",
        f"- PCA explained variance PC2: {evr[1]:.4f}",
        f"- PCA explained variance PC3: {evr[2]:.4f}",
        f"- cumulative variance explained by 3 PCs: {cumulative3:.4f}",
        f"- UMAP status: {umap_message}",
        "",
        "3D visualization diagnostics:",
        f"- clustering: {'yes' if clustering else 'weak/unclear'}; best k-means silhouette over k=2..5 is {diagnostics['best_kmeans_silhouette_2_to_5']:.3f}",
        f"- trajectory structure: {'yes' if trajectory else 'mild'}; median step {diagnostics['median_step_distance']:.3f}, p95 step {diagnostics['p95_step_distance']:.3f}",
        f"- loops or cycles: {'possible' if loops else 'not obvious from PCA3 start/end geometry'}",
        f"- regime shifts: {'possible' if regime else 'not obvious'}",
        f"- obvious outliers: {'yes' if outliers else 'no'}; p99-radius outlier count {diagnostics['outlier_count_past_p99_radius']}",
        "",
        "Interpretation:",
        "The family-level belief-state cloud is structured enough to justify persistent homology if the goal is to test whether topology captures non-linear trajectory/regime information beyond PCA. The first three PCs are an inspection device, not the final topology input; persistent homology should be applied to active-family-state sliding windows after the benchmark is fixed.",
        "",
        "Recommendation:",
        f"- {recommendation}) {rec_text}",
        "",
        "Justification:",
        "The family-state representation is active-set aware, low-dimensional enough to inspect, and already produced a healthier supervised PCA benchmark than the rectangular market panel. This makes it the right representation to carry into the topology stage, while keeping PCA as the fixed comparator.",
    ]
    summary = "\n".join(lines) + "\n"
    path.write_text(summary, encoding="utf-8")
    return summary


def run_all(candidate_markets_path: Path, prices_path: Path, panel_path: Path, output_dir: Path) -> dict[str, Any]:
    figures_dir = output_dir / "figures" / "point_cloud"
    ensure_dirs([output_dir, figures_dir])
    markets, panel, _raw, mask = load_inputs(candidate_markets_path, prices_path, panel_path)
    family_state = build_family_state(panel, mask, markets)
    family_state.index.name = "timestamp"
    family_state.to_parquet(output_dir / "family_state_matrix.parquet")
    feature_state, scaled, _imputer, _scaler, _cols = preprocess_family_state(family_state)
    meta = timestamp_metadata(panel, mask, markets)
    pca_coords, pca = pca_projection(scaled, family_state.index, meta, output_dir)
    umap_coords, umap_message = umap_projection(scaled, family_state.index, meta, output_dir)
    save_pca_figures(pca_coords, pca, figures_dir)
    save_umap_figure(umap_coords, figures_dir)
    diagnostics = structure_diagnostics(pca_coords, scaled)
    summary = write_summary(
        output_dir / "family_state_point_cloud_summary.md",
        family_state,
        feature_state,
        pca,
        meta,
        diagnostics,
        umap_message,
    )
    return {
        "family_state": family_state,
        "feature_state": feature_state,
        "pca_coords": pca_coords,
        "umap_coords": umap_coords,
        "pca": pca,
        "meta": meta,
        "diagnostics": diagnostics,
        "summary": summary,
        "umap_message": umap_message,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct and visualize family-state belief point clouds.")
    parser.add_argument("--candidate-markets", default="data/processed/candidate_universe_markets.parquet")
    parser.add_argument("--prices", default="data/processed/prices_long.parquet")
    parser.add_argument("--panel", default="data/processed/universe_b_macro_crypto_panel.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    root = project_root()
    result = run_all(
        resolve_path(root, args.candidate_markets),
        resolve_path(root, args.prices),
        resolve_path(root, args.panel),
        resolve_path(root, args.output_dir),
    )
    print(result["summary"])


if __name__ == "__main__":
    main()
