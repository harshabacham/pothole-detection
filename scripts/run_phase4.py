"""
scripts/run_phase4.py
=====================
Phase 4 orchestration: PCA dimensionality reduction, K-Means clustering,
DBSCAN density clustering, and comparative analysis.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.feature_scaling import get_feature_columns
from src.pca_model import fit_pca, transform_pca, plot_explained_variance, plot_pca_2d
from src.kmeans_clustering import experiment_kmeans, plot_elbow_and_metrics, fit_kmeans, visualize_cluster_gallery
from src.dbscan_clustering import experiment_dbscan, fit_dbscan, visualize_dbscan_2d
from src.clustering_evaluation import plot_clustering_comparison, generate_clustering_report


def main():
    print("======================================================================")
    print("  POTHOLE DETECTION - PHASE 4: PCA & CLUSTERING (K-MEANS + DBSCAN)")
    print("======================================================================")

    if not os.path.exists(config.SCALED_FEATURES_CSV):
        print(f"[ERROR] Scaled features file not found at: {config.SCALED_FEATURES_CSV}")
        print("Please ensure Phase 3 has been completed first.")
        sys.exit(1)

    df_scaled = pd.read_csv(config.SCALED_FEATURES_CSV)
    feat_cols = get_feature_columns(df_scaled)
    print(f"[INFO] Loaded {len(df_scaled)} scaled patch observations with {len(feat_cols)} features.")

    # ------------------------------------------------------------------
    # PART A: Principal Component Analysis (PCA)
    # ------------------------------------------------------------------
    print("\n-- PART A: Principal Component Analysis (PCA) -----------------------")
    pca_model, standardizer, n_components, _ = fit_pca(
        df_scaled,
        n_components=None,
        variance_target=config.PCA_VARIANCE_RATIO_TARGET,
        model_save_path=config.PCA_MODEL_PATH,
    )
    explained_var = float(np.sum(pca_model.explained_variance_ratio_))
    print(f"[INFO] Selected {n_components} principal components explaining {explained_var*100:.2f}% variance.")

    # Project into PCA space
    pca_df = transform_pca(df_scaled, pca_model, standardizer=standardizer, feature_cols=feat_cols)
    pca_df.to_csv(config.PCA_REDUCED_CSV, index=False)
    print(f"[INFO] Saved PCA-transformed dataset to {config.PCA_REDUCED_CSV}")

    # Visualizations
    pca_var_plot = os.path.join(config.METADATA_DIR, "pca_variance.png")
    pca_2d_plot = os.path.join(config.METADATA_DIR, "pca_2d.png")
    plot_explained_variance(pca_model, variance_target=config.PCA_VARIANCE_RATIO_TARGET, save_path=pca_var_plot)
    plot_pca_2d(pca_df, title=f"2D PCA Projection ({n_components} components fitted)", save_path=pca_2d_plot)

    # Use PCA features for subsequent clustering
    pca_cols = [f"pca_{i}" for i in range(n_components)]
    X_pca = pca_df[pca_cols].to_numpy()

    # ------------------------------------------------------------------
    # PART B: K-Means Clustering
    # ------------------------------------------------------------------
    print("\n-- PART B: K-Means Clustering ---------------------------------------")
    kmeans_results = experiment_kmeans(X_pca, k_range=config.KMEANS_K_RANGE)

    kmeans_elbow_plot = os.path.join(config.METADATA_DIR, "kmeans_elbow.png")
    plot_elbow_and_metrics(kmeans_results, save_path=kmeans_elbow_plot)

    # Fit selected K
    chosen_k = config.KMEANS_DEFAULT_K
    kmeans_model, kmeans_labels = fit_kmeans(X_pca, k=chosen_k, model_save_path=config.KMEANS_MODEL_PATH)
    print(f"[INFO] Fitted final K-Means with K={chosen_k}. Cluster sizes: {np.bincount(kmeans_labels)}")

    # Visual patch gallery per cluster
    meta_csv = os.path.join(config.PATCH_DIR, f"ps{config.FEATURE_PATCH_SIZE}", "metadata.csv")
    kmeans_gallery_plot = os.path.join(config.METADATA_DIR, "kmeans_cluster_gallery.png")
    if os.path.exists(meta_csv):
        metadata_df = pd.read_csv(meta_csv)
        visualize_cluster_gallery(metadata_df, kmeans_labels, n_samples_per_cluster=4, save_path=kmeans_gallery_plot)

    # ------------------------------------------------------------------
    # PART C: DBSCAN Density-Based Clustering
    # ------------------------------------------------------------------
    print("\n-- PART C: DBSCAN Density-Based Clustering --------------------------")
    dbscan_df = experiment_dbscan(X_pca)

    # Fit selected DBSCAN
    dbscan_model, dbscan_labels = fit_dbscan(
        X_pca,
        eps=config.DBSCAN_DEFAULT_EPS,
        min_samples=config.DBSCAN_DEFAULT_MIN_SAMPLES,
        model_save_path=config.DBSCAN_MODEL_PATH,
    )
    n_noise = int(np.sum(dbscan_labels == -1))
    n_clusters_db = len(set(dbscan_labels) - {-1})
    print(f"[INFO] DBSCAN (eps={config.DBSCAN_DEFAULT_EPS}, min_samples={config.DBSCAN_DEFAULT_MIN_SAMPLES}) -> {n_clusters_db} clusters, {n_noise} noise points.")

    dbscan_plot = os.path.join(config.METADATA_DIR, "dbscan_clusters.png")
    visualize_dbscan_2d(pca_df, dbscan_labels, save_path=dbscan_plot)

    # ------------------------------------------------------------------
    # PART D: Comparison & Summary Report
    # ------------------------------------------------------------------
    print("\n-- PART D: Clustering Comparison & Report ---------------------------")
    comp_plot = os.path.join(config.METADATA_DIR, "clustering_comparison.png")
    plot_clustering_comparison(pca_df, kmeans_labels, dbscan_labels, save_path=comp_plot)

    report_path = os.path.join(config.METADATA_DIR, "clustering_report.md")
    generate_clustering_report(
        pca_n_components=n_components,
        pca_explained_variance=explained_var,
        kmeans_results=kmeans_results,
        best_k=chosen_k,
        dbscan_df=dbscan_df,
        best_dbscan_eps=config.DBSCAN_DEFAULT_EPS,
        best_dbscan_min_samples=config.DBSCAN_DEFAULT_MIN_SAMPLES,
        n_dbscan_noise=n_noise,
        report_save_path=report_path,
    )

    # ------------------------------------------------------------------
    # Checklist verification
    # ------------------------------------------------------------------
    checks = [
        ("PCA model saved", os.path.exists(config.PCA_MODEL_PATH)),
        ("PCA reduced CSV created", os.path.exists(config.PCA_REDUCED_CSV)),
        ("PCA variance plot saved", os.path.exists(pca_var_plot)),
        ("2D PCA projection saved", os.path.exists(pca_2d_plot)),
        ("K-Means elbow & metrics plot saved", os.path.exists(kmeans_elbow_plot)),
        ("K-Means model saved", os.path.exists(config.KMEANS_MODEL_PATH)),
        ("K-Means cluster gallery saved", os.path.exists(kmeans_gallery_plot)),
        ("DBSCAN clusters plot saved", os.path.exists(dbscan_plot)),
        ("DBSCAN model saved", os.path.exists(config.DBSCAN_MODEL_PATH)),
        ("Clustering comparison plot saved", os.path.exists(comp_plot)),
        ("Clustering markdown report saved", os.path.exists(report_path)),
    ]

    print("\n======================================================================")
    print("  PHASE 4 COMPLETION CHECKLIST")
    print("======================================================================")
    all_passed = True
    for desc, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {desc}")
        if not passed:
            all_passed = False

    print("======================================================================")
    if all_passed:
        print("  All Phase 4 checks passed - PCA & Clustering pipeline complete!")
    else:
        print("  Some Phase 4 checks failed.")
        sys.exit(1)
    print("======================================================================\n")


if __name__ == "__main__":
    main()
