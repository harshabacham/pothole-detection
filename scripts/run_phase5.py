"""
scripts/run_phase5.py
=====================
Phase 5 orchestration: Isolation Forest anomaly detection, threshold experiments,
anomaly scoring, prediction persistence, and post-hoc evaluation.
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
from src.model_training import train_isolation_forest, experiment_contamination, plot_contamination_experiments
from src.model_prediction import predict_patches, save_predictions
from src.model_evaluation import evaluate_anomaly_predictions, plot_confusion_matrix, plot_score_distribution_by_class


def main():
    print("======================================================================")
    print("  POTHOLE DETECTION - PHASE 5: ISOLATION FOREST ANOMALY DETECTION")
    print("======================================================================")

    # 1. Feature Representation Selection
    if config.USE_PCA_FOR_ANOMALY and os.path.exists(config.PCA_REDUCED_CSV):
        print(f"[INFO] Using orthogonal PCA features from: {config.PCA_REDUCED_CSV}")
        data_df = pd.read_csv(config.PCA_REDUCED_CSV)
        feature_cols = [c for c in data_df.columns if c.startswith("pca_")]
    else:
        print(f"[INFO] Using scaled feature space from: {config.SCALED_FEATURES_CSV}")
        data_df = pd.read_csv(config.SCALED_FEATURES_CSV)
        from src.feature_scaling import get_feature_columns
        feature_cols = get_feature_columns(data_df)

    X = data_df[feature_cols].to_numpy()
    print(f"[INFO] Dataset shape for anomaly detection: {X.shape[0]} samples, {X.shape[1]} features")

    # Metadata alignment
    meta_csv = os.path.join(config.PATCH_DIR, f"ps{config.FEATURE_PATCH_SIZE}", "metadata.csv")
    metadata_df = pd.read_csv(meta_csv) if os.path.exists(meta_csv) else data_df

    # 2. Contamination Sensitivity Experiments
    print("\n-- STEP 1: Contamination Parameter Experiments ---------------------")
    contam_df = experiment_contamination(X, config.IFOREST_CONTAMINATION_EXPERIMENTS)
    contam_plot = os.path.join(config.METADATA_DIR, "iforest_contamination_experiments.png")
    plot_contamination_experiments(contam_df, save_path=contam_plot)

    # 3. Model Training (Fully Unsupervised — zero label leakage)
    print("\n-- STEP 2: Training Production Isolation Forest --------------------")
    model, artifact = train_isolation_forest(
        X,
        contamination=config.IFOREST_CONTAMINATION,
        n_estimators=config.IFOREST_N_ESTIMATORS,
        feature_names=feature_cols,
        save_path=config.IFOREST_MODEL_PATH,
    )

    # 4. Patch Prediction, Continuous Scoring, and Ranking
    print("\n-- STEP 3: Computing Anomaly Scores & Rankings ---------------------")
    predictions_df = predict_patches(model, X, metadata_df=metadata_df)
    save_predictions(predictions_df, save_path=config.ANOMALY_PREDICTIONS_CSV)

    n_anomalies = int(predictions_df["is_anomaly"].sum())
    print(f"[INFO] Detected {n_anomalies}/{len(predictions_df)} anomalous patches ({(n_anomalies/len(predictions_df))*100:.1f}%)")

    # 5. Post-Hoc Ground-Truth Evaluation (Validation only — never used for training)
    print("\n-- STEP 4: Post-Hoc Evaluation on Benchmark Splits -----------------")
    # Ground truth: patches originating from damaged road images represent abnormal surface candidates
    y_true = predictions_df["image_filename"].astype(str).str.contains("damaged").astype(int).to_numpy()
    y_pred = predictions_df["is_anomaly"].astype(int).to_numpy()
    y_scores = predictions_df["anomaly_score"].to_numpy()

    metrics = evaluate_anomaly_predictions(y_true, y_pred, y_scores=y_scores)

    cm_plot = os.path.join(config.METADATA_DIR, "iforest_confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names=["Normal Road", "Damaged Surface"], save_path=cm_plot)

    score_dist_plot = os.path.join(config.METADATA_DIR, "iforest_score_distribution.png")
    plot_score_distribution_by_class(y_true, y_scores, class_names=["Normal Road", "Damaged Surface"], save_path=score_dist_plot)

    # 6. Checklist Verification
    checks = [
        ("Trained Isolation Forest saved", os.path.exists(config.IFOREST_MODEL_PATH)),
        ("Anomaly predictions CSV saved", os.path.exists(config.ANOMALY_PREDICTIONS_CSV)),
        ("Contamination experiments plot saved", os.path.exists(contam_plot)),
        ("Confusion matrix plot saved", os.path.exists(cm_plot)),
        ("Score distribution comparison plot saved", os.path.exists(score_dist_plot)),
    ]

    print("\n======================================================================")
    print("  PHASE 5 COMPLETION CHECKLIST")
    print("======================================================================")
    all_passed = True
    for desc, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {desc}")
        if not passed:
            all_passed = False

    print("======================================================================")
    if all_passed:
        print("  All Phase 5 checks passed - Isolation Forest pipeline complete!")
    else:
        print("  Some Phase 5 checks failed.")
        sys.exit(1)
    print("======================================================================\n")


if __name__ == "__main__":
    main()

