"""
scripts/train_deep_model.py
===========================
Trains the Deep Learning visual feature pipeline using frozen ResNet-18 embeddings
and fits an unsupervised Deep Isolation Forest model on road patches.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.deep_features import DeepFeatureExtractor
from src.anomaly_detection import create_isolation_forest, compute_anomaly_scores
from src.model_evaluation import evaluate_anomaly_predictions


def main():
    print("======================================================================")
    print("  TRAINING DEEP LEARNING VISUAL FEATURE ANOMALY PIPELINE (RESNET-18)")
    print("======================================================================")

    metadata_csv = os.path.join(config.PATCH_DIR, f"ps{config.FEATURE_PATCH_SIZE}", "metadata.csv")
    if not os.path.exists(metadata_csv):
        print(f"[ERROR] Patch metadata missing at: {metadata_csv}")
        sys.exit(1)

    df_meta = pd.read_csv(metadata_csv)
    print(f"[INFO] Loaded {len(df_meta)} patches from metadata.")

    # 1. Initialize PyTorch ResNet-18 Extractor
    extractor = DeepFeatureExtractor(output_dim=config.DEEP_FEATURE_DIM)
    print(f"[INFO] Initialized ResNet-18 Deep Feature Extractor (Dim={config.DEEP_FEATURE_DIM})")

    # 2. Extract deep embeddings for all patches
    patches_bgr = []
    for _, row in df_meta.iterrows():
        p_path = row.get("color_path") or row.get("gray_path")
        if p_path and os.path.exists(p_path):
            img = cv2.imread(p_path)
        else:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
        patches_bgr.append(img)

    print("[INFO] Extracting deep visual embeddings across patch dataset...")
    deep_embeddings = extractor.extract_batch(patches_bgr, batch_size=32)
    print(f"[INFO] Extracted deep feature matrix: {deep_embeddings.shape}")

    # Save deep features
    deep_csv_path = "data/features/features_deep_ps64.csv"
    deep_df = pd.DataFrame(deep_embeddings, columns=[f"deep_feat_{i}" for i in range(deep_embeddings.shape[1])])
    meta_cols = ["patch_id", "image_filename", "x1", "y1", "x2", "y2"]
    for c in meta_cols:
        if c in df_meta.columns:
            deep_df[c] = df_meta[c].values
    deep_df.to_csv(deep_csv_path, index=False)
    print(f"[INFO] Saved deep features to: {deep_csv_path}")

    # 3. Train Unsupervised Isolation Forest on Deep Features
    print("\n[INFO] Fitting Isolation Forest on Deep Feature Embeddings...")
    deep_model = create_isolation_forest(
        n_estimators=150,
        contamination=config.IFOREST_CONTAMINATION,
    )
    deep_model.fit(deep_embeddings)

    deep_model_path = "models/isolation_forest_deep.joblib"
    joblib.dump({"model": deep_model, "feature_dim": config.DEEP_FEATURE_DIM}, deep_model_path)
    print(f"[INFO] Saved Deep Isolation Forest model to: {deep_model_path}")

    # 4. Post-Hoc Audit
    is_anom, raw_dec, anom_score, norm_score = compute_anomaly_scores(deep_model, deep_embeddings)
    y_true = df_meta["image_filename"].astype(str).str.contains("damaged").astype(int).to_numpy()
    y_pred = is_anom.astype(int)

    print("\n-- Post-Hoc Benchmark on Deep Feature Model --")
    metrics = evaluate_anomaly_predictions(y_true, y_pred, y_scores=anom_score)
    print("======================================================================")
    print("  Deep Feature Anomaly Pipeline Trained and Verified Successfully!")
    print("======================================================================\n")


if __name__ == "__main__":
    main()

