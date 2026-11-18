"""
src/anomaly_detection.py
========================
Core anomaly detection logic and score normalization.

Unsupervised algorithms implemented:
  1. Isolation Forest  — axis-aligned random splits to isolate anomalies.
  2. Extended Isolation Forest (EIF) — uses random hyperplane (oblique) cuts
     instead of axis-aligned splits, producing cleaner anomaly boundaries and
     better separating directional feature shifts (e.g. vehicle vibration)
     from true road surface anomalies.
  3. PCA Reconstruction Error — after projecting patch features into the
     low-dimensional PCA subspace and reconstructing them, the mean-squared
     error between original and reconstruction acts as an independent anomaly
     signal: anomalous patches lie far from the learned manifold.
"""

import os
import sys
import numpy as np
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


# ---------------------------------------------------------------------------
# Extended Isolation Forest (EIF) — soft dependency
# ---------------------------------------------------------------------------
try:
    import eif as _eif_lib
    EIF_AVAILABLE = True
except ImportError:
    EIF_AVAILABLE = False



def create_isolation_forest(
    n_estimators: int = None,
    contamination: float = None,
    max_samples=None,
    random_state: int = None,
) -> IsolationForest:
    """
    Factory function to instantiate an Isolation Forest model with project defaults.
    """
    if n_estimators is None:
        n_estimators = config.IFOREST_N_ESTIMATORS
    if contamination is None:
        contamination = config.IFOREST_CONTAMINATION
    if max_samples is None:
        max_samples = config.IFOREST_MAX_SAMPLES
    if random_state is None:
        random_state = config.IFOREST_RANDOM_STATE

    return IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )


def compute_anomaly_scores(
    model: IsolationForest,
    X: np.ndarray,
) -> tuple:
    """
    Compute anomaly predictions and calibrated anomaly scores.

    Notes on Isolation Forest Scoring:
    ---------------------------------
    In scikit-learn, decision_function(X) yields:
      - Positive values for standard inliers.
      - Negative values for isolated outliers/anomalies.
    To make the score intuitive (higher value = more anomalous):
      anomaly_score = -raw_decision_score
    Additionally, we compute a normalized score scaled between [0.0, 1.0]
    relative to the observed dataset distribution.

    DO NOT interpret this score as a probability or confidence value.
    It is a relative distance/isolation index.

    Parameters
    ----------
    model : IsolationForest
        Fitted IsolationForest model.
    X : np.ndarray
        Feature matrix (N samples, D features).

    Returns
    -------
    tuple
        (is_anomaly_bool, raw_decision_score, anomaly_score, normalized_anomaly_score)
    """
    raw_decision = model.decision_function(X)  # Higher is normal, lower is anomalous
    preds = model.predict(X)  # 1 for inlier, -1 for outlier

    is_anomaly = (preds == -1)

    # Invert so higher score = more anomalous
    anomaly_score = -raw_decision

    # Normalize across the batch to [0.0, 1.0] for consistent visual mapping
    s_min = np.min(anomaly_score)
    s_max = np.max(anomaly_score)
    if s_max > s_min:
        norm_score = (anomaly_score - s_min) / (s_max - s_min)
    else:
        norm_score = np.zeros_like(anomaly_score)

    return is_anomaly, raw_decision, anomaly_score, norm_score


# ---------------------------------------------------------------------------
# Extended Isolation Forest (EIF)
# ---------------------------------------------------------------------------

def _c_factor(n: int) -> float:
    """Average path length of unsuccessful search in a Binary Search Tree."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    euler = 0.5772156649
    return 2.0 * (np.log(n - 1) + euler) - 2.0 * (n - 1) / n


class _NumpyExtendedTree:
    """Pure-NumPy Extended Isolation Tree node with random hyperplane cuts."""
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.normal = None
        self.intercept = None
        self.left = None
        self.right = None
        self.size = 0
        self.is_leaf = True

    def fit(self, X: np.ndarray, depth: int, rng: np.random.RandomState):
        self.size = len(X)
        if depth >= self.max_depth or self.size <= 1:
            self.is_leaf = True
            return

        d = X.shape[1]
        n = rng.normal(size=d)
        norm = np.linalg.norm(n)
        if norm > 1e-9:
            n = n / norm
        else:
            self.is_leaf = True
            return

        projections = X.dot(n)
        p_min, p_max = float(projections.min()), float(projections.max())
        if np.isclose(p_min, p_max):
            self.is_leaf = True
            return

        intercept = rng.uniform(p_min, p_max)
        left_mask = projections <= intercept
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            self.is_leaf = True
            return

        self.normal = n
        self.intercept = intercept
        self.is_leaf = False

        self.left = _NumpyExtendedTree(self.max_depth)
        self.left.fit(X[left_mask], depth + 1, rng)
        self.right = _NumpyExtendedTree(self.max_depth)
        self.right.fit(X[right_mask], depth + 1, rng)

    def path_length(self, X: np.ndarray) -> np.ndarray:
        if self.is_leaf:
            c = _c_factor(self.size)
            return np.full(len(X), c, dtype=float)

        projections = X.dot(self.normal)
        left_mask = projections <= self.intercept
        lengths = np.zeros(len(X), dtype=float)

        if np.any(left_mask):
            lengths[left_mask] = 1.0 + self.left.path_length(X[left_mask])
        if np.any(~left_mask):
            lengths[~left_mask] = 1.0 + self.right.path_length(X[~left_mask])
        return lengths


def compute_eif_anomaly_scores(
    X: np.ndarray,
    ntrees: int = 100,
    sample_size: int = 256,
    random_state: int = 42,
) -> tuple:
    """
    Score anomalies using Extended Isolation Forest (EIF).

    EIF difference from standard IForest:
    -------------------------------------
    Standard Isolation Forest always splits along axis-aligned hyperplanes
    (one feature at a time). This creates artificial score ridges along the
    feature axes that can mis-score directional road artefacts (e.g. lane
    marking edges appear anomalous along the horizontal axis).

    EIF uses randomly oriented hyperplanes (random intercept + random normal
    vector), producing smoother, rotation-invariant anomaly boundaries.

    Uses native 'eif' C-extension if available; otherwise uses high-speed
    vectorized NumPy implementation.

    Returns
    -------
    tuple : (eif_scores_normalized, raw_scores)
        eif_scores_normalized: float array [0,1], higher = more anomalous.
        raw_scores: raw EIF anomaly scores.
    """
    N = len(X)
    if N == 0:
        return np.array([]), np.array([])

    if EIF_AVAILABLE:
        try:
            subsample = min(sample_size, N)
            forest = _eif_lib.iForest(X, ntrees=ntrees, sample_size=subsample)
            raw_scores = np.array(forest.compute_paths(X_in=X), dtype=float)
            s_min, s_max = raw_scores.min(), raw_scores.max()
            if s_max > s_min:
                norm_scores = (raw_scores - s_min) / (s_max - s_min)
            else:
                norm_scores = np.zeros_like(raw_scores)
            return norm_scores, raw_scores
        except Exception as exc:
            print(f"[INFO] Using native NumPy EIF engine: {exc}")

    # Pure NumPy implementation
    rng = np.random.RandomState(random_state)
    subsample = min(sample_size, N)
    max_depth = int(np.ceil(np.log2(max(subsample, 2))))
    c_sub = _c_factor(subsample)

    path_matrix = np.zeros((ntrees, N), dtype=float)
    for i in range(ntrees):
        indices = rng.choice(N, size=subsample, replace=False) if N > subsample else np.arange(N)
        tree = _NumpyExtendedTree(max_depth=max_depth)
        tree.fit(X[indices], depth=0, rng=rng)
        path_matrix[i] = tree.path_length(X)

    mean_paths = np.mean(path_matrix, axis=0)
    # Anomaly score: s = 2 ^ (- mean_path / c_sub)
    if c_sub > 0:
        raw_scores = 2.0 ** (-mean_paths / c_sub)
    else:
        raw_scores = np.zeros(N, dtype=float)

    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        norm_scores = (raw_scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.zeros_like(raw_scores)

    return norm_scores, raw_scores


# ---------------------------------------------------------------------------
# PCA Reconstruction Error — Secondary Anomaly Signal
# ---------------------------------------------------------------------------

def compute_pca_reconstruction_error(
    X_original: np.ndarray,
    pca_model,
    pca_standardizer=None,
) -> np.ndarray:
    """
    Compute per-sample PCA reconstruction error (mean squared error).

    Theory:
    -------
    PCA learns the principal directions that explain maximum variance in
    *normal* road surface patches. When an anomalous patch (pothole, crack)
    is projected into the low-dimensional PCA subspace and then reconstructed
    back to the original feature space, the reconstruction is poor — the patch
    lies far from the learned normal manifold.

    This MSE serves as an independent, complementary anomaly signal alongside
    the Isolation Forest score.

    Parameters
    ----------
    X_original : np.ndarray  (N, D)
        Pre-scaled feature matrix (the same features fed to PCA).
    pca_model : sklearn.decomposition.PCA
        Fitted PCA object (has components_, mean_).
    pca_standardizer : StandardScaler or None
        If used as a pre-PCA standardizer, apply the same transform first.

    Returns
    -------
    np.ndarray (N,) — reconstruction MSE per patch, normalized [0, 1].
    """
    X = X_original.copy()

    if pca_standardizer is not None:
        try:
            X = pca_standardizer.transform(X)
        except Exception:
            pass

    # Project to PCA subspace then reconstruct
    X_pca = pca_model.transform(X)
    X_reconstructed = pca_model.inverse_transform(X_pca)

    # Per-sample MSE
    mse = np.mean((X - X_reconstructed) ** 2, axis=1)

    # Normalize to [0, 1]
    mse_min, mse_max = mse.min(), mse.max()
    if mse_max > mse_min:
        mse_norm = (mse - mse_min) / (mse_max - mse_min)
    else:
        mse_norm = np.zeros_like(mse)

    return mse_norm


def combine_anomaly_signals(
    iforest_norm_scores: np.ndarray,
    pca_recon_errors: np.ndarray,
    iforest_weight: float = 0.70,
    recon_weight: float = 0.30,
) -> np.ndarray:
    """
    Fuse Isolation Forest score with PCA reconstruction error into a single
    combined anomaly confidence.

    Both signals measure different properties of anomalousness:
      - IForest: isolation in feature space (structural outlier distance)
      - PCA recon error: deviation from the learned normal data manifold

    Combined score is a weighted average, normalized to [0, 1].
    """
    combined = iforest_weight * iforest_norm_scores + recon_weight * pca_recon_errors
    c_min, c_max = combined.min(), combined.max()
    if c_max > c_min:
        return (combined - c_min) / (c_max - c_min)
    return np.zeros_like(combined)

