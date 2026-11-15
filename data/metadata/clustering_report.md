# Phase 4 — Unsupervised Dimensionality Reduction & Clustering Report

## 1. Dimensionality Reduction (PCA)
- **Original Feature Dimension**: 45 numerical features
- **Retained Components**: 9 components
- **Cumulative Explained Variance**: 91.11% (Target: 90%)
- **Observations**: The first 2–3 principal components capture dominant road texture uniformity and brightness gradients. Beyond that, higher components encode subtle high-frequency edge irregularities.

## 2. Partition-Based Clustering (K-Means)
- **Selected K**: 3
- **Inertia Trend**: The elbow curve shows consistent drop through K=2 to K=4, with diminishing returns beyond K=4.
- **Cluster Characteristics**:
  - Cluster assignments group patches broadly by macro-brightness and edge density.
  - Smooth road patches form a dense central core.
  - Distressed patches (containing cracks or darker regions) scatter toward the periphery.
- **Methodological Note**: A K-Means cluster does not represent a "pothole class". It represents a geometric partition of the feature space.

## 3. Density-Based Clustering (DBSCAN)
- **Parameters**: eps=1.5, min_samples=5
- **Isolated Noise Points**: 333 patches
- **Observations**:
  - Normal homogeneous asphalt patches pack into dense core regions.
  - Highly irregular road sections with severe texture disruptions fail the density criteria and are flagged as noise (-1).
- **Methodological Note**: DBSCAN noise points are not automatically potholes; they merely represent low-density deviations from standard road textures (could include lane markings, shadows, debris, or potholes).

## 4. Synthesis for Anomaly Detection
Clustering demonstrates that standard road surfaces occupy a compact feature manifold, whereas abnormal asphalt structures manifest as peripheral or low-density points. This validates the rationale for dedicated boundary/density models like Isolation Forest in Phase 5.
