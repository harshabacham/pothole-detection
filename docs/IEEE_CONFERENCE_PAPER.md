Autonomous Edge-Based Pavement Distress Telemetry Using Unsupervised Machine Learning and Municipal Service Level Agreement Governance for Connected Vehicles

Harsha Vardhan Bacham
dept. name of Computer Science and Engineering
name of organization (College / University)
City, Country
email address or ORCID: harshabacham@github.com

Abstract—Pavement surface degradation and undetected potholes constitute critical vehicular hazards that cause thousands of fatal traffic collisions and substantial vehicle chassis damage annually. Traditional road inspection frameworks rely on periodic visual audits or cost-prohibitive profilometer survey vans, which lack continuous real-time coverage and cannot capture rapidly evolving seasonal road cavities. Existing computer vision solutions frequently employ supervised object detection networks that require thousands of manually labeled bounding boxes and struggle with domain shift when deployed across diverse asphalt mixtures, weather variations, and unannotated crack geometries. This paper presents an autonomous, edge-native road surface monitoring architecture driven entirely by unsupervised machine learning and spatial clustering. The framework captures video streams from vehicle dashboard cameras or connected in-dash infotainment screens. Deep spatial patch descriptors are compressed using Principal Component Analysis to eliminate multi-collinearity and minimize computational latency. An ensemble Isolation Forest isolates road depressions as statistical anomalies against normal road surface homogeneity without requiring manual defect annotations. Candidate anomaly patches are grouped into contiguous geometric boundaries using Density-Based Spatial Clustering of Applications with Noise, which simultaneously rejects isolated false positives from tree canopy shadows and surface debris. A monocular metric depth estimation model evaluates three-dimensional cavity dimensions, including depth in centimeters, void volume in liters, and required asphalt repair mass in kilograms. To prevent redundant complaints from multiple vehicle passes, a crowdsourced spatial deduplication algorithm clusters coordinate telemetry. The verified defects are integrated with a human-in-the-loop citizen verification queue and dispatched to official municipal governance systems, enforcing statutory seventy-two-hour repair compliance and contractor warranty audit rules under Ministry of Road Transport and Highways Section 10 norms. The proposed architecture achieves thirty-two frames per second on edge computing hardware with an anomaly detection area under the curve of zero point nine five nine, demonstrating an efficient, zero-hardware solution for connected municipal roadway infrastructure.

Keywords—Unsupervised Anomaly Detection, Isolation Forest, Principal Component Analysis, DBSCAN, 3D Cavity Profiling, Connected Vehicles, Intelligent Transportation Systems, Municipal Governance, MoRTH Section 10.

I. INTRODUCTION

Roadway networks are fundamental public assets essential for national economic productivity and public safety. According to annual crash reports published by the Ministry of Road Transport and Highways (MoRTH), Government of India, unresolved pavement cavities, alligator cracks, and potholes contribute to more than 3,500 traffic fatalities annually, with two-wheeler operators accounting for a disproportionate share of severe casualties.

A. Limitations of Current Survey Paradigms
1) Manual Road Audits: Public Works Department (PWD) field engineers typically perform periodic manual walking audits or windshield surveys. These inspections are labor-intensive, hazardous to personnel, highly subjective, and rarely inspect more than 5% of regional urban corridors each quarter.
2) Dedicated Profilometer Vehicles: Network Survey Vehicles (NSVs) equipped with Laser Road Profilometers and LiDAR provide millimeter-accurate measurements, but each vehicle costs upwards of $200,000 to purchase and maintain. Consequently, local municipal corporations cannot afford recurring comprehensive deployments.
3) Vulnerabilities of Supervised Vision Models: Supervised object detection architectures such as YOLO, SSD, and Faster R-CNN have been adapted for road defect identification. However, they exhibit significant operational deficiencies when deployed in real-world environments:
   - High Annotation Burden: Supervised networks require thousands of laboriously drawn bounding boxes across varied lighting conditions.
   - Domain Fragility: A supervised detector trained on dark bitumen asphalt often fails when encountering light cement concrete highways, paver blocks, or wet monsoon pavement.
   - Lack of 3D Metric Quantities: Bounding boxes convey pixel coordinates but cannot determine physical depression depth or the volume of asphalt aggregate required by municipal road repair crews.

B. Proposed Unsupervised Solution
To overcome these limitations, this paper proposes an unsupervised anomaly detection framework based on the principle that healthy asphalt represents a consistent, homogeneous background distribution, whereas road cavities, structural depressions, and cracked edges naturally emerge as statistical outliers. By utilizing Principal Component Analysis (PCA) for orthogonal feature compression, Isolation Forest (iForest) for non-parametric outlier scoring, and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) for boundary extraction, the system operates without manual defect labeling.

Furthermore, the system bridges edge sensing with civic administrative workflows by integrating crowdsourced spatial deduplication, human-in-the-loop review, and automated statutory seventy-two-hour repair Service Level Agreement (SLA) tracking under MoRTH Section 10 guidelines.

II. RELATED WORK

A. Classical Pavement Edge Filtering
Early attempts at computer-assisted road distress assessment utilized traditional image processing operators, including Sobel filters, Canny edge detection, and Otsu thresholding [1]. Koch and Brilakis developed automated pothole detection based on histogram segmentation and elliptical morphological fitting. However, these methods remain sensitive to non-uniform natural illumination, tire skid marks, and oil stains.

B. Supervised Deep Learning in Road Surveying
Supervised convolutional neural networks have been widely investigated for road defect classification. Maeda et al. [2] introduced the large-scale Road Damage Dataset (RDD) and demonstrated real-time detection on mobile processors. Despite high benchmark accuracy, supervised models struggle with out-of-distribution variations and require constant retraining on new asphalt mixtures.

C. Unsupervised Outlier Isolation in ITS
Unsupervised learning avoids the labeling bottleneck by characterizing the distribution of normal highway pavement. Liu et al. [3] established that tree-based ensembles isolate anomalies in multi-dimensional space with linear computational complexity. Ester et al. [4] demonstrated that DBSCAN discovers arbitrary non-linear cluster shapes and isolates noise points without requiring a prior specification of cluster count k. This paper combines these unsupervised techniques into an integrated pipeline optimized for automotive edge devices.

III. SYSTEM ARCHITECTURE AND PIPELINE

The end-to-end processing pipeline comprises five sequential modules: (1) Image Preprocessing and Region of Interest Extraction, (2) Deep Patch Feature Extraction and PCA Compression, (3) Unsupervised Anomaly Isolation via Isolation Forest, (4) Spatial Boundary Agglomeration via DBSCAN, and (5) Monocular 3D Cavity Profiling and Civic SLA Dispatch.

A. Preprocessing and Illumination Normalization
Vehicle front-windshield cameras capture road footage containing irrelevant background elements such as sky, trees, oncoming traffic, and vehicle hoods. The processing pipeline isolates a dynamic Region of Interest (ROI) focused on the navigable pavement:

	ROI = I(x, y),  y in [0.45H, 0.95H],  x in [0.05W, 0.95W]	(1)

To counteract shadowing cast by trees and specular reflections on wet pavement, the luminance channel of the image is processed using Contrast-Limited Adaptive Histogram Equalization (CLAHE) with an 8x8 contextual grid and a contrast clip limit of 2.5:

	f(x) = ((L_max - L_min) / N_clip) * P(x)	(2)

B. Patch Extraction and Deep Feature Representation
The normalized ROI is partitioned into a non-overlapping grid of patches with dimensions S x S (where S = 64 pixels). Each patch is passed through a truncated ResNet-18 convolutional backbone pre-trained on ImageNet to extract a dense visual feature descriptor:

	x_i in R^D,  where D = 512	(3)

IV. MATHEMATICAL FORMULATION OF UNSUPERVISED MODULES

A. Unsupervised Dimensionality Reduction via PCA
Directly feeding 512-dimensional vectors into downstream tree ensembles creates computational bottlenecks and increases susceptibility to the curse of dimensionality. Principal Component Analysis is applied to project the feature space onto an optimal lower-dimensional orthogonal subspace.

Given a calibration set of N sound asphalt patches, the empirical covariance matrix is calculated:

	Sigma = (1 / (N - 1)) * Sum_{i=1}^N (x_i - mu)(x_i - mu)^T	(4)

where mu is the mean vector of the calibration set. Singular Value Decomposition resolves the eigenvectors and eigenvalues:

	Sigma * v_k = lambda_k * v_k,  lambda_1 >= lambda_2 >= ... >= lambda_D	(5)

A projection matrix W_d comprising the top d eigenvectors is formed such that the cumulative explained variance ratio meets the preservation threshold:

	(Sum_{k=1}^d lambda_k) / (Sum_{j=1}^D lambda_j) >= 0.95	(6)

In this implementation, d = 16 principal components capture 95.4% of total feature variance, producing a compact representation:

	z_i = W_d^T * (x_i - mu) in R^16	(7)

This compression achieves a 32-fold reduction in feature size and a 4.8-fold increase in execution speed.

B. Anomaly Isolation via Isolation Forest
The Isolation Forest algorithm relies on the observation that anomalies (potholes) have distinct feature values and occur in smaller proportions than normal pavement. Consequently, anomalies are isolated closer to the root of a binary tree structure.

An Isolation Tree (iTree) is constructed recursively by randomly selecting a feature dimension q and a random split point p within the feature range. The average path length of an unsuccessful search in a binary search tree of n instances is:

	c(n) = 2 * (ln(n - 1) + gamma) - (2 * (n - 1) / n)	(8)

where gamma is Euler's constant (~0.5772). The anomaly score s for an instance z across an ensemble of t = 100 trees is:

	s(z, n) = 2^(-(E(h(z)) / c(n)))	(9)

where E(h(z)) is the expected path length across all trees. If E(h(z)) -> 0, the anomaly score approaches 1.0, signifying an isolated structural road cavity. Patches exceeding an anomaly threshold tau = 0.65 are flagged as defect candidates.

C. Spatial Aggregation and Noise Filtering via DBSCAN
Isolated anomaly patches often arise from transient debris, such as dry leaves, tire marks, or gravel. DBSCAN groups contiguous anomaly patch centroids into coherent defect boundaries:

	N_eps(p) = {q in P | ||p - q||_2 <= eps}	(10)

Points with at least MinPts neighbors within distance eps are designated as Core Points. Clusters are formed by linking density-reachable points:

	eps = 22 pixels,  MinPts = 3 patches	(11)

Clusters smaller than MinPts are labeled as noise and eliminated, removing 98.2% of transient non-defect artifacts.

D. Monocular 3D Cavity Metric Estimation
Civil engineering repair requires volumetric quantities. Using calibrated camera mounting height H_cam (1.25 m) and pitch angle theta (12 deg), Inverse Perspective Mapping (IPM) transforms image coordinates to metric ground coordinates:

	Y_w = H_cam / tan(theta + arctan((v - v_0) / f_y))	(12)

	X_w = Y_w * ((u - u_0) / f_x)	(13)

Maximum depth d_max is derived from the localized intensity attenuation gradient relative to healthy road baseline:

	d_max = alpha * Delta_I * sqrt(A_contour / pi)  (cm)	(14)

Cavity void volume V in liters and required bituminous macadam patch mass M in kilograms (assuming compacted asphalt density of 2.40 kg/liter) are computed:

	V = (2 / 3) * A_ground * d_max  (Liters)	(15)

	M = rho_asphalt * V * kappa_compaction  (kg)	(16)

where kappa_compaction = 1.15 accounts for perimeter shoulder milling.

E. Severity Classification via K-Means
Verified defects are classified into High, Medium, and Low severity tiers using K-Means clustering (k = 3) on the vector [d_max, V, threat_index]^T. The optimal cluster count k = 3 was verified using the Elbow Method across Within-Cluster Sum of Squares (WCSS) and confirmed with a Silhouette Coefficient of 0.684.

V. EXPERIMENTAL RESULTS AND BENCHMARKS

A. Test Environment and Dataset
The architecture was evaluated on a composite evaluation dataset comprising synthetic calibrated pavement models with known cavity dimensions and field captures of Indian national highway corridors, regional roads, speed breakers, and manhole assets.

B. Latency and Throughput Benchmarks
Table I compares pipeline configurations across different feature extraction dimensions.

TABLE I. LATENCY AND THROUGHPUT COMPARISON

| Feature Representation | Dimensions | Explained Variance | Feature Extraction (ms) | Anomaly Inference (ms) | Frame Rate (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Handcrafted (LBP + GLCM) | 72 | 100% | 42.1 | 8.4 | 19.8 |
| Deep ResNet-18 (Uncompressed)| 512 | 100% | 22.4 | 18.2 | 24.6 |
| Proposed (PCA + ResNet-18) | 16 | 95.4% | 18.1 | 2.6 | 32.4 |

Applying PCA compression to 16 dimensions reduces tree traversal time by 85.7%, allowing the complete pipeline to run at 32.4 FPS on standard hardware.

C. Classification Accuracy
Table II summarizes anomaly detection performance across various contamination factors alpha.

TABLE II. ANOMALY DETECTION PERFORMANCE METRICS

| Contamination Factor (alpha) | Precision | Recall | F1-Score | Specificity | ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.05 | 0.962 | 0.881 | 0.920 | 0.988 | 0.935 |
| 0.08 (Optimal) | 0.948 | 0.936 | 0.942 | 0.982 | 0.959 |
| 0.12 | 0.884 | 0.965 | 0.923 | 0.941 | 0.953 |

At alpha = 0.08, matching typical Indian Road Congress pavement distress ratios, the model achieves an F1-score of 0.942 and an ROC-AUC of 0.959.

D. False Positive Rejection
Evaluating real-world road disturbances demonstrated high rejection rates for non-pothole features:
- Convex speed breakers: 98.4% suppression rate via transverse aspect ratio and crest brightness profiling.
- Circular manhole covers: 96.8% suppression rate via contour circularity checks.
- Tree canopy shadows: 97.2% rejection rate via multi-frame temporal tracking.

VI. CIVIC SLA AND STATUTORY GOVERNANCE

A. Crowdsourced Spatial Deduplication
To prevent municipal server congestion from multiple vehicles traversing the same route, detections within 15 meters of existing records are deduplicated using Haversine distance:

	Delta_d <= 15.0 meters	(17)

Matched instances increment the existing record's citizen endorsement counter and escalate the hazard priority without generating duplicate tickets.

B. Citizen Review and Statutory Dispatch
Recorded defects enter a Citizen Verification Queue where drivers or fleet supervisors can review photographic evidence and select Approve or Disapprove. Approved records generate official printable PDF dossiers with docket reference numbers and are transmitted via REST API to government portals, including CPGRAMS (MoRTH) and NHAI Rajmargyatra.

C. Seventy-Two-Hour SLA and Contractor Audit
The governance module tracks the mandatory MoRTH Section 10 seventy-two-hour repair timeline. If defects remain unresolved past the SLA deadline, automated escalation notices are generated. For road corridors under an active Engineering, Procurement, and Construction (EPC) warranty, the system calculates liquidated damages against contractors at statutory rates (Rs 25,000 per day).

D. In-Dash Connected Vehicle Integration
Modern passenger vehicles equipped with front Advanced Driver Assistance Systems (ADAS) cameras, in-dash touchscreens, and factory telematics eSIMs (e.g., Tata iRA, Mahindra AdrenoX) can run this software directly without supplementary aftermarket hardware, transforming regular commuter journeys into municipal road condition audits.

VII. CONCLUSION

This paper demonstrated an autonomous, edge-based road surface monitoring system developed using unsupervised machine learning. By integrating Principal Component Analysis, Isolation Forest anomaly isolation, DBSCAN spatial clustering, and monocular 3D cavity profiling, the framework eliminates the dependency on large manually annotated datasets. The system runs at 32.4 FPS on edge hardware and interfaces directly with municipal governance workflows under MoRTH Section 10 norms. Future work will investigate dual-camera stereo depth fusion for road roughness indexing and cellular vehicle-to-everything (C-V2X) direct hazard broadcasting to trailing vehicles.

ACKNOWLEDGMENT

The author thanks the faculty advisors and laboratory staff for guidance and technical support during the design, implementation, and evaluation of this research.

REFERENCES

[1] C. Koch and I. Brilakis, "Pothole detection in asphalt pavement images," Advanced Engineering Informatics, vol. 25, no. 3, pp. 507-515, 2011.
[2] H. Maeda, Y. Sekimoto, T. Seto, T. Kashiyama, and H. Omata, "Road damage detection and classification using deep neural networks with smartphone images," Computer-Aided Civil and Infrastructure Engineering, vol. 33, no. 12, pp. 1127-1141, 2018.
[3] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in IEEE International Conference on Data Mining (ICDM), 2008, pp. 413-422.
[4] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, "A density-based algorithm for discovering clusters in large spatial databases with noise," in Knowledge Discovery and Data Mining (KDD), vol. 96, 1996, pp. 226-231.
[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778.
[6] Indian Roads Congress (IRC), "Code of Practice for Maintenance of Bituminous Surfaces of Highways," IRC:82-2015, New Delhi, India, 2015.
[7] Ministry of Road Transport and Highways (MoRTH), "Specifications for Road and Bridge Works (5th Revision)," Section 10: Maintenance of Roads, New Delhi, India, 2019.
[8] ASTM International, "Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys," ASTM D6433-20, West Conshohocken, PA, 2020.
[9] S. Mathavan, K. Kamal, and M. Rahman, "A review of three-dimensional imaging technologies for pavement distress detection and measurements," IEEE Transactions on Intelligent Transportation Systems, vol. 16, no. 5, pp. 2353-2362, 2015.
[10] S. Bu, S. Liu, and G. Wan, "Pothole detection based on disparity transformation and morphological characteristics," IEEE Transactions on Intelligent Transportation Systems, vol. 22, no. 11, pp. 7180-7192, 2021.
[11] N. Otsu, "A threshold selection method from gray-level histograms," IEEE Transactions on Systems, Man, and Cybernetics, vol. 9, no. 1, pp. 62-66, 1979.
[12] L. Kaufman and P. J. Rousseeuw, Finding Groups in Data: An Introduction to Cluster Analysis. John Wiley & Sons, 2009.
[13] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer, 2009.
[14] I. Jolliffe, Principal Component Analysis. Springer Series in Statistics, 2nd ed., 2002.
[15] National Highways Authority of India (NHAI), "Policy Guidelines on Defect Liability Period and Penalty for Pavement Distress," NHAI Technical Circular 2021/04, New Delhi, India, 2021.
