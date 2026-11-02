# Leakage-Resistant Evaluation Protocol

**Project:** Rail-VIVID Dataset Benchmarking and Feasibility Audit  
**Date:** November 2026  
**Auditor:** Senior Machine Learning Research Engineer  
**Implementation:** `src/evaluation/splitters.py`, `tests/test_splitters.py`  

---

## 1. The Data Leakage Threat Model in Railway Infrastructure Monitoring

In time-series and sensor-based structural health monitoring, naive machine learning evaluation practices routinely produce catastrophically inflated, non-generalizable benchmark figures. This protocol identifies and eliminates four primary leakage vectors:

### 1.1 Temporal Autocorrelation & Window Overlap Leakage
When continuous 2,000 Hz vibration signals are sliced into sliding windows with overlap (e.g. 50% overlap), adjacent windows share raw sensor points. Randomly partitioning windows into training and testing sets places nearly identical samples in both sets, enabling the model to achieve near-perfect performance by memorizing high-frequency sensor noise.

### 1.2 Track Geometry & Physical Location Memorization
All 20 runs in Rail-VIVID traverse the exact same 1.25 km physical segment. Every run encounters the identical track curvature, welded joints, sleeper spacing, and track bed stiffness. If a model is trained on Run 1 and tested on Run 2 without spatial partitioning, the test set presents the *exact same physical anomalies* at the *exact same track coordinates*. The model may simply learn to recognize the spatial signature of Orbisonia Track Milepost 0.8 rather than learning what constitutes a rail defect.

### 1.3 Post-Impact Transient Ringing (Boundary Smearing)
When a train wheel crosses a 2 cm rail joint, the physical impact excites damped structural resonance across the vehicle suspension that rings for 0.2 to 1.0 seconds ($1.5 - 7.5\text{ m}$ of track distance). If a time window immediately following the joint is labeled "Normal", the classifier is penalized for detecting genuine defect dynamics, or if included in training, causes label pollution.

### 1.4 Preprocessing and Normalization Information Leakage
Fitting `StandardScaler`, min-max scalers, PCA projections, or frequency filters on the entire dataset prior to splitting leaks mean and variance statistics from the test set into the model. All preprocessing pipelines must strictly fit parameters on training folds only.

---

## 2. Evaluation Split Strategies

To provide complete scientific rigor, four evaluation strategies are implemented in `src/evaluation/splitters.py`:

```
Strategy A: Random-Window Split (Optimistic Baseline / Control)
  [ Train Windows ] <--- Overlap Leakage ---> [ Test Windows ]
  * Result: Artificially inflated AUC (0.95+). Serves solely as a negative control.

Strategy B: Run-Held-Out Split (Cross-Run Generalization)
  [ Train: Runs 1-13 ] ----> [ Val: Runs 14-16 ] ----> [ Test: Runs 17-20 ]
  * Evaluates temporal and environmental robustness across speeds and days.
  * Limitation: Physical defect locations are identical between train and test.

Strategy C: Spatially Held-Out Split (Spatial Generalization)
  [ Train: Anomalies 1-5 ] ----> [ Val: Anomaly 6 ] ----> [ Test: Anomalies 7-9 ]
  * Physically isolates northern vs southern track sectors.
  * Evaluates whether the model detects novel defects at unseen track locations.

Strategy D: Combined Run-and-Spatial Holdout (Zero-Shot Dual Holdout)
  [ Train: Runs 1-13 @ Sites 1-5 ] ----> [ Test: Runs 17-20 @ Sites 7-9 ]
  * Evaluates simultaneous cross-run and spatial generalization.
  * Strictest and most realistic deployment test.
```

---

## 3. Boundary Buffer Zone and Spatial Quarantining

To eliminate label noise and transient ringing leakage, a three-zone spatial classification model is enforced:

$$\text{Zone}(d) = \begin{cases}
1 \text{ (Definite Anomaly)}, & d \le R_{\text{anomaly}} \\
-1 \text{ (Quarantined Buffer)}, & R_{\text{anomaly}} < d \le R_{\text{buffer}} \\
0 \text{ (Conservative Normal)}, & d > R_{\text{buffer}}
\end{cases}$$

Where:
* $d$ is the minimum great-circle (Haversine) distance from the window center to the nearest documented ground-truth anomaly coordinate.
* **$R_{\text{anomaly}} = 6.0\text{ meters}$:** Captures the physical wheelset impact and primary bogie response window.
* **$R_{\text{buffer}} = 18.0\text{ meters}$:** Quarantines post-impact vehicle chassis ringing and GNSS positioning offsets. Windows in this zone are completely excluded from both training and testing.
* **Normal Track Reservation:** Only windows where $d > 18.0\text{ meters}$ from *all* 9 anomalies are admitted as class $0$ (Normal).

---

## 4. Feasibility Constraints Imposed by the Dataset

The dataset presents fundamental statistical constraints:
1. **Total Number of Anomaly Clusters ($K=9$):**
   * Training on 5 anomaly locations, validating on 1, and testing on 3 leaves the test set with exactly **3 physical defect locations** (Anomalies 7, 8, and 9: all rail joint gaps).
   * Any metric computed on the test set is evaluated on repeated passes over just 3 physical points.
2. **Defect Type Generalization is Impossible:**
   * Because the test set contains only rail joints (or switch frogs if held out), we cannot evaluate whether a model trained on joints detects broken rails or squats. Rail-VIVID contains zero broken rail instances.
3. **Resampling and Statistical Significance:**
   * Bootstrap confidence intervals must resample at the **cluster level** (Run level or Spatial Anomaly level), not at the individual window level. Treating 2,000 adjacent windows as independent samples violates the i.i.d. assumption and artificially deflates confidence interval widths by a factor of $\sqrt{2000} \approx 45$.

