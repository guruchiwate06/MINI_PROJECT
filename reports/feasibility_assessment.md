# Research Contribution and Feasibility Assessment Report

**Project Title:** Cross-Run Consistent Multimodal Anomaly Detection with Spatially Held-Out Evaluation on Rail-VIVID  
**Target Scope:** Third-Year Undergraduate Mini-Project / Applied Machine Learning Research  
**Author:** Senior Machine Learning Research Engineer  
**Date:** November 2026  

---

## 1. Executive Feasibility Verdict

| Research Dimension | Verdict & Status | Key Bottlenecks & Practical Constraints |
| :--- | :---: | :--- |
| **Vibration Cross-Run Consistency** | **Supported by Available Data** | Excellent 2 kHz DAQ quality, zero NaNs, $<0.8\text{ m}$ spatial alignment across all 20 runs. |
| **Spatially Held-Out Evaluation** | **Feasible with Important Limitations** | Feasible for the 9 anomalies ($N=9$), but test set contains only 3 physical joint sites. |
| **Fine-Grained Multimodal Fusion** | **Requires Additional Data / Reformulation** | Camera timestamp drift ($\pm 15\text{ m}$) precludes microsecond late fusion with vibration spikes. |
| **Broad Defect Class Generalization** | **Not Currently Testable** | Only rail joint gaps (2-3 cm) and switches exist. Broken rails, squats, and cracks are absent. |
| **Undergraduate Mini-Project Execution** | **Highly Feasible (with Narrowed Scope)** | CPU-friendly when focused on vibration time series and 1D CNNs; infeasible for heavy vision/3D. |

---

## 2. Answers to the Ten Core Research Feasibility Questions

### 1. Can repeated runs be aligned to common physical track locations with adequate accuracy?
* **Status:** **Supported by the available data.**
* **Evidence:** The onboard Columbus P-7 Pro GNSS receiver (upsampled to 2 kHz) yields a maximum cross-run lateral track discrepancy of $< 0.8\text{ meters}$. Across all 20 runs, the minimum distance to each of the 9 documented anomalies is strictly $\le 0.8\text{ m}$. Physical vibration energy peaks align to identical track coordinates across all speeds and directions (see `figures/06_cross_run_consistency_and_speed.png`).

### 2. Are anomaly labels sufficiently precise and numerous for meaningful evaluation?
* **Status:** **Feasible with important limitations.**
* **Evidence:** Coordinates and physical sizes are precisely documented in Table 2 of the companion *Nature Scientific Data* paper (`s41597-026-07555-y.pdf`). However, there are **only 9 distinct physical defect locations** on the entire 1.4-km track. 7 are mechanical rail joints (gaps between 2.0 and 3.0 cm) and 2 are track switches (turnouts). The effective statistical sample size of independent anomalies is $N=9$, not the number of sliding windows ($N \approx 18,000$).

### 3. Is there sufficient overlap between runs to measure cross-run consistency?
* **Status:** **Supported by the available data.**
* **Evidence:** There is **100% spatial overlap**. All 20 experimental runs traverse the identical 1.25 km track segment (8 uphill transits at 20.0 and 22.5 km/h, and 12 downhill transits at 22.5, 25.0, and 27.5 km/h). This enables direct measurement of repeated sensor responses at the exact same physical coordinates under varying speeds, directions, and ambient temperatures.

### 4. Is spatially held-out evaluation possible without making the training set unusable?
* **Status:** **Feasible with important limitations.**
* **Evidence:** Partitioning the track into a northern sector (Anomalies 1–5 for training), a transition sector (Anomaly 6 for validation), and a southern sector (Anomalies 7–9 for testing) produces viable splits ($7,065$ train windows and $5,427$ test windows). However, the test set is evaluated on **only 3 physical defect locations** (Anomalies 7, 8, and 9: all rail joints). Class imbalance must be handled carefully, and results should be reported with cluster-level confidence intervals.

### 5. Can multimodal fusion be evaluated fairly given camera synchronization uncertainty?
* **Status:** **Requires additional data / Reformulation.**
* **Evidence:** The camera logging pipeline in `Acquisition.py` suffered from disk I/O bottlenecks, causing frame timestamps to record file-save completion rather than shutter exposure. This introduced a variable spatial drift of up to **$15\text{ meters}$ ($\pm 2.0\text{ seconds}$)** between images and track coordinates. Directly concatenating raw video frames with 2 kHz vibration peaks produces severe spatial misregistration. Multimodal fusion is only fair if:
  1. Visual evaluation is framed as self-supervised track reconstruction (as in Ma et al.), or
  2. Optical cone detections are used as spatial anchors to realign frames before fusion.

### 6. Can we distinguish genuine anomaly detection from memorization of run identity, speed, or location?
* **Status:** **Feasible with important limitations.**
* **Evidence:** In our benchmark experiments (see `reports/baseline_results.md`):
  * When GPS coordinates were provided to a Random Forest classifier, it achieved an apparent ROC-AUC of $0.924$ on unseen runs, but its recall collapsed to **$0.0000$** on unseen track locations!
  * Conversely, when restricted to pure vibration features (RMS, kurtosis, crest factor, spectral bands), the model achieved **$\text{Recall} = 0.635$** and **$\text{F1} = 0.576$** on held-out spatial locations.
  * This proves that stripping geographic coordinates and environmental variables is mandatory to force the model to learn physical defect dynamics.

### 7. Which proposed claims are directly testable, and which require additional data?
* **Directly Testable Claims:**
  * Transferability of vibration anomaly signatures across different vehicle operational speeds ($20.0 - 27.5\text{ km/h}$).
  * Spatial generalization between mechanical rail joints at different locations on the same track line.
  * Unsupervised vibration reconstruction vs supervised classification.
* **Claims Requiring Additional Data:**
  * Detection of broken rails, squats, or rail corrugation (absent from Rail-VIVID).
  * Generalization to unseen railway track types (cWR continuous welded rail vs jointed rail) or different rolling stock (freight cars, high-speed bogies).

### 8. What are the strongest simple baselines against which the final method should be compared?
1. **Statistical Feature Extractor + Random Forest / Gradient Boosting:** Highly competitive on tabular vibration features (achieves F1 of $0.576$ on spatial holdouts).
2. **Lightweight 1D Temporal CNN:** Processes raw 6-channel acceleration waveforms without manual feature engineering.
3. **Signal Reconstruction Autoencoder (AE / VAE):** Standard unsupervised structural health monitoring baseline.

### 9. What experimental results would falsify our central hypothesis?
* **Falsification Condition 1:** If a proposed cross-run model fails to outperform a naive threshold on the bogie z-axis acceleration standard deviation under the **Combined Run-and-Spatial Split**.
* **Falsification Condition 2:** If the addition of camera imagery fails to improve detection precision/recall over vibration-only models when evaluated under spatial tolerance buffers that account for the 15-meter camera drift.

### 10. Is this feasible as a third-year undergraduate mini-project given available hardware?
* **Status:** **Highly Feasible with Narrowed Scope.**
* **Compute Reality:** The development environment is **CPU-only (16 GB RAM, no CUDA GPU)**.
* **Recommendations for Student Success:**
  * **Abandon:** Training dense Vision Transformers (ViT) from scratch, end-to-end multimodal video-audio networks, and 3D point cloud deep nets on raw FARO `.fls` files.
  * **Focus On:** Multichannel 1D CNNs, statistical feature engineering, spatial vs run-held-out evaluation splits, and vibration sensor placement comparison (bogie vs axle vs body). This scope aligns perfectly with undergraduate capabilities and runs completely in $< 5\text{ minutes}$ on a laptop CPU.

---

## 3. Explicit Research Scope Recommendations

```
Original Proposed Title:
"Cross-Run Consistent Multimodal Anomaly Detection with Spatially Held-Out Evaluation on Rail-VIVID"

Recommended Narrowed Title:
"Cross-Run Consistent Vibration Anomaly Detection and Spatial Generalization on the Rail-VIVID Benchmark"
```

### Strategic Adjustments:
1. **Focus on Vibration Modalities:** Leverage all 6 calibrated accelerometer channels (bogie, axle, car body). Sensor placement comparison provides rich mechanical engineering insight.
2. **Treat Vision as a Downstream Verification Step:** Use camera images to visually verify detected anomalies (or cite published Ma et al. vision reconstruction scores) rather than building an end-to-end video fusion network on CPU.
3. **Strict Evaluation Protocol:** Mandate the **Combined Run-and-Spatial Split** (`CombinedRunAndSpatialSplitter`) to prevent accidental GPS coordinate memorization.

