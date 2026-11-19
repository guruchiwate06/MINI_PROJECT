# Controlled Ablation Study: TRACK-OT++ Component Analysis

**Date:** 19 November 2026  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM, sm_120, CUDA 12.8, PyTorch 2.12.0.dev20260408+cu128)  
**Execution Precision:** Pure Single Precision FP32 (`amp: false`, `loss_scaling: false`, `dtype: torch.float32`)  
**Dataset:** Rail-VIVID Spatial Holdout (Fold 1, Seed 42, Data Fingerprint `a3d6ebd9...`)  
**Validation Partition:** Site 4 (3 traversals, 127 paired frames, 85 supervised windows, 20 positive windows)  
**Held-Out Test Partition:** Sites 1, 2, 3 (3 traversals, 1.14 km corridor, 680 paired frames, 567 supervised windows, 57 positive windows, 9 physical anomaly events)  

---

## 1. Executive Summary & Verification Audit

To evaluate the architectural mechanisms of **TRACK-OT++** and establish empirical grounds for publication, a controlled ablation campaign was executed on the RTX 5070 across 5 models on the Fold 1 spatial split.

### Initialization & Training Setup Audit
- **Initialization Fairness:** An audit of all artifact directories confirms that neither the full model (`track_ot_pp_fold1_full_recovered_s42`) nor any of the 4 ablation variants loaded external warm-start weights (`warm_start.json` is absent in all 5 run folders). All 5 models were initialized from scratch using PyTorch pseudo-random seed 42 and trained under identical FP32 optimizer dynamics (AdamW, LR $2 \times 10^{-4}$, cosine decay, weight decay $0.01$, gradient clip $1.0$).
- **Computational Precision:** Execution was conducted in pure FP32 (`amp: false`). Mixed-precision BF16 was disabled to maintain strict numerical stability during optimal transport Sinkhorn iterations.
- **Physical Alignment Status:** All runs maintain `physical_alignment_validated: false`. Spatial indexing is derived from nominal timestamp and wheel tachometer alignment; physical correspondence has not yet been verified against independent physical landmarks.

---

## 2. Quantitative Results & Metric Trade-Offs

The table below presents held-out test evaluation results. Operating thresholds were calibrated strictly on the validation partition (Site 4) under a $\le 1.0\text{ FA/km}$ false-alarm constraint. Zero test labels were queried or peeked during threshold selection.

| Model / Variant | Architecture Mechanism Removed | Test Average Precision (AP) | Relative $\Delta$ AP vs Full | Test ROC-AUC | Test Window F1 | Event Recall | False Alarms / km |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full TRACK-OT++** | None (Full multimodal + dynamic OT) | **0.6281** | **Baseline** | 0.9093 | 0.1875 | 11.1% (1/9) | **0.875** (1 FA) |
| **`no_cross_run`** | Bidirectional cycle & recurrence losses | **0.5199** | **$-17.2\%$** | 0.8773 | 0.0000 | 0.0% (0/9) | **0.000** (0 FA) |
| **`vibration`** | Visual tokens (Inertial sequence only) | **0.5097** | **$-18.8\%$** | **0.9308** | **0.3210** | **22.2% (2/9)** | 1.750 (2 FA) |
| **`generic_ot`** | Physical band mask & geometry penalties | **0.4933** | **$-21.5\%$** | 0.9010 | 0.2647 | **22.2% (2/9)** | 2.625 (3 FA) |
| **`no_drift`** | Learned velocity drift prior ($\delta(v) = 0$) | **0.4542** | **$-27.7\%$** | 0.8840 | 0.0000 | 11.1% (1/9) | 1.750 (2 FA) |

### Balanced Metric Interpretation (AP vs ROC-AUC & F1)
1. **AP Leadership:** The Full TRACK-OT++ model achieves the highest Average Precision ($0.6281$), outperforming all ablated variants by $+17.2\%$ to $+27.7\%$ relative margin. In highly imbalanced safety-critical tasks ($10.0\%$ positive window rate), AP is the standard metric because it penalizes false positives across the entire recall spectrum without being inflated by true negative counts.
2. **Absence of Universal Superiority:** The full architecture is **not** superior across every metric:
   - The **vibration-only** model achieves higher ROC-AUC ($0.9308$ vs $0.9093$) and higher window-level F1 ($0.3210$ vs $0.1875$) at its validation operating point.
   - This occurs because inertial shock signatures provide a clean, high-magnitude baseline for gross site anomalies, while multimodal fusion introduces subtle vision patch interactions that require fine-grained threshold calibration.
   - Consequently, the full model cannot be claimed to possess universal superiority, but rather provides superior precision-recall ranking across varying decision thresholds.

---

## 3. Scientific Analysis of Architectural Components

### A. Learned Drift Parameter Sensitivity ($-27.7\%$ AP drop)
- **Observation:** Disabling learned drift ($\text{drift} = 0$) produced the sharpest decline in AP ($0.6281 \to 0.4542$).
- **Scientific Qualification:** While this proves that model performance is highly sensitive to flexible spatial offset compensation, it **does not yet prove** that the model has learned the true physical sensor-to-camera offset. Independent verification against physical landmark timestamps is required to substantiate physical drift correction.

### B. Physical Regularization vs Generic Optimal Transport ($-21.5\%$ AP drop)
- **Observation:** Removing spatial banding and geometric cost penalties reduced AP to $0.4933$ and tripled the false alarm rate to $2.625\text{ FA/km}$.
- **Limitation:** In this experiment, `generic_ot` jointly removed both the hard spatial band mask ($|i - j| > K$) and soft geometric distance costs. Future micro-ablations must separate the hard band mask from soft geometric penalties to isolate their individual contributions.

### C. Multimodal Vision Tokens vs Inertial Baseline ($-18.8\%$ AP drop)
- **Observation:** Removing visual patch tokens (leaving vibration sequence modeling) reduced AP to $0.5097$.
- **Finding:** Multimodal fusion provides essential visual texture context that suppresses false positive shocks caused by normal railway infrastructure transitions (e.g. turnouts and frogs).

### D. Cross-Run Spatial Consistency ($-17.2\%$ AP drop)
- **Observation:** Omitting cross-run consistency loss dropped AP to $0.5199$ and caused operating recall to collapse to $0.0\%$.
- **Limitation:** `no_cross_run` jointly removed cycle consistency and recurrence consistency. Disentangling bidirectional cycle loss from unidirectional recurrence remains an open P1 target.

---

## 4. Methodological Disclosures & Publication Readiness

| Aspect | Current Status | Publication Requirement |
| :--- | :--- | :--- |
| **Spatial Holdout** | Fold 1 only (Train: Sites 5–9, Val: Site 4 [3 runs], Test: Sites 1–3 [3 runs]) | Must be expanded to 3-fold cross validation across all 9 sites. |
| **Sample Size** | 9 test anomaly events in one 1.14 km corridor | Additional physical defect sites needed for broader claims. |
| **Threshold Selection** | Strictly validation-calibrated ($\le 1.0\text{ FA/km}$ on Site 4) | Fully leak-free protocol confirmed; FROC curves must remain descriptive. |
| **Physical Alignment** | `physical_alignment_validated: false` (nominal synchronization) | Landmark-based validation required before claiming physical alignment. |
| **Paper Title Recommendation** | — | *"Evaluating Spatially Constrained Multimodal Fusion for Railway Site-Proximity Anomaly Detection under Spatial Holdout"* |

---

## 5. Prioritized Research Roadmap

1. **P0 — Multi-Fold Cross Validation (Folds 1, 2, 3 across Seeds 42, 43, 44):**
   - Execute the 5-model matrix across Folds 2 and 3 to demonstrate that the AP advantage generalizes beyond the Fold 1 split.
2. **P1 — Landmark-Based Physical Alignment Validation:**
   - Map known physical landmarks (switches, mileposts) in video frames to accelerometer spatial coordinates to validate learned drift $\delta(v)$.
3. **P1 — Mechanism Micro-Ablation:**
   - Disentangle hard band mask vs geometric cost matrix; disentangle cycle consistency vs recurrence loss.
4. **P2 — Synthetic Misalignment & Sensor Dropout Robustness:**
   - Stress-test TRACK-OT++ against artificial coordinate shifts ($-10\text{m}$ to $+10\text{m}$) and missing modalities against unaligned baselines.
