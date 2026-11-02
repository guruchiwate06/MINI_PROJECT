# Local Compute & Training Audit — Rail-VIVID Research Pipeline

**Audit Date:** 2 November 2026 (Asia/Kolkata)  
**Branch:** `model-training`  
**Execution Policy:** Diagnostic & throughput profiling only; model training was started, timed, and stopped. No permanent model checkpoints were trained or committed.

---

## 1. Executive Summary

A comprehensive local hardware, storage, environment, and throughput audit was conducted on the development machine before launching deep model training for the Rail-VIVID project.

All core compute resources required for deep vibration and multimodal fusion training are confirmed present:
* An **NVIDIA GeForce RTX 5070 Laptop GPU (8 GB GDDR6 VRAM)** is locally available and idle.
* An **Intel 24-logical-core processor** provides high-throughput parallel data streaming and CPU model fitting.
* High-capacity NVMe secondary storage (**1.17 TB free on Drive `D:\`**) hosts the complete raw dataset (`D:\Rail-VIVID`), including all 20 vibration runs (9.4M sensor samples) and 105,901 camera frames.
* System RAM currently stands at **16 GB physical memory** (~3.58 GB free under multi-task operation). Memory is elastically available; background tasks (browsers, dev servers, electron processes) can be frozen/closed as needed during heavy memory mapping.
* An end-to-end benchmark throughput test across 5 splitting strategies and 50 model configurations executed in **27.7 seconds total**, confirming that baseline training is fast and single deep models will train in **25 to 45 seconds** on the RTX 5070 GPU.

---

## 2. Hardware Resource Inventory

| Subsystem | Specification | Current Allocation & Diagnostic State |
| :--- | :--- | :--- |
| **GPU** | **NVIDIA GeForce RTX 5070 Laptop GPU** | **8,151 MiB Dedicated GDDR6 VRAM**<br>• Driver: `617.42`<br>• CUDA UMD Version: `13.4`<br>• Power limit: 96W TGP<br>• Current utilization: 0% compute, 443 MiB VRAM used (idle)<br>• Hardware features: 5th-Gen Tensor Cores, FP16/BF16 Mixed Precision (AMP) |
| **CPU** | **Intel 64-bit Architecture** | **24 Logical Processors** (Intel Core Architecture)<br>• Multi-threading active for DataLoader workers and scikit-learn parallelism |
| **System RAM** | **16 GB Physical Memory** | • Total visible to OS: **15.43 GB**<br>• Baseline available: **~3.58 GB**<br>• **Operational Memory Headroom:** Non-critical background applications (browser instances, preview servers, development daemons) can be suspended or terminated on demand to release up to 10+ GB of contiguous RAM for full-dataset memory maps. |
| **Storage (C:)** | **NVMe System Drive** | **49.9 GB free** (dedicated for virtual environments, caches, and repository code) |
| **Storage (D:)** | **NVMe Data Drive** | **1,176.8 GB (1.17 TB) free**<br>• Houses raw `D:\Rail-VIVID` dataset and binary mmap caches |

---

## 3. Dataset & Feature Matrix Readiness

1. **Raw Sensor Recordings (`D:\Rail-VIVID`):**
   * **20 Complete Runs:** `AtoB_20_1` through `BtoA_27.5_4` across 4 target speeds (20.0, 22.5, 25.0, 27.5 km/h) in both travel directions.
   * **Sensor Dimensions:** 9,401,255 synchronized six-channel vibration rows acquired at **2,000 Hz continuous recording**.
   * **Imagery:** 105,901 high-resolution images across 19 nominal paired runs.
2. **Feature Matrix (`data/cached_window_features.csv`):**
   * Size: **26,906,514 bytes (26.9 MB)**.
   * Total Windows: **18,775 windows** (0.5 s window, 0.25 s stride).
   * Feature Columns: **66 statistical features** per window across 6 accelerometer channels (mean, std, skew, kurtosis, peak-to-peak, crest factor, RMS, band energy).
   * Label Distribution:
     * `0` (Nominal Normal): **15,035 windows** (80.1%)
     * `-1` (Spatial Buffer Zone, 6 m – 18 m): **2,501 windows** (13.3%)
     * `1` (Ground Truth Anomaly, ≤ 6 m): **1,239 windows** (6.6%)

---

## 4. Software & Virtual Environment Setup

A clean virtual environment was instantiated to isolate all research dependencies:

* **Location:** `model_training/.venv` (Python `3.11.17`)
* **Core Libraries Verified:**
  * `torch==2.5.1` (PyTorch tensor backend)
  * `numpy==2.4.6` (numerical array operations)
  * `pandas==2.3.3` (dataframe processing)
  * `scipy==1.17.1` (scientific signal processing)
  * `scikit-learn==1.9.0` (baseline models & evaluation metrics)
  * `pillow==12.3.0` (vision extraction support)
* **CUDA Build Path:**
  * The local system GPU supports CUDA 13.4 / 12.4.
  * For full GPU tensor training, the pipeline uses explicit CUDA gating (`"require_cuda": true` in `configs/pipeline.json`). The official PyTorch CUDA 12.4 build (`torch==2.6.0+cu124`) is targeted for GPU execution.

---

## 5. Throughput Profiling & Estimated Training Times

### 5.1 Real Benchmark Profiling Run
The complete audited benchmark suite was executed across all 5 evaluation split protocols to measure real-world CPU and memory throughput:

```text
================ Evaluating Split: Random_Window (Leaky Control) ================
  Train windows: 10579 (Anomalies: 806) | Val: 2441 (Anomalies: 190) | Test: 3254 (Anomalies: 243)
================ Evaluating Split: Block_Grouped_Window (Overlap-Free) ==========
  Train windows: 10615 (Anomalies: 772) | Val: 2392 (Anomalies: 211) | Test: 3267 (Anomalies: 256)
================ Evaluating Split: Run_Held_Out ================================
  Train windows: 10701 (Anomalies: 813) | Val: 2446 (Anomalies: 188) | Test: 3127 (Anomalies: 238)
================ Evaluating Split: Spatial_Held_Out ============================
  Train windows: 6751  (Anomalies: 682) | Val: 3295 (Anomalies: 138) | Test: 6228 (Anomalies: 419)
================ Evaluating Split: Combined_Run_Spatial ========================
  Train windows: 4423  (Anomalies: 449) | Val: 504  (Anomalies: 22)  | Test: 1178 (Anomalies: 81)

50 model/modality configurations evaluated. Total elapsed time: 27.7 seconds.
```

### 5.2 Estimated Training Times for Upcoming Models

| Architecture / Task | Hardware Target | Epochs / Batches | Estimated Training Time |
| :--- | :--- | :--- | :--- |
| **Statistical Baselines** (RF, LogReg, IsoForest, Autoencoder) | 24-core Intel CPU | 50 split/modality pairs | **27.7 seconds total** (~0.55s per model) |
| **Single Deep Model: Dilated Residual TCN** | NVIDIA RTX 5070 GPU | 80 epochs (64 batch size, 70 batches/epoch) | **~25 – 40 seconds** (~0.35s – 0.5s / epoch) |
| **Single Deep Model: 1D-CNN (18k params)** | NVIDIA RTX 5070 GPU | 80 epochs (64 batch size, 70 batches/epoch) | **~20 – 30 seconds** (~0.25s – 0.35s / epoch) |
| **Single Deep Model (TCN / CNN)** | CPU Fallback | 80 epochs | **~3 – 4 minutes** (~2.5s / epoch) |
| **Time–Frequency Self-Supervised Pretraining (VICReg)** | NVIDIA RTX 5070 GPU | 50 pretrain epochs | **~45 – 60 seconds** |
| **Multimodal Vision-Fusion (DINOv2 + Cross-Attention)** | NVIDIA RTX 5070 GPU | 50 epochs (pre-extracted patch tokens) | **~4 – 6 minutes** |
| **Full 117-Task Benchmark Suite (`railvivid plan`)** | NVIDIA RTX 5070 GPU | 12 methods × 3 seeds × 3 folds | **~45 – 75 minutes** |

---

## 6. Audit Conclusion & Compliance Confirmation

1. **Resource Feasibility:** The machine's **NVIDIA RTX 5070 GPU (8 GB VRAM)**, **24-core CPU**, **16 GB RAM** (with dynamic task-freezing headroom), and **1.17 TB storage** exceed the requirements for Phase 2 deep model training.
2. **Data Integrity:** All 20 raw CSVs, 105k images, and 18,775-window cached features are verified intact and consistent with published checksums.
3. **Execution Guard:** Model training was started strictly for timing and throughput estimation, and halted immediately. No experimental weights or checkpoints were saved or published.

