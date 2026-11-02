# Rail-VIVID Dataset Benchmarking and Feasibility Audit

**Research Title:** Cross-Run Consistent Multimodal Anomaly Detection with Spatially Held-Out Evaluation on Rail-VIVID  
**Auditor:** Senior Machine Learning Research Engineer  
**Companion Publication:** Ma, S., Flanigan, K. A., Bergés, M., & Brooks, J. D. (2026). *A Multimodal Railway Vibration and Vision Dataset (Rail-VIVID)*. Nature Scientific Data 13:1343. DOI: [10.1038/s41597-026-07555-y](https://doi.org/10.1038/s41597-026-07555-y)  
**Hardware Environment:** CPU-only (16 GB RAM, 1.1 TB SSD)  

---

## 1. Project Overview

This repository establishes a reproducible benchmark and feasibility audit of the **Rail-VIVID** dataset for railway structural health monitoring, time-series anomaly detection, and spatial generalization. 

### Key Audit Findings:
1. **Zero CSV Missingness & Perfect Spatial Repeatability:** The 2,000 Hz vibration signals are clean and free from dropouts. GNSS trajectories across all 20 transits repeat with $<0.8\text{ m}$ lateral deviation.
2. **Defect Sample Size Constraint ($N=9$):** The dataset contains only **9 physical track irregularities** (7 rail joint gaps and 2 switches) across a single 1.4-km line.
3. **The "GPS Trap" Proved Empirically:** Models provided with GPS coordinates collapse to **0% recall** when tested on unseen spatial sectors, proving severe coordinate memorization. Vibration-only models achieve **$F_1 = 0.58$** on held-out spatial defects.
4. **Camera Drift Limitation:** Disk-write bottlenecks introduced up to $\pm 15\text{ m}$ of timestamp displacement in video frames, rendering microsecond video-vibration late fusion invalid without spatial landmark realignment.
5. **Undergraduate Project Feasibility:** **Highly feasible** if focused on vibration dynamics, 1D CNNs, and spatial holdout evaluation. Video transformers and 3D point-cloud deep nets are infeasible on CPU.

---

## 2. Directory Structure

```
rail-vivid-project/
├── README.md                          # Project documentation and reproduction guide
├── requirements.txt                   # Tested Python dependencies
├── configs/
│   └── baseline_config.json          # Experiment and splitting hyperparameters
├── data/
│   ├── README.md                      # Data inventory and anomaly descriptions
│   ├── ORIGINAL_DATASET_README.md     # Upstream dataset archive documentation
│   └── cached_window_features.csv     # Extracted 18,775 window features
├── notebooks/
│   └── 01_dataset_exploration.ipynb   # Interactive EDA and signal audit notebook
├── src/
│   ├── data/
│   │   └── loader.py                  # Streaming CSV and lazy frame loading
│   ├── features/
│   │   └── extractor.py               # Time/frequency domain feature pipeline
│   ├── models/
│   │   ├── statistical_baselines.py   # Isolation Forest, Random Forest, Logistic Reg.
│   │   ├── autoencoder.py             # PyTorch signal reconstruction autoencoder
│   │   └── cnn1d.py                   # PyTorch 1D temporal convolutional network
│   └── evaluation/
│       └── splitters.py               # Leakage-resistant train/val/test splitters
├── tests/
│   ├── test_splitters.py              # Unit tests for evaluation protocols
│   ├── test_features.py               # Unit tests for feature extraction
│   └── test_models.py                 # Unit tests for model inference
├── figures/                           # High-resolution audit plots (PNG)
│   ├── 01_gps_trajectories_and_anomalies.png
│   ├── 02_acceleration_timeseries_channels.png
│   ├── 03_acceleration_distributions.png
│   ├── 04_psd_spectrogram.png
│   ├── 05_environmental_distributions.png
│   ├── 06_cross_run_consistency_and_speed.png
│   └── 07_camera_frame_rate_and_sharpness.png
├── reports/
│   ├── dataset_inventory.md           # Full run, sensor, and channel inventory
│   ├── dataset_inventory.json         # Machine-readable inventory metadata
│   ├── environment_report.md          # Hardware, CPU, RAM, and toolchain audit
│   ├── ground_truth_audit.md          # Spatial ground truth and synchronization audit
│   ├── data_quality_report.md         # EDA, distributions, and confounding factors
│   ├── evaluation_protocol.md         # Leakage-resistant splitting methodologies
│   ├── baseline_results.md            # Benchmark comparison table across 4 splits
│   └── feasibility_assessment.md      # Answers to 10 core research questions
└── outputs/                           # Directory for model checkpoints and logs
```

---

## 3. Installation and Setup

Ensure Python 3.10+ (tested on Python 3.13 64-bit Windows) is installed.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests to verify splitters and models
python -m pytest tests/
```

---

## 4. Reproducing the Benchmark Experiments

All figures, feature tables, and benchmark evaluation tables can be reproduced with single commands:

```bash
# 1. Generate all 7 publication-quality EDA figures
python generate_eda_figures.py

# 2. Run the complete benchmarking suite (extracts features, fits models across 4 splits)
python run_benchmarks.py
```

---

## 5. Summary Benchmark Results

| Model | Modality | Split Strategy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Vibration + GPS | Random-Window (Leaky) | 0.4765 | 0.8354 | 0.6069 | 0.9579 | 0.6010 |
| **Random Forest** | Vibration Only | Random-Window (Leaky) | 0.4362 | 0.8025 | 0.5652 | 0.9431 | 0.4730 |
| **Random Forest** | Vibration Only | **Run-Held-Out** | 0.3790 | 0.5924 | **0.4623** | 0.9036 | 0.3736 |
| **Random Forest** | Vibration Only | **Spatial-Held-Out** | 0.5278 | 0.6348 | **0.5764** | 0.9156 | 0.5496 |
| **Random Forest** | Vibration + GPS | **Spatial-Held-Out** | **0.0000** | **0.0000** | **0.0000** | 0.9170 | 0.5182 |
| **Random Forest** | Vibration Only | **Combined Run & Spatial** | 0.3492 | 0.2716 | **0.3056** | 0.8862 | 0.3474 |
| **Isolation Forest** | Vibration Only | Combined Run & Spatial | 0.1068 | 0.1358 | 0.1196 | 0.5836 | 0.1228 |
| **Signal Autoencoder** | Vibration Only | Combined Run & Spatial | 0.1359 | 0.1728 | 0.1522 | 0.6151 | 0.1097 |
| *ViT VAE (Ma et al.)* | Vision Only | Image Reconstruction | N/A | N/A | N/A | **0.7300** | N/A |

See `reports/baseline_results.md` for complete details.
