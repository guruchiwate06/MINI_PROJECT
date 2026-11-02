# Environment and Hardware Inspection Report

**Project:** Rail-VIVID Dataset Benchmarking and Feasibility Audit  
**Date of Audit:** November 2026  
**Auditor:** Senior Machine Learning Research Engineer  

---

## 1. System Hardware Specifications

| Component | Specification | Observed Status |
| :--- | :--- | :--- |
| **Operating System** | Microsoft Windows (x86_64 / AMD64) | Active Workspace: `d:\Rail-VIVID` |
| **CPU Architecture** | Multi-core x86_64 Architecture | Fully Operational |
| **System Memory (RAM)** | 16.0 GB Total Physical Capacity | **15.69 GB Available** |
| **Disk Storage** | Primary Workspace Drive (`D:\`) | **1,096.07 GB Free** (~1.10 TB SSD) |
| **GPU Acceleration** | Hardware Acceleration Status | **None (CUDA Not Available / CPU Only)** |

> [!WARNING]
> **Compute Bottleneck Warning (CPU-Only Environment):**  
> `torch.cuda.is_available()` returns `False`. No dedicated NVIDIA CUDA GPU is provisioned in this execution runtime. All deep learning inference and training must run on the CPU. Consequently, large vision-language backbones, multi-frame video transformers, and heavy 3D point-cloud architectures (e.g., PointNet++ on dense FARO `.fls` lidar scans) are computationally prohibitive. Modeling baselines must be lightweight, vectorized, and computationally optimized (e.g., 1D temporal CNNs, feature-based gradient boosters, Isolation Forests, and compact linear/autoencoder projections).

---

## 2. Python Environment and Toolchain

* **Python Version:** `3.13.2 (tags/v3.13.2:4f8bb39, Feb 4 2025, 15:23:48) [MSC v.1942 64 bit (AMD64)]`
* **Python Executable:** System 64-bit Python 3.13 runtime

### Core Scientific & Machine Learning Packages
The environment contains modern scientific Python libraries:

| Library | Version | Role in Project |
| :--- | :--- | :--- |
| **`torch`** | `2.14.0+cpu` | Tensor computation and lightweight 1D CNN baseline |
| **`torchvision`** | `0.29.0+cpu` | Vision data transformations and image utilities |
| **`scikit-learn`** | `1.7.2` | Classical ML baselines, Isolation Forest, ROC/PR evaluation |
| **`scipy`** | `1.18.1` | Digital signal processing (Butterworth filtering, PSD, FFT) |
| **`pandas`** | `2.2.3` | Tabular data manipulation and chunked CSV streaming |
| **`numpy`** | `2.2.3` | Vectorized numerical operations and signal feature extraction |
| **`opencv-python`** (`cv2`) | `4.11.0` | High-speed image reading, Laplacian variance sharpness, edge maps |
| **`matplotlib` / `seaborn`** | `3.10.0` / `0.13.2` | Exploratory data analysis visualization and spectral plotting |
| **`huggingface_hub`** | `1.32.0` | Hugging Face repository and artifact verification |
| **`pymupdf` (`fitz`) / `pypdf`** | Installed | Extraction and verification of paper documentation |

---

## 3. Storage and Memory Budget Analysis

1. **Local Dataset Footprint:**
   * Tabular Data (20 CSV files): **~1.73 GB** (9,401,255 total rows).
   * Vision Data (105,901 high-resolution images across 20 subfolders): **~28.4 GB**.
   * Anomaly 3D Scans (`Anomaly/Anomaly3DScanning/` FARO `.fls` files): **~1.2 GB**.
   * Total Local Storage Footprint: **~31.3 GB**.
   * Storage Margin: With **1,096 GB** free disk space, local disk capacity is abundant (utilizing < 3% of available disk space).

2. **RAM Consumption & Streaming Constraints:**
   * In-memory expansion of full 6-channel 2 kHz floating-point data across all 20 runs exceeds 4.5 GB in raw uncompressed DataFrame allocations.
   * Loading 105,901 full-resolution (1920×1080) RGB/monochrome images into memory simultaneously would require over **210 GB RAM**, which would instantly cause an Out-Of-Memory (OOM) kernel crash on this 16 GB machine.
   * **Mandatory Architecture Requirement:** All data processing pipelines must utilize chunked/streaming I/O, lazy image generators, windowed feature extraction, and disk caching.

---

## 4. Environment Feasibility Determination

| Criterion | Evaluation | Recommendation |
| :--- | :--- | :--- |
| **Tabular Vibration Processing** | **High Feasibility** | 9.4M rows at 2 kHz can be processed in sub-minute chunks with NumPy/Pandas. |
| **Signal-based ML & Anomaly Detection** | **High Feasibility** | Feature extraction (RMS, kurtosis, crest factor, PSD) + Scikit-Learn baselines run rapidly on CPU. |
| **Deep Multichannel 1D CNNs** | **Moderate Feasibility** | Compact 1D CNNs with small kernel sizes train within minutes on CPU with batch sizes of 32–64. |
| **Dense Vision-Language / 3D Point-Cloud Models** | **Infeasible on Local CPU** | Discard dense 3D point cloud end-to-end training. Use feature-extracted / masked visual sharpness / self-supervised compact representations as per paper. |

