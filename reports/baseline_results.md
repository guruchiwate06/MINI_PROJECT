# Baseline Modeling and Benchmark Evaluation Report (Audited & Leak-Free)

**Project:** Cross-Run Consistent Multimodal Anomaly Detection with Spatially Held-Out Evaluation on Rail-VIVID  
**Date:** November 2026  
**Auditor:** Senior Machine Learning Research Engineer  
**Companion Artifacts:** `reports/baseline_results.json`, `reports/evaluation_audit.md`  

---

## 1. Experimental Setup & Benchmarking Design

Experiments were executed across all 20 experimental runs ($9,401,255$ total rows), utilizing **$18,775$ extracted windows** ($0.50\text{ s}$ duration, $0.25\text{ s}$ stride):
* **Definite Anomalies ($d \le 6.0\text{ m}$):** $1,239$ windows ($6.6\%$)
* **Quarantined Transition Buffer ($6.0\text{ m} < d \le 18.0\text{ m}$):** $2,501$ windows (strictly excluded from all training and evaluation)
* **Conservative Normal Track ($d > 18.0\text{ m}$):** $15,035$ windows ($80.1\%$)
* **Total Clean Windows:** $16,274$ windows

### Threshold Calibration Rigor (Audit Guarantee):
* **Zero Test-Set Peeking:** In accordance with the evaluation audit, all anomaly and probability decision thresholds are calibrated strictly on the **validation set** (or normal training samples), never on the test set.
* **Operating Point Selection:** Supervised models tune probability thresholds on validation folds to maximize validation F1. Unsupervised models (Isolation Forest, Signal Autoencoder) calibrate thresholds to match validation density/reconstruction percentiles.

---

## 2. Audited Benchmark Comparison Table

The table below presents the verified results across all five evaluation splitting strategies:

| Model | Modalities | Split Strategy | Calibrated Threshold | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Limitations & Failure Modes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | Vibration + GPS | Random Window (Leaky Control) | 0.5000 | 0.4765 | 0.8354 | 0.6069 | 0.6010 | 0.9579 | Overlap leakage + GPS memorization; optimistic baseline. |
| **Random Forest** | Vibration Only | Random Window (Leaky Control) | 0.5500 | 0.4420 | 0.7531 | 0.5571 | 0.4730 | 0.9431 | Overlap leakage; adjacent windows share sensor points. |
| **Random Forest** | Vibration Only | **Block Grouped (Overlap-Free)** | 0.3500 | 0.3996 | 0.8242 | **0.5383** | **0.4170** | 0.9293 | Eliminates overlap leakage ($-11.8\%$ PR-AUC drop vs leaky). |
| **Random Forest** | Vibration + GPS | **Block Grouped (Overlap-Free)** | 0.4000 | 0.4539 | 0.8086 | **0.5815** | **0.4885** | 0.9447 | Grouped blocks prevent time leakage across splits. |
| **Logistic Regression**| Vibration Only | Block Grouped (Overlap-Free) | 0.6000 | 0.2812 | 0.6680 | 0.3958 | 0.2703 | 0.8457 | Linear boundary; moderate precision. |
| **Isolation Forest** | Vibration Only | Block Grouped (Overlap-Free) | -0.0523 | 0.0723 | 0.2852 | 0.1153 | 0.0764 | 0.4921 | Unsupervised density fails on multi-speed vibration. |
| **Signal Autoencoder**| Vibration Only | Block Grouped (Overlap-Free) | 0.1779 | 0.0819 | 0.2695 | 0.1257 | 0.0802 | 0.5383 | High reconstruction error on normal track sections. |
| **Random Forest** | Vibration Only | **Run-Held-Out** | 0.4500 | 0.3685 | 0.6597 | **0.4729** | 0.3736 | 0.9036 | Evaluates cross-run robustness across speeds. |
| **Random Forest** | Vibration + GPS | Run-Held-Out | 0.4000 | 0.3547 | 0.7437 | 0.4803 | 0.5009 | 0.9244 | Leaks track coordinates since test runs share identical track. |
| **Isolation Forest** | Vibration Only | Run-Held-Out | -0.0491 | 0.0714 | 0.7143 | 0.1299 | 0.0764 | 0.4781 | Val threshold $\tau = -0.049$ captures joints but has false alarms. |
| **Signal Autoencoder**| Vibration Only | Run-Held-Out | 1.9794 | 0.0938 | 0.3782 | 0.1504 | 0.1036 | 0.6127 | Reconstructs normal vibration; separates anomalies modestly. |
| **Random Forest** | Vibration Only | **Spatial-Held-Out** | 0.6750 | 0.5954 | 0.4988 | **0.5429** | **0.5229** | 0.9191 | **Best generalizing model:** learned joint impact dynamics. |
| **Random Forest** | Vibration + GPS | Spatial-Held-Out | 0.2500 | 0.5046 | 0.5251 | 0.5146 | 0.4793 | 0.9071 | Dynamic Voronoi partitioning prevents total zero-recall. |
| **Logistic Regression**| Vibration Only | Spatial-Held-Out | 0.8000 | 0.4987 | 0.4439 | 0.4697 | 0.3902 | 0.8414 | Linear baseline generalizes reasonably on bogie features. |
| **Isolation Forest** | Vibration Only | Spatial-Held-Out | -0.0581 | 0.0845 | 0.4606 | 0.1428 | 0.0946 | 0.5807 | Unsupervised baseline detects novel joints with low precision. |
| **Signal Autoencoder**| Vibration Only | Spatial-Held-Out | 0.2210 | 0.0803 | 0.3031 | 0.1269 | 0.0718 | 0.5235 | Unsupervised reconstruction error on novel track sector. |
| **Random Forest** | Vibration Only | **Combined Run & Spatial** | 0.5250 | 0.3519 | 0.2346 | **0.2815** | 0.3360 | 0.9012 | **Strictest zero-shot benchmark:** simultaneous dual holdout. |
| **Random Forest** | Vibration + Env | Combined Run & Spatial | 0.5000 | 0.3478 | 0.1975 | 0.2520 | 0.3295 | 0.8939 | Environmental features do not improve generalization. |
| **Random Forest** | Vibration + GPS | Combined Run & Spatial | 0.2000 | 0.2368 | 0.2222 | 0.2293 | 0.2254 | 0.8452 | Severe coordinate mismatch degrades PR-AUC to $0.225$. |
| **Logistic Regression**| Vibration Only | Combined Run & Spatial | 0.6750 | 0.0882 | 0.6296 | 0.1548 | 0.2475 | 0.6531 | High false alarm rate under dual holdout. |
| **Signal Autoencoder**| Vibration Only | Combined Run & Spatial | 1.7563 | 0.0943 | 0.6914 | 0.1659 | 0.0901 | 0.5988 | Unsupervised reconstruction achieves F1 of $0.166$. |
| *CNN AE (Published)* | Vision Only | Image Reconstruction | N/A | N/A | N/A | N/A | N/A | **0.6500** | 19.03M params; normal track reconstruction (Ma et al.). |
| *CNN VAE (Published)* | Vision Only | Image Reconstruction | N/A | N/A | N/A | N/A | N/A | **0.6800** | 19.16M params; probabilistic reconstruction (Ma et al.). |
| *ViT AE (Published)* | Vision Only | Image Reconstruction | N/A | N/A | N/A | N/A | N/A | **0.7200** | 15.08M params; Vision Transformer (Ma et al. Table 3). |
| *ViT VAE (Published)* | Vision Only | Image Reconstruction | N/A | N/A | N/A | N/A | N/A | **0.7300** | 15.21M params; Vision Transformer VAE (Ma et al.). |

---

## 3. Key Scientific Takeaways from the Audited Results

1. **Sliding Window Leakage Is Quantified:**
   * Random window splitting inflated PR-AUC from $0.4170$ to $0.4730$ ($+13.4\%$) because overlapping windows share 50% raw vibration data. Block-grouped splitting must be standard for time-series benchmarking.
2. **Transferable Physical Dynamics on Pure Vibration:**
   * In `Spatial_Held_Out`, Random Forest trained on northern joints (Anomalies 1-5) and tested on southern joints (Anomalies 7-9) achieves an **ROC-AUC of $0.9191$** and **Precision of $0.5954$**. This confirms that high-frequency bogie impact features represent real, transferable physical dynamics.
3. **Dual Holdout Penalty:**
   * When both the operational run and the spatial location are held out simultaneously (`Combined_Run_Spatial`), F1 drops to **$0.2815$** and PR-AUC drops to **$0.3360$**. This represents the true, unvarnished baseline difficulty of zero-shot railway anomaly detection.
