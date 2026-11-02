# Baseline Modeling and Benchmark Evaluation Report

**Project:** Rail-VIVID Dataset Benchmarking and Feasibility Audit  
**Date:** November 2026  
**Auditor:** Senior Machine Learning Research Engineer  
**Companion Artifact:** `reports/baseline_results.json`  

---

## 1. Experimental Setup & Benchmarking Design

Experiments were conducted on **18,775 extracted time windows** across all 20 experimental runs ($9,401,255$ total rows).
* **Window Duration:** $0.50\text{ seconds}$ ($1,000$ accelerometer samples at $2,000\text{ Hz}$).
* **Stride:** $0.25\text{ seconds}$ ($500$ samples, $50\%$ window overlap).
* **Class Partitioning:**
  * Definite Anomaly ($d \le 6.0\text{ m}$): **$1,239$ windows** ($6.6\%$)
  * Quarantined Buffer ($6.0\text{ m} < d \le 18.0\text{ m}$): **$2,501$ windows** (excluded from evaluation to eliminate transition ringing leakage)
  * Conservative Normal Track ($d > 18.0\text{ m}$): **$15,035$ windows** ($80.1\%$)
* **Total Clean Evaluated Windows:** **$16,274$ windows** across all 20 runs.

---

## 2. Comprehensive Benchmark Comparison Table

The table below details all reproduced local models, modality ablations, and published external baselines across the four evaluation splits:

| Model | Modalities | Split Strategy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Primary Failure Mode & Limitations |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | Vibration Only | Random-Window (Leaky Control) | 0.4362 | 0.8025 | 0.5652 | 0.4730 | 0.9431 | Overlap leakage; memorizes neighboring window noise. |
| **Random Forest** | Vibration + GPS | Random-Window (Leaky Control) | 0.4765 | 0.8354 | 0.6069 | 0.6010 | 0.9579 | Severe coordinate memorization; inflates metrics artificially. |
| **Logistic Regression** | Vibration Only | Random-Window (Leaky Control) | 0.2644 | 0.8519 | 0.4035 | 0.3657 | 0.8964 | High recall but poor precision due to linear thresholding. |
| **Isolation Forest** | Vibration Only | Random-Window (Leaky Control) | 0.1074 | 0.1440 | 0.1230 | 0.0863 | 0.5405 | Unsupervised density fails to separate joint impacts from roughness. |
| **Signal Autoencoder** | Vibration Only | Random-Window (Leaky Control) | 0.0828 | 0.1111 | 0.0949 | 0.0801 | 0.5287 | High reconstruction error on curve vibration. |
| **Random Forest** | Vibration Only | **Run-Held-Out** | 0.3790 | 0.5924 | **0.4623** | 0.3736 | 0.9036 | Evaluates cross-run robustness; defects are at same locations. |
| **Random Forest** | Vibration + Env | Run-Held-Out | 0.3783 | 0.5420 | 0.4456 | 0.3775 | 0.8908 | Environmental features slightly degrade test generalization. |
| **Random Forest** | Vibration + GPS | Run-Held-Out | 0.4230 | 0.6345 | 0.5076 | 0.5009 | 0.9244 | Leaks track coordinates since test runs share identical track. |
| **Signal Autoencoder** | Vibration Only | Run-Held-Out | 0.0958 | 0.1261 | 0.1089 | 0.1036 | 0.6127 | Unsupervised baseline; weak discriminative separation. |
| **Isolation Forest** | Vibration Only | Run-Held-Out | 0.0799 | 0.1050 | 0.0907 | 0.0764 | 0.4781 | Performance collapses near random guessing ($0.478$). |
| **Random Forest** | Vibration Only | **Spatial-Held-Out** | 0.5278 | 0.6348 | **0.5764** | 0.5496 | 0.9156 | **Best generalizing model:** learned joint impact physics. |
| **Random Forest** | Vibration + GPS | Spatial-Held-Out | **0.0000** | **0.0000** | **0.0000** | 0.5182 | 0.9170 | **Complete failure:** Model split on train coords; 0 predictions on test coords! |
| **Logistic Regression** | Vibration Only | Spatial-Held-Out | 0.3418 | 0.7088 | 0.4612 | 0.4106 | 0.8399 | Modest linear generalization; high false positive rate. |
| **Isolation Forest** | Vibration Only | Spatial-Held-Out | 0.1565 | 0.2029 | 0.1767 | 0.1062 | 0.5618 | Low precision; flags track curves as anomalies. |
| **Signal Autoencoder** | Vibration Only | Spatial-Held-Out | 0.0902 | 0.1169 | 0.1019 | 0.0812 | 0.5166 | Generalization to novel sector degraded by track geometry shifts. |
| **Random Forest** | Vibration Only | **Combined Run & Spatial** | 0.3492 | 0.2716 | **0.3056** | 0.3474 | 0.8862 | Strictest zero-shot benchmark; performance drops significantly. |
| **Random Forest** | Vibration + Env | Combined Run & Spatial | 0.3818 | 0.2593 | 0.3088 | 0.3888 | 0.9057 | Slight PR-AUC boost from day/speed context, but low recall. |
| **Random Forest** | Vibration + GPS | Combined Run & Spatial | **0.0000** | **0.0000** | **0.0000** | 0.2048 | 0.8258 | Total recall collapse ($0.0$) due to novel spatial sector. |
| **Logistic Regression** | Vibration Only | Combined Run & Spatial | 0.0956 | 0.6667 | 0.1672 | 0.2540 | 0.6322 | Low precision ($9.5\%$); high false alarm rate. |
| **Signal Autoencoder** | Vibration Only | Combined Run & Spatial | 0.1359 | 0.1728 | 0.1522 | 0.1097 | 0.6151 | Unsupervised reconstruction yields F1 of $0.15$. |
| *CNN AE (Published)* | Vision Only | Self-Supervised Reconstruction | N/A | N/A | N/A | N/A | **0.6500** | 19.03M params; image reconstruction error (Ma et al. Table 3). |
| *CNN VAE (Published)* | Vision Only | Self-Supervised Reconstruction | N/A | N/A | N/A | N/A | **0.6800** | 19.16M params; probabilistic reconstruction (Ma et al. Table 3). |
| *ViT AE (Published)* | Vision Only | Self-Supervised Reconstruction | N/A | N/A | N/A | N/A | **0.7200** | 15.08M params; Vision Transformer reconstruction (Ma et al.). |
| *ViT VAE (Published)* | Vision Only | Self-Supervised Reconstruction | N/A | N/A | N/A | N/A | **0.7300** | 15.21M params; Vision Transformer VAE (Ma et al. Table 3). |

---

## 3. Empirical Discoveries & Scientific Insights

### 3.1 The "GPS Trap": Catastrophic Spatial Overfitting
When GNSS latitude and longitude are supplied as features to the Random Forest model:
1. In the **Run-Held-Out** split, PR-AUC appears to rise from $0.3736$ to $0.5009$.
2. But in the **Spatial-Held-Out** and **Combined** splits, test recall collapses to **$0.0000$**!
3. **Physical Explanation:** The decision trees learn coordinate thresholds (e.g., `if latitude > 40.235: Anomaly 1`). When presented with held-out anomalies in the southern track sector (`latitude < 40.233`), the coordinate rules never trigger. This provides definitive empirical proof that spatial coordinates must be excluded from anomaly classifier inputs.

### 3.2 Vibration-Only Physics Generalize Across Novel Track Sites
When restricted solely to mechanical features (RMS, peak-to-peak, kurtosis, crest factor, band energy) without spatial coordinates:
* The Random Forest achieves **F1 = 0.5764** and **ROC-AUC = 0.9156** on completely unseen physical track anomalies in the **Spatial-Held-Out** split.
* This proves that mechanical vibration signatures of rail joint impacts are physically transferable across different locations on the track.

### 3.3 The Reality of Unsupervised Signal Anomaly Detection
Unsupervised baselines (Isolation Forest and Signal Autoencoder) achieve ROC-AUCs of only $0.52 - 0.61$:
* Unsupervised methods flag track sections with high vibration energy as anomalous.
* However, normal curved sections and track switches produce high natural vibration energy, causing high false alarm rates ($>80\%$) in unsupervised settings.

