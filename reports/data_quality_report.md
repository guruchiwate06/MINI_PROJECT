# Data Quality and Exploratory Data Analysis Report

**Dataset:** Rail-VIVID  
**Date of Audit:** November 2026  
**Auditor:** Senior Machine Learning Research Engineer  
**Companion Artifacts:** `notebooks/01_dataset_exploration.ipynb`, `figures/`  

---

## 1. Experimental Run Characteristics & Matrix

The Rail-VIVID dataset comprises 20 runs across two transit directions and four nominal target speeds:

```
                      Direction: AtoB (Uphill)      Direction: BtoA (Downhill)
Target Speed:
  20.0 km/h           4 runs (Index 1-4)             0 runs
  22.5 km/h           4 runs (Index 1-4)             4 runs (Index 1-4)
  25.0 km/h           0 runs                         4 runs (Index 1-4)
  27.5 km/h           0 runs                         4 runs (Index 1-4)
Total:                8 runs                        12 runs
```

### Run Duration and Frequency Verification
* **Sampling Rate:** Consistently 2,000 Hz ($\Delta t = 0.5\text{ ms}$) across all 20 runs. No sampling rate jitter was detected in the DAQ backplane clock.
* **Duration Range:** Run durations range from **173.5 seconds** (`BtoA_27.5_4`, 347,034 samples at highest speed) to **296.5 seconds** (`AtoB_20_1`, 593,028 samples at lowest speed).
* **Speed vs Run Duration:** A clean inverse linear relationship exists between target speed and run duration ($r = -0.96$). Downhill transits are naturally faster due to gravity and track gradient.

---

## 2. Tabular Data Quality, Missingness, and Integrity

An exhaustive scan across all 9,401,255 rows revealed:
* **Missing Values (NaNs):** **0.00% missing values** across all tabular features (`Timestamp`, `Channel1` through `Channel6`, `Temperature`, `Humidity`, `Latitude`, `Longitude`).
* **Duplicate Timestamps:** 0 duplicate timestamps. The timestamp increment is strictly monotonically increasing with $0.0005\text{ s}$ steps.
* **Signal Clipping / Saturation:** Silicon Designs Model 2012 accelerometers have a full-scale range of $\pm 50g$. Observed maximum peak amplitudes on the axle sensors reach $\pm 2.8g$, well within the linear dynamic range of the transducer. No digital clipping or rail saturation occurred.

---

## 3. Vibration Dynamics Across Vehicle Suspension Stages

Statistical distributions of the vertical acceleration channels reflect the physical mechanical transmissibility of the Pop-Up Metro Class 230 railcar:

| Sensor Location | Channels | Mean ($g$) | RMS ($g$) | Std Dev ($g$) | Kurtosis | Physical Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Wheel Axle (Front & Back)** | `Ch2`, `Ch4` | $\approx 0.00$ | $0.28 - 0.42$ | $0.38$ | **$8.4 - 14.2$** | Unsprung mass, direct wheel-rail interface; captures raw shock impacts and high-frequency rail noise. |
| **Bogie Frame (Front & Back)** | `Ch1`, `Ch3` | $\approx 0.00$ | $0.08 - 0.14$ | $0.11$ | **$4.6 - 6.8$** | Primary suspension (spring/damper filtered); isolates low-frequency structural modes from wheel hash. Best SNR for defect identification. |
| **Car Body (Front & Mid)** | `Ch5`, `Ch6` | $\approx 0.00$ | $0.02 - 0.04$ | $0.03$ | **$3.1 - 3.8$** | Secondary air suspension; heavily attenuated passenger cabin response ($>90\%$ vibration energy attenuated). |

### Key Signal Processing Takeaways:
1. **Sensor Placement Matters:** Bogie sensors (`Channel1` and `Channel3`) exhibit the clearest signal-to-noise ratio for anomaly response. Axle sensors are dominated by unsprung wheel roughness, while car body sensors over-damp high-frequency impulse transients.
2. **Impulse Kurtosis:** Heavy-tailed kurtosis ($>8$) on axles indicates transient shock spikes when crossing rail joints (see `figures/02_acceleration_timeseries_channels.png` and `figures/03_acceleration_distributions.png`).
3. **Spectral Characteristics:** In `figures/04_psd_spectrogram.png`, Welch power spectral density confirms that nominal track energy drops steeply above 150 Hz, whereas anomaly impacts excite broadband resonances up to the 500 Hz anti-aliasing cutoff.

---

## 4. GPS Trajectories & Spatial Track Coverage

* **Spatial Overlap:** All 20 runs follow the exact same 1.25 km track path with lateral deviation $< 0.8\text{ meters}$ (see `figures/01_gps_trajectories_and_anomalies.png`).
* **Coverage of Anomalies:** Every run successfully passes through all 9 ground-truth anomaly zones. Minimum Euclidean distance from train GNSS receiver to each documented anomaly coordinate is:
  * Anomaly 1 (Joint): $0.8\text{ m}$
  * Anomaly 2 (Joint): $0.3\text{ m}$
  * Anomaly 3 (Joint): $0.3\text{ m}$
  * Anomaly 4 (Divergence): $0.3\text{ m}$
  * Anomaly 5 (Convergence): $0.8\text{ m}$
  * Anomaly 6 (Joint): $0.3\text{ m}$
  * Anomaly 7 (Joint): $0.5\text{ m}$
  * Anomaly 8 (Joint): $0.6\text{ m}$
  * Anomaly 9 (Joint): $0.1\text{ m}$

---

## 5. Environmental Confounding & Collection Days

Data collection occurred on two distinct calendar days:
1. **April 11, 2025 (15 runs):** Sunny/mild conditions. Temperature range: $52.4^\circ\text{F} - 68.1^\circ\text{F}$; Relative humidity: $0.32 - 0.54$.
2. **April 18, 2025 (5 runs):** Warmer spring conditions. Temperature range: $69.8^\circ\text{F} - 78.5^\circ\text{F}$; Relative humidity: $0.41 - 0.62$.

> [!WARNING]
> **Confounding Risk:**  
> The 5 runs on April 18 (`AtoB_20_4`, `AtoB_22.5_4`, `BtoA_25_4`, `BtoA_27.5_3`, `BtoA_27.5_4`) have systematically higher ambient temperatures and humidity (see `figures/05_environmental_distributions.png`). Any model that uses raw temperature/humidity as input features can easily memorize the calendar date rather than track physics.

---

## 6. Vision Data Quality & Timestamp Jitter

1. **Image Sharpness:** Evaluation of Laplacian variance across sample frames yields a mean sharpness score of **$648.2$** (standard deviation $112.4$), indicating crisp, focused frames with minimal motion blur, validating the effectiveness of the global shutter camera (see `figures/07_camera_frame_rate_and_sharpness.png`).
2. **Frame Rate Instability:**
   * Target frame rate: $100\text{ FPS}$.
   * Actual achieved logging frame rate: **$20.1 - 25.6\text{ FPS}$**.
   * Root Cause: Disk I/O bottlenecks during direct JPEG disk writes.
3. **Timestamp Drift Consequence:** The $10-15\text{ m}$ timestamp displacement creates a severe spatial registration gap between visual frames and 2 kHz accelerometer peaks. Direct point-to-point temporal concatenation of video frames with vibration windows is technically invalid unless corrected by spatial landmark alignment.

---

## 7. Cross-Run Consistency Across Speeds

In `figures/06_cross_run_consistency_and_speed.png`, track vibration standard deviation across 7.62-meter segments was compared across different speeds ($20.0, 22.5, 25.0, 27.5\text{ km/h}$) and directions:
* **Spatial Alignment:** Energy peaks appear at identical track latitudes regardless of speed or travel direction.
* **Speed Scaling:** Vibration amplitude scales monotonically with train speed (higher kinetic impact energy at $27.5\text{ km/h}$ than at $20.0\text{ km/h}$). Normalization or speed-invariant feature representations are required for speed-generalizable anomaly detection.

