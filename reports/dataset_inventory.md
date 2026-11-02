# Rail-VIVID Dataset Inventory and Repository Audit

**Dataset Title:** A Multimodal Railway Vibration and Vision Dataset (Rail-VIVID)  
**Permanent Citation:** Ma, S., Flanigan, K. A., Bergés, M., & Brooks, J. D. (2026). *A Multimodal Railway Vibration and Vision Dataset (Rail-VIVID)*. Scientific Data 13:1343. DOI: [10.1038/s41597-026-07555-y](https://doi.org/10.1038/s41597-026-07555-y)  
**Repository Source:** [atharva1305/Rail-VIVID on Hugging Face](https://huggingface.co/datasets/atharva1305/Rail-VIVID) (DOI: 10.57967/hf/841142)  
**Audit Date:** November 2026  

---

## 1. Executive Summary & Inventory Highlights

The Rail-VIVID dataset is a multimodal railway sensing benchmark collected along a dedicated 1.4-kilometer track section at the Railroad Development Corporation (RDC) facility in Orbisonia, Pennsylvania, utilizing a Pop-Up Metro Class 230 passenger train. The dataset encompasses synchronized high-rate vibration data, high-resolution track imagery, multi-constellation GNSS positioning, and ambient environmental conditions across multiple operational speeds and bi-directional track transits.

* **Total Recorded Runs:** 20 complete transits.
* **Total Time-Series Rows:** 9,401,255 rows.
* **Primary Sampling Rate:** 2,000 Hz (2 kHz) continuous digitization.
* **Total Camera Image Frames:** 105,901 high-resolution images (1920×1080 pixels).
* **Documented Ground-Truth Physical Anomalies:** 9 distinct track anomalies (7 joint gaps, 1 divergence turnout, 1 convergence turnout).
* **3D Point Cloud Laser Scans:** 4 proprietary FARO Focus 3D scan directories (`Anomaly1`, `Anomaly2`, `Anomaly4`, `Anomaly5`).

---

## 2. Inventory of Experimental Runs

The dataset is systematically divided into 20 run directories matching the strict naming convention `Direction_Speed_Index`:
* **Direction `AtoB`:** Transits from Point A (Southbound/Orbisonia origin, 40.23951° N, −77.89730° W) to Point B (Northbound terminus, 40.23184° N, −77.87387° W) representing an uphill gradient. Target speeds: 20.0 km/h and 22.5 km/h.
* **Direction `BtoA`:** Transits from Point B to Point A representing a downhill gradient. Target speeds: 22.5 km/h, 25.0 km/h, and 27.5 km/h.

| Run ID | Direction | Target Speed (km/h) | Collection Date | CSV Size (MB) | Row Count (2 kHz) | Effective Duration (s) | Camera Image Count | Effective Frame Rate (FPS) | Start Time (UTC/Local) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`AtoB_20_1`** | AtoB | 20.0 | 2025-04-11 | 114.97 | 593,028 | 296.51 | 5,971 | 20.1 | 2025-04-11 13:25:47.116 |
| **`AtoB_20_2`** | AtoB | 20.0 | 2025-04-11 | 105.21 | 577,370 | 288.68 | 5,871 | 20.3 | 2025-04-11 13:39:00.845 |
| **`AtoB_20_3`** | AtoB | 20.0 | 2025-04-11 | 88.73 | 490,782 | 245.39 | 6,004 | 24.5 | 2025-04-11 14:55:05.728 |
| **`AtoB_20_4`** | AtoB | 20.0 | 2025-04-18 | 106.61 | 546,783 | 273.39 | 6,367 | 23.3 | 2025-04-18 15:33:14.256 |
| **`AtoB_22.5_1`** | AtoB | 22.5 | 2025-04-11 | 98.46 | 504,131 | 252.06 | 6,461 | 25.6 | 2025-04-11 14:31:56.514 |
| **`AtoB_22.5_2`** | AtoB | 22.5 | 2025-04-11 | 94.71 | 486,172 | 243.09 | 5,391 | 22.2 | 2025-04-11 14:45:00.021 |
| **`AtoB_22.5_3`** | AtoB | 22.5 | 2025-04-11 | 83.44 | 462,856 | 231.43 | 5,119 | 22.1 | 2025-04-11 15:55:30.140 |
| **`AtoB_22.5_4`** | AtoB | 22.5 | 2025-04-18 | 103.50 | 501,354 | 250.68 | 5,119 | 20.4 | 2025-04-18 12:10:07.056 |
| **`BtoA_22.5_1`** | BtoA | 22.5 | 2025-04-11 | 103.15 | 497,506 | 248.75 | 5,960 | 24.0 | 2025-04-11 14:00:12.739 |
| **`BtoA_22.5_2`** | BtoA | 22.5 | 2025-04-11 | 86.68 | 478,817 | 239.41 | 5,806 | 24.3 | 2025-04-11 14:50:21.200 |
| **`BtoA_22.5_3`** | BtoA | 22.5 | 2025-04-11 | 106.07 | 544,004 | 272.00 | 6,347 | 23.3 | 2025-04-11 14:25:03.735 |
| **`BtoA_22.5_4`** | BtoA | 22.5 | 2025-04-11 | 108.81 | 559,009 | 279.50 | 5,727 | 20.5 | 2025-04-11 14:36:57.222 |
| **`BtoA_25_1`** | BtoA | 25.0 | 2025-04-11 | 73.05 | 403,885 | 201.94 | 4,617 | 22.9 | 2025-04-11 15:07:12.235 |
| **`BtoA_25_2`** | BtoA | 25.0 | 2025-04-11 | 72.72 | 399,228 | 199.61 | 4,428 | 22.2 | 2025-04-11 15:23:41.962 |
| **`BtoA_25_3`** | BtoA | 25.0 | 2025-04-11 | 75.75 | 417,837 | 208.92 | 4,664 | 22.3 | 2025-04-11 16:03:40.830 |
| **`BtoA_25_4`** | BtoA | 25.0 | 2025-04-18 | 84.83 | 410,091 | 205.05 | 4,487 | 21.9 | 2025-04-18 13:35:19.510 |
| **`BtoA_27.5_1`** | BtoA | 27.5 | 2025-04-11 | 74.79 | 413,233 | 206.62 | 4,613 | 22.3 | 2025-04-11 15:35:28.582 |
| **`BtoA_27.5_2`** | BtoA | 27.5 | 2025-04-11 | 70.23 | 388,020 | 194.01 | 4,534 | 23.4 | 2025-04-11 15:49:34.403 |
| **`BtoA_27.5_3`** | BtoA | 27.5 | 2025-04-18 | 78.21 | 380,115 | 190.06 | 4,192 | 22.1 | 2025-04-18 11:43:08.152 |
| **`BtoA_27.5_4`** | BtoA | 27.5 | 2025-04-18 | 71.77 | 347,034 | 173.52 | 4,223 | 24.3 | 2025-04-18 12:39:36.550 |
| **Totals / Averages** | - | - | - | **1,745.2 MB** | **9,401,255** | **4,700.6 s** | **105,901** | **22.8 FPS** | - |

---

## 3. Sensor Modalities, Variables, and Channel Mapping

Each run's `.csv` file presents 11 continuous tabular columns on a shared 2,000 Hz index:

| Column Name | Modality | Physical Unit | Sampling / Processing Method | Hardware Sensor & Location |
| :--- | :--- | :--- | :--- | :--- |
| `Timestamp` | Time Index | `YYYY-MM-DD HH:MM:SS.ffffff` | System clock reference (0.5 ms resolution) | Central DAQ Laptop Clock |
| `Channel1` | Vibration | Acceleration ($g$) | 2 kHz, 16-bit, 500 Hz Butterworth low-pass | Silicon Designs Model 2012 (Bogie Frame, Front) |
| `Channel2` | Vibration | Acceleration ($g$) | 2 kHz, 16-bit, 500 Hz Butterworth low-pass | Silicon Designs Model 2012 (Wheel Axle, Front) |
| `Channel3` | Vibration | Acceleration ($g$) | 2 kHz, 16-bit, 500 Hz Butterworth low-pass | Silicon Designs Model 2012 (Bogie Frame, Back) |
| `Channel4` | Vibration | Acceleration ($g$) | 2 kHz, 16-bit, 500 Hz Butterworth low-pass | Silicon Designs Model 2012 (Wheel Axle, Back) |
| `Channel5` | Vibration | Acceleration ($g$) | 2 kHz, 16-bit, 500 Hz Butterworth low-pass | Silicon Designs Model 2012 (Car Body, Front) |
| `Channel6` | Vibration | Acceleration ($g$) | 2 kHz, 16-bit, 500 Hz Butterworth low-pass | Silicon Designs Model 2012 (Car Body, Middle) |
| `Temperature` | Environmental | Degrees Fahrenheit (°F) | Upsampled to 2 kHz via cubic spline from hourly log | Weather station 40.23 km from test site |
| `Humidity` | Environmental | Relative Humidity ($0.0 - 1.0$) | Upsampled to 2 kHz via cubic spline from hourly log | Weather station 40.23 km from test site |
| `Latitude` | Positioning | Decimal Degrees | Linearly upsampled from 1 kHz GNSS estimates | Columbus P-7 Pro multi-GNSS receiver |
| `Longitude` | Positioning | Decimal Degrees | Linearly upsampled from 1 kHz GNSS estimates | Columbus P-7 Pro multi-GNSS receiver |

### Vision Sensor Specifications
* **Sensor:** See3CAM_50CUGM industrial camera with 5 MP Sony IMX264LLR-C monochrome CMOS sensor.
* **Optical Setup:** Mounted 1.52 m above the sleeper plane, angled downward at 15°.
* **Shutter Type:** Global Shutter (engineered to prevent rolling-shutter spatial warping).
* **Frame Resolution:** Full HD ($1920 \times 1080$ pixels).
* **Target FPS vs Actual FPS:** Configured for 100 FPS target; throttled to 20–36 FPS due to disk I/O bottlenecks.

---

## 4. Ground-Truth Anomaly Inventory (Physical vs File Audit)

A pivotal finding of this audit is that **no machine-readable ground truth labels (e.g. per-row or per-window labels) exist inside the CSV files**. Ground truth is documented in Table 2 of the companion Nature Scientific Data paper:

| Anomaly ID | Physical Defect Type | Measured Discontinuity Gap | Ground Truth Latitude | Ground Truth Longitude | 3D Point Cloud Available? | Physical Landmark Marker |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Rail Joint Gap | 2.0 cm | 40.23553° N | −77.88898° W | Yes (`Anomaly1/FARO_LS03353_017.fls`) | Orange Cone |
| **2** | Rail Joint Gap | 2.0 cm | 40.23512° N | −77.88770° W | Yes (`Anomaly2/FARO_LS03353_018.fls`) | Orange Cone |
| **3** | Rail Joint Gap | 2.4 cm | 40.23493° N | −77.88711° W | **No** (Missing from directory) | Orange Cone |
| **4** | Track Divergence (Switch) | N/A (Turnout) | 40.23461° N | −77.88612° W | Yes (`Anomaly4/FARO_LS03353_019.fls`) | Black Cone |
| **5** | Track Convergence (Switch) | N/A (Turnout) | 40.23425° N | −77.88485° W | Yes (`Anomaly5/FARO_LS03353_020.fls`) | Black Cone |
| **6** | Rail Joint Gap | 2.0 cm | 40.23388° N | −77.88320° W | **No** (Missing from directory) | Orange Cone |
| **7** | Rail Joint Gap | 2.5 cm | 40.23242° N | −77.87795° W | **No** (Missing from directory) | Orange Cone |
| **8** | Rail Joint Gap | 2.3 cm | 40.23205° N | −77.87651° W | **No** (Missing from directory) | Orange Cone |
| **9** | Rail Joint Gap | 3.0 cm | 40.23194° N | −77.87508° W | **No** (Missing from directory) | Orange Cone |

### Critical Findings & Anomaly Discrepancies
1. **Incomplete 3D Scanning Data:** The `Anomaly/Anomaly3DScanning/` folder contains point cloud data for only 4 anomalies (`Anomaly1`, `Anomaly2`, `Anomaly4`, `Anomaly5`). Scans for Anomalies 3, 6, 7, 8, and 9 are absent from the dataset distribution.
2. **Homogeneity of Defects:** Of the 9 anomalies, 7 are rail joints (gaps ranging from 2.0 to 3.0 cm) and 2 are structural switches/turnouts. No broken rail fractures, squats, corrugation, or internal fatigue defects are present.
3. **Point Cloud Format:** The available 3D scans are stored in FARO's proprietary `.fls` workspace format rather than open `.ply` or `.las` formats, requiring proprietary SDKs or specialized converters for point retrieval.

