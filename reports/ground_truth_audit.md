# Ground-Truth and Anomaly Label Audit Report

**Project:** Rail-VIVID Dataset Benchmarking and Feasibility Audit  
**Date of Audit:** November 2026  
**Status:** Complete Feasibility & Label Verification  

---

## 1. Executive Summary & Core Verdict

A rigorous audit of the Rail-VIVID dataset reveals a fundamental architectural reality:
1. **Absence of Machine-Readable Per-Sample/Per-Window Labels:** There is no dedicated label column, annotation mask, or bounding box file shipped in the dataset.
2. **Authoritative Ground Truth Defined in Literature:** Ground-truth defects are explicitly documented in Table 2 of the accompanying *Nature Scientific Data* paper (`s41597-026-07555-y.pdf`), specifying 9 distinct spatial coordinates.
3. **Severe Class Imbalance and Limited Defect Diversity:** Across the entire 1.4-km testbed, there are exactly **9 independent physical anomalies**:
   * **7 Rail Joints** (expansion/mechanical gaps between rail sections: 2.0 cm, 2.0 cm, 2.4 cm, 2.0 cm, 2.5 cm, 2.3 cm, 3.0 cm).
   * **2 Turnout / Switch Geometries** (1 divergence track switch, 1 convergence track switch).
   * **0 Structural Rail Breaks, Cracks, or Severe Squats.**
4. **All 20 Runs Cover the Same 9 Anomalies:** Because all 20 runs travel over the exact same 1.25 km physical segment (AtoB uphill, BtoA downhill), every single run encounters the exact same 9 defects in sequence.
5. **Effective Sample Size for Generalization:** From a statistical learning perspective, the number of independent defect instances is **$N=9$**, not the number of time windows ($N \approx 50,000$) or camera frames ($N \approx 105,901$).

---

## 2. Inventory and Spatial Analysis of Documented Anomalies

The table below summarizes the 9 ground-truth anomalies, their physical characteristics, spatial coordinates, and their pairwise track separations:

| Anomaly ID | Physical Type | Metric Dimension | Latitude (°N) | Longitude (°W) | Distance from Prev Anomaly (m) | Cumulative Track Offset from Anom 1 (m) | Visual Cone Marker | 3D Point Cloud Available? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Rail Joint Gap | 2.0 cm gap | 40.23553 | −77.88898 | 0.0 m | 0.0 m | Orange Cone | Yes (`FARO_LS03353_017.fls`) |
| **2** | Rail Joint Gap | 2.0 cm gap | 40.23512 | −77.88770 | 117.8 m | 117.8 m | Orange Cone | Yes (`FARO_LS03353_018.fls`) |
| **3** | Rail Joint Gap | 2.4 cm gap | 40.23493 | −77.88711 | **54.4 m** | 172.2 m | Orange Cone | **No** (Missing) |
| **4** | Track Divergence | Turnout frog | 40.23461 | −77.88612 | 91.3 m | 263.4 m | Black Cone | Yes (`FARO_LS03353_019.fls`) |
| **5** | Track Convergence | Turnout frog | 40.23425 | −77.88485 | 115.0 m | 378.4 m | Black Cone | Yes (`FARO_LS03353_020.fls`) |
| **6** | Rail Joint Gap | 2.0 cm gap | 40.23388 | −77.88320 | 146.0 m | 523.8 m | Orange Cone | **No** (Missing) |
| **7** | Rail Joint Gap | 2.5 cm gap | 40.23242 | −77.87795 | **474.3 m** | 998.1 m | Orange Cone | **No** (Missing) |
| **8** | Rail Joint Gap | 2.3 cm gap | 40.23205 | −77.87651 | 129.0 m | 1,127.1 m | Orange Cone | **No** (Missing) |
| **9** | Rail Joint Gap | 3.0 cm gap | 40.23194 | −77.87508 | 122.0 m | 1,245.6 m | Orange Cone | **No** (Missing) |

### Key Spatial Observations:
* **Tight Clustering:** Anomalies 2 and 3 are separated by only **54.4 meters**. At 27.5 km/h (7.64 m/s), the train transits between Anomaly 2 and Anomaly 3 in just **7.1 seconds**.
* **Long Nominal Span:** The segment between Anomaly 6 and Anomaly 7 spans **474.3 meters** (~62 to 85 seconds of travel), providing the single largest contiguous block of uncontaminated normal track in the entire dataset.

---

## 3. Sensor Synchronization and Uncertainty Audit

### 3.1 GNSS Positioning & Track Mapping
* **Hardware:** Columbus P-7 Pro multi-GNSS receiver mounted on the vehicle roof with unobstructed sky view.
* **Update Rate & Smoothing:** Receiver logs at 1 kHz with internal Kalman smoothing; linearly upsampled to 2 kHz in the CSV file.
* **Spatial Resolution:** Empirical verification across all 20 runs confirms that the closest recorded GNSS point to each anomaly coordinate is strictly **$\le 0.8\text{ meters}$**.
* **Limitation:** The GNSS receiver measures antenna position on the vehicle roof, not the contact patch of the wheel on the rail. The front wheelset is offset from the GNSS antenna by the vehicle body geometry (~3–5 meters).

### 3.2 Accelerometer Synchronization
* **Digitizer:** NI cDAQ-9185 chassis with simultaneous 16-bit analog-to-digital conversion at 2,000 Hz.
* **Cross-Channel Alignment:** All 6 vibration channels are digitized on the same backplane clock. Channel-to-channel time skew is virtually zero ($< 1\,\mu\text{s}$).
* **Dynamic Delay:** When the train hits a joint gap, the wheelset (Channel 2/4) experiences an immediate shock impulse, whereas the bogie (Channel 1/3) responds after ~2–5 ms through the primary suspension, and the car body (Channel 5/6) responds through the secondary air springs with prolonged low-frequency oscillation (~50–200 ms delay).

### 3.3 Camera Timestamp Drift (Major Bottleneck)
* **Official Dataset Warning (Release Notes):**
  > *"Under variable-FPS conditions this introduces a non-constant drift that can displace an image by up to ~15 m from its true location. Location-dependent analyses should therefore validate image positions against in-scene references (e.g., the surveyed cones at Point A, Point B, and each anomaly) rather than relying on timestamps alone."*
* **Root Cause Identified in `Acquisition.py`:**
  Frame capture used `cv2.imwrite(VIDEO_PATH + str(datetime.now().strftime(...)) + ".jpg", frame)` within a busy logging loop. The recorded timestamp represents the completion time of disk write buffering rather than the hardware CMOS sensor trigger moment.
* **Multimodal Fusion Implication:**
  Attempting sample-level or microsecond-level late fusion between raw video frames and 2 kHz vibration time series based solely on CSV timestamps will introduce severe temporal misalignment (up to $\pm 15$ meters or $\pm 2.0$ seconds). Visual models must be evaluated either:
  1. In a self-supervised image reconstruction paradigm (as done by Ma et al. in the paper), or
  2. With wide spatial bounding tolerances ($\pm 15$ m) anchored by the surveyed physical cones.

---

## 4. Normal vs Abnormal Track Formulation

### How Did the Original Paper Define Labels?
In Section 4 (*Technical Validation - Anomalies Validation*), the authors established the following scientific convention:
> *"The test track was computationally divided into contiguous 7.62-m segments. Each segment was labeled as 'Abnormal' if a ground-truth anomaly was located within its or its neighbor's boundaries, and 'Normal' otherwise."*

### Scientific Critique and Audit of This Definition:
1. **Conservative vs Aggressive Normal Labeling:**
   * Claiming an unlabeled 7.62 m track segment is "guaranteed 100% normal" is an assumption. Track can possess unrecorded roughness, ballast settling, or railhead wear.
   * However, segments situated $> 25$ meters away from any documented anomaly (especially within the 474 m span between Anomaly 6 and 7) represent the highest-confidence normal reference data.
2. **Buffer Zones and Leakage Protection:**
   * Because vehicle dynamic ringing persists after impacting an irregularity, labeling only the exact millimeter of the gap as abnormal causes severe label noise.
   * A temporal window immediately following a defect contains strong residual impulse vibrations. If that window is labeled "Normal", supervised models are penalized for detecting valid fault dynamics.
   * **Recommendation:** Adopt a three-zone labeling scheme:
     * **Core Anomaly Zone:** $\pm 5$ meters around ground truth ($\text{Label} = 1$).
     * **Transition Buffer Zone:** $5$ to $15$ meters from anomaly ($\text{Label} = \text{Ignore/Excluded}$ to prevent leakage into normal class).
     * **Conservative Normal Zone:** $> 20$ meters from any documented anomaly ($\text{Label} = 0$).

---

## 5. Statistical Summary of the Ground Truth

* **Number of Unique Physical Anomaly Sites:** **9**
* **Number of Normal Reference Zones:** **8 distinct inter-anomaly track intervals**
* **Total Repeated Observations per Defect:** **20 transits** (8 uphill at 20.0/22.5 km/h, 12 downhill at 22.5/25.0/27.5 km/h).
* **Cross-Run Repeatability:** Excellent spatial repeatability across all 20 runs (GPS track divergence $< 0.8$ m).
* **Sample Size Danger:** Any supervised model trained on all 9 anomalies cannot be claimed to "detect railway anomalies in general"; it can only demonstrate that it has memorized the 9 specific physical geometries of this 1.4-km track.

