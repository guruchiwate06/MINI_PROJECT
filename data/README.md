# Rail-VIVID Data Directory

This directory stores intermediate artifacts, metadata, and cached feature matrices for the Rail-VIVID benchmarking audit.

## Directory Structure
- `cached_window_features.csv`: Windowed time-domain and spectral features extracted from 2,000 Hz vibration channels across experimental runs, annotated with spatial boundary labels ($1 = \text{Anomaly}$, $-1 = \text{Buffer Quarantined}$, $0 = \text{Normal}$).
- `README.md`: Documentation of data formats, sensor configurations, and physical coordinate mappings.

## Ground Truth Physical Anomalies
Documented in Table 2 of Ma et al. (2026), Nature Scientific Data:
- **Anomaly 1:** Rail Joint Gap (2.0 cm) @ `40.23553° N, -77.88898° W`
- **Anomaly 2:** Rail Joint Gap (2.0 cm) @ `40.23512° N, -77.88770° W`
- **Anomaly 3:** Rail Joint Gap (2.4 cm) @ `40.23493° N, -77.88711° W`
- **Anomaly 4:** Track Divergence Turnout @ `40.23461° N, -77.88612° W`
- **Anomaly 5:** Track Convergence Turnout @ `40.23425° N, -77.88485° W`
- **Anomaly 6:** Rail Joint Gap (2.0 cm) @ `40.23388° N, -77.88320° W`
- **Anomaly 7:** Rail Joint Gap (2.5 cm) @ `40.23242° N, -77.87795° W`
- **Anomaly 8:** Rail Joint Gap (2.3 cm) @ `40.23205° N, -77.87651° W`
- **Anomaly 9:** Rail Joint Gap (3.0 cm) @ `40.23194° N, -77.87508° W`

## Raw Dataset Reference
Raw 2 kHz vibration CSVs and Full HD camera image folders are stored in the root directory under `AtoB_[Speed]_[Index]` and `BtoA_[Speed]_[Index]`.

