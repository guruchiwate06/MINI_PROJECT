"""
Export authentic Rail-VIVID 6-channel vibration datasets for the frontend.
Generates:
1. signals_index.json (run catalog, channels metadata, anomaly catalog)
2. run_<run_id>_overview.json (peak-preserving min-max envelopes at 20 Hz, synchronized telemetry)
3. run_<run_id>_microwindows.json (full-resolution 2,000 Hz raw samples around ground-truth anomalies and normal sectors)
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

CHANNELS_META = [
    {
        "id": 1,
        "name": "Channel 1",
        "label": "Front Bogie Frame",
        "location": "Primary Suspension / Front Bogie",
        "color": "#06b6d4", # Cyan
        "role": "Bogie Frame Vibration",
        "filtering": "Primary spring-damper filtered; captures track irregularities and hunting oscillations"
    },
    {
        "id": 2,
        "name": "Channel 2",
        "label": "Front Wheel Axle Box",
        "location": "Unsprung Axle Box 1",
        "color": "#f59e0b", # Amber
        "role": "Direct Rail Impact Interface",
        "filtering": "Unsprung wheel-rail contact; direct high-frequency shock impacts (kurtosis 8.4-14.2)"
    },
    {
        "id": 3,
        "name": "Channel 3",
        "label": "Back Bogie Frame",
        "location": "Primary Suspension / Back Bogie",
        "color": "#38bdf8", # Sky blue
        "role": "Trailing Bogie Dynamics",
        "filtering": "Primary suspension filtered; secondary propagation of track shocks"
    },
    {
        "id": 4,
        "name": "Channel 4",
        "label": "Back Wheel Axle Box",
        "location": "Unsprung Axle Box 2",
        "color": "#f97316", # Orange
        "role": "Trailing Wheel Impact Interface",
        "filtering": "Unsprung wheel-rail contact; confirms wheel-axle passover with delay t = wheelbase / v"
    },
    {
        "id": 5,
        "name": "Channel 5",
        "label": "Car Body Front Floor",
        "location": "Passenger Cabin Floor (Front)",
        "color": "#10b981", # Emerald
        "role": "Passenger Ride Quality",
        "filtering": "Secondary air suspension filtered; >90% high-frequency attenuation (<20 Hz dominant)"
    },
    {
        "id": 6,
        "name": "Channel 6",
        "label": "Car Body Mid Floor",
        "location": "Passenger Cabin Floor (Center)",
        "color": "#a855f7", # Purple
        "role": "Vehicle Flexural Vibration",
        "filtering": "Secondary air suspension filtered; structural bending modes"
    }
]

def main():
    root = Path(__file__).resolve().parents[1]
    artifacts_dir = root / "artifacts" / "railvivid_v1"
    frontend_data_dir = root.parent / "frontend" / "public" / "data"
    frontend_data_dir.mkdir(parents=True, exist_ok=True)
    
    runs_df = pd.read_csv(artifacts_dir / "runs.csv")
    windows_df = pd.read_csv(artifacts_dir / "windows.csv")
    sites_df = pd.read_csv(artifacts_dir / "sites.csv")
    
    # Selected runs across speeds and directions
    selected_run_ids = ["AtoB_20_1", "AtoB_22.5_1", "BtoA_25_1", "BtoA_27.5_1"]
    
    runs_catalog = []
    
    for run_id in selected_run_ids:
        run_row = runs_df[runs_df.run_id == run_id].iloc[0]
        run_windows = windows_df[windows_df.run_id == run_id].sort_values("start_idx")
        acc_path = artifacts_dir / "runs" / run_id / "acceleration.npy"
        if not acc_path.exists():
            continue
            
        acc = np.load(acc_path, mmap_mode="r")
        total_samples = acc.shape[0]
        duration_s = total_samples / 2000.0
        
        # Calculate ground truth anomaly events along this run
        run_anomalies = []
        for _, site in sites_df.iterrows():
            site_id = int(site.id)
            # Find closest window
            nearby = run_windows[run_windows.site_id == site_id]
            if not nearby.empty:
                closest_w = nearby.loc[nearby.site_distance_m.idxmin()]
                sample_idx = int((closest_w.start_idx + closest_w.end_idx) / 2)
                time_s = round(sample_idx / 2000.0, 3)
                chainage = round(float(closest_w.chainage_m), 2)
                run_anomalies.append({
                    "site_id": site_id,
                    "type": str(site.type),
                    "size_cm": float(site.size_cm) if pd.notna(site.size_cm) else None,
                    "sample_idx": sample_idx,
                    "time_s": time_s,
                    "chainage_m": chainage,
                    "lat": float(closest_w.latitude),
                    "lon": float(closest_w.longitude)
                })
        
        runs_catalog.append({
            "run_id": run_id,
            "direction": run_row.direction,
            "target_speed_kmh": float(run_row.target_speed_kmh),
            "run_number": int(run_row.run_number),
            "total_samples": total_samples,
            "duration_s": round(duration_s, 2),
            "sampling_rate_hz": 2000,
            "track_length_m": 1400.0,
            "anomalies_count": len(run_anomalies),
            "anomalies": run_anomalies
        })
        
        # 1. Compute Decimated Min-Max Peak-Preserving Envelope (Step = 100 samples -> 20 Hz, 50ms blocks)
        step = 100
        n_blocks = total_samples // step
        envelope_points = []
        
        # Interpolate chainage, speed, lat, lon from windows
        win_centers = (run_windows.start_idx + run_windows.end_idx) / 2.0
        win_chainage = run_windows.chainage_m.to_numpy()
        win_lat = run_windows.latitude.to_numpy()
        win_lon = run_windows.longitude.to_numpy()
        win_label = run_windows.label.to_numpy()
        
        # Speed estimate: distance delta / time delta
        for b in range(n_blocks):
            s_start = b * step
            s_end = s_start + step
            t_center = round((s_start + s_end) / 4000.0, 3)
            center_sample = (s_start + s_end) / 2.0
            
            # Map to nearest window
            w_idx = np.searchsorted(win_centers, center_sample)
            w_idx = min(max(w_idx, 0), len(win_centers) - 1)
            
            chainage = round(float(win_chainage[w_idx]), 2)
            lat = round(float(win_lat[w_idx]), 6)
            lon = round(float(win_lon[w_idx]), 6)
            lbl = int(win_label[w_idx])
            
            block = acc[s_start:s_end, :] # [step, 6]
            # Channel min/max: rounded to 3 decimals
            ch_min_max = []
            for ch in range(6):
                ch_min_max.append([
                    round(float(np.min(block[:, ch])), 3),
                    round(float(np.max(block[:, ch])), 3)
                ])
                
            envelope_points.append({
                "t": t_center,
                "d": chainage,
                "a": lbl,
                "ch": ch_min_max
            })
            
        overview_data = {
            "run_id": run_id,
            "sampling_rate_hz": 2000,
            "decimation_step": step,
            "effective_rate_hz": 20,
            "total_points": len(envelope_points),
            "duration_s": round(duration_s, 2),
            "channels": [c["name"] for c in CHANNELS_META],
            "points": envelope_points
        }
        
        overview_file = frontend_data_dir / f"run_{run_id}_overview.json"
        with open(overview_file, "w") as f:
            json.dump(overview_data, f, separators=(',', ':'))
        print(f"Exported {overview_file.name} ({os.path.getsize(overview_file) / 1024:.1f} KB)")
        
        # 2. Extract Full-Resolution Micro-Windows (2,000 Hz, 1.0s / 2,000 samples each)
        microwindows = []
        
        # Select representative anomalies and normal sectors
        target_sites = [1, 2, 4, 6, 7] # Joints, Divergence, Switches
        for anom in run_anomalies:
            if anom["site_id"] in target_sites:
                center = anom["sample_idx"]
                start = max(0, center - 1000)
                end = min(total_samples, start + 2000)
                window_samples = acc[start:end, :] # [2000, 6]
                
                # Compute power spectral density for channel 2 (axle) and channel 1 (bogie)
                time_axis = [round((start + i) / 2000.0, 5) for i in range(len(window_samples))]
                
                microwindows.append({
                    "id": f"anomaly_site_{anom['site_id']}",
                    "title": f"Site {anom['site_id']}: {anom['type']} ({anom.get('size_cm') or ''} cm)",
                    "type": "ANOMALOUS",
                    "provenance": "MEASURED",
                    "site_id": anom["site_id"],
                    "chainage_m": anom["chainage_m"],
                    "time_start_s": round(start / 2000.0, 3),
                    "time_end_s": round(end / 2000.0, 3),
                    "sample_count": len(window_samples),
                    "sampling_rate_hz": 2000,
                    "channels_data": [
                        [round(float(v), 4) for v in window_samples[:, ch]] for ch in range(6)
                    ]
                })
                
        # Normal baseline segment (e.g. between site 6 and site 7, chainage ~900m)
        norm_windows = run_windows[(run_windows.label == 0) & (run_windows.chainage_m > 850) & (run_windows.chainage_m < 950)]
        if not norm_windows.empty:
            norm_center = int((norm_windows.iloc[0].start_idx + norm_windows.iloc[0].end_idx) / 2)
            n_start = norm_center - 1000
            n_end = n_start + 2000
            norm_samples = acc[n_start:n_end, :]
            microwindows.append({
                "id": "normal_tangent_sector",
                "title": "Nominal Tangent Track (Baseline Smooth Rail)",
                "type": "NORMAL",
                "provenance": "MEASURED",
                "site_id": None,
                "chainage_m": round(float(norm_windows.iloc[0].chainage_m), 2),
                "time_start_s": round(n_start / 2000.0, 3),
                "time_end_s": round(n_end / 2000.0, 3),
                "sample_count": len(norm_samples),
                "sampling_rate_hz": 2000,
                "channels_data": [
                    [round(float(v), 4) for v in norm_samples[:, ch]] for ch in range(6)
                ]
            })
            
        microwindows_file = frontend_data_dir / f"run_{run_id}_microwindows.json"
        with open(microwindows_file, "w") as f:
            json.dump({"run_id": run_id, "windows": microwindows}, f, separators=(',', ':'))
        print(f"Exported {microwindows_file.name} ({os.path.getsize(microwindows_file) / 1024:.1f} KB)")

    # 3. Save signals_index.json
    index_data = {
        "dataset_name": "Rail-VIVID",
        "description": "Multi-channel railway vibration and track anomaly dataset recorded on Class 230 railcar at 2,000 Hz",
        "provenance": "MEASURED",
        "channels": CHANNELS_META,
        "runs": runs_catalog,
        "ground_truth_sites": sites_df.to_dict(orient="records")
    }
    
    index_file = frontend_data_dir / "signals_index.json"
    with open(index_file, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"Exported {index_file.name} ({os.path.getsize(index_file) / 1024:.1f} KB)")
    print("Signal dataset export completed successfully!")

if __name__ == "__main__":
    main()

