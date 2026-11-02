import os
import glob
import json
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

os.makedirs('figures', exist_ok=True)
sns.set_theme(style='whitegrid', font_scale=1.1)

# Ground truth anomalies
anomalies = [
    {'id': 1, 'type': 'Joints (2.0cm)', 'lat': 40.23553, 'lon': -77.88898},
    {'id': 2, 'type': 'Joints (2.0cm)', 'lat': 40.23512, 'lon': -77.88770},
    {'id': 3, 'type': 'Joints (2.4cm)', 'lat': 40.23493, 'lon': -77.88711},
    {'id': 4, 'type': 'Divergence', 'lat': 40.23461, 'lon': -77.88612},
    {'id': 5, 'type': 'Convergence', 'lat': 40.23425, 'lon': -77.88485},
    {'id': 6, 'type': 'Joints (2.0cm)', 'lat': 40.23388, 'lon': -77.88320},
    {'id': 7, 'type': 'Joints (2.5cm)', 'lat': 40.23242, 'lon': -77.87795},
    {'id': 8, 'type': 'Joints (2.3cm)', 'lat': 40.23205, 'lon': -77.87651},
    {'id': 9, 'type': 'Joints (3.0cm)', 'lat': 40.23194, 'lon': -77.87508},
]
df_anom = pd.DataFrame(anomalies)

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(delta_lambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

print('Generating Figure 1: GPS Trajectories and Anomalies...')
# 1. GPS Trajectories
plt.figure(figsize=(12, 7))
sample_runs = ['AtoB_20_1', 'AtoB_22.5_1', 'BtoA_22.5_1', 'BtoA_25_1', 'BtoA_27.5_1']
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

for r, c in zip(sample_runs, colors):
    csv_path = os.path.join(r, f'{r}.csv')
    df_gps = pd.read_csv(csv_path, usecols=['Latitude', 'Longitude'])[::200]
    plt.plot(df_gps['Longitude'], df_gps['Latitude'], label=r, color=c, alpha=0.6, linewidth=1.5)

# Plot anomalies
for _, a in df_anom.iterrows():
    marker = 'X' if 'Joints' in a['type'] else 'D'
    mcolor = 'crimson' if 'Joints' in a['type'] else 'black'
    plt.scatter(a['lon'], a['lat'], color=mcolor, s=120, zorder=5, marker=marker)
    plt.annotate(f"A{a['id']}: {a['type']}", (a['lon'], a['lat']),
                 textcoords='offset points', xytext=(5, 5), fontsize=9,
                 weight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))

# Endpoints Point A & B
pt_A = (40.23951169072744, -77.89729802853955)
pt_B = (40.23184123709821, -77.87387005111827)
plt.scatter([pt_A[1]], [pt_A[0]], color='darkgreen', s=160, marker='o', label='Point A (Orbisonia)', zorder=6)
plt.scatter([pt_B[1]], [pt_B[0]], color='darkblue', s=160, marker='s', label='Point B (Terminus)', zorder=6)

plt.xlabel('Longitude (Degrees East)')
plt.ylabel('Latitude (Degrees North)')
plt.title('Rail-VIVID GPS Track Trajectories and 9 Ground-Truth Anomaly Sites', fontsize=14, weight='bold')
plt.legend(loc='lower left', framealpha=0.9)
plt.tight_layout()
plt.savefig('figures/01_gps_trajectories_and_anomalies.png', dpi=300)
plt.close()

print('Generating Figure 2: Acceleration Time-Series Across 6 Channels...')
# 2. Time-Series across 6 channels around Anomaly 1
df_run = pd.read_csv('AtoB_20_1/AtoB_20_1.csv')
# find index closest to Anomaly 1
dists_a1 = haversine_np(df_run['Latitude'].values, df_run['Longitude'].values, anomalies[0]['lat'], anomalies[0]['lon'])
idx_a1 = np.argmin(dists_a1)

# window: +/- 2 seconds (4000 samples)
win_start = max(0, idx_a1 - 4000)
win_end = min(len(df_run), idx_a1 + 4000)
df_win = df_run.iloc[win_start:win_end].copy()
time_axis = np.arange(len(df_win)) / 2000.0 - 2.0  # seconds relative to anomaly

fig, axes = plt.subplots(6, 1, figsize=(14, 10), sharex=True)
channel_labels = [
    'Ch1: Bogie Front (SD2012)',
    'Ch2: Axle Front (SD2012)',
    'Ch3: Bogie Back (SD2012)',
    'Ch4: Axle Back (SD2012)',
    'Ch5: Car Body Front (SD2012)',
    'Ch6: Car Body Mid (SD2012)'
]
ch_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

for i in range(6):
    ch_col = f'Channel{i+1}'
    axes[i].plot(time_axis, df_win[ch_col], color=ch_colors[i], linewidth=0.8)
    axes[i].axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    axes[i].set_ylabel('Accel (g)', fontsize=9)
    axes[i].set_title(channel_labels[i], fontsize=10, loc='left', weight='bold')
    axes[i].grid(True, alpha=0.5)

axes[0].annotate('Anomaly 1 (2.0 cm Joint Gap)', xy=(0, axes[0].get_ylim()[1]*0.7),
                 xytext=(0.5, axes[0].get_ylim()[1]*0.7),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, weight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.8))

axes[-1].set_xlabel('Time Relative to Closest Anomaly Approach (Seconds)')
fig.suptitle('Multichannel Vibration Response (2 kHz) Transiting Anomaly 1 (AtoB_20_1)', fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('figures/02_acceleration_timeseries_channels.png', dpi=300)
plt.close()

print('Generating Figure 3: Acceleration Distributions and Sensor Location Effects...')
# 3. Acceleration Distributions
ch_cols = [f'Channel{i}' for i in range(1, 7)]
df_sample = df_run[ch_cols].sample(n=50000, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Boxplot
sns.boxplot(data=df_sample, ax=axes[0], palette=ch_colors)
axes[0].set_ylabel('Acceleration (g)')
axes[0].set_title('Dynamic Amplitude Range by Sensor Channel', weight='bold')
axes[0].set_xticklabels(['Ch1 (Bogie)', 'Ch2 (Axle)', 'Ch3 (Bogie)', 'Ch4 (Axle)', 'Ch5 (Body)', 'Ch6 (Body)'])

# Histograms / KDE for Axle (Ch2) vs Bogie (Ch1) vs Body (Ch5)
sns.kdeplot(df_sample['Channel2'], ax=axes[1], color='#d62728', label='Ch2 (Axle Front) - Unsprung', linewidth=2)
sns.kdeplot(df_sample['Channel1'], ax=axes[1], color='#1f77b4', label='Ch1 (Bogie Front) - Primary Susp.', linewidth=2)
sns.kdeplot(df_sample['Channel5'], ax=axes[1], color='#9467bd', label='Ch5 (Car Body) - Secondary Susp.', linewidth=2)
axes[1].set_xlim(-1.5, 1.5)
axes[1].set_xlabel('Acceleration (g)')
axes[1].set_ylabel('Probability Density')
axes[1].set_title('Vibration Attenuation Across Vehicle Suspension Stages', weight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/03_acceleration_distributions.png', dpi=300)
plt.close()

print('Generating Figure 4: Power Spectral Density and Spectrogram...')
# 4. PSD & Spectrogram
fs = 2000.0
# Normal segment (between anom 6 and 7)
dists_a6 = haversine_np(df_run['Latitude'].values, df_run['Longitude'].values, anomalies[5]['lat'], anomalies[5]['lon'])
dists_a7 = haversine_np(df_run['Latitude'].values, df_run['Longitude'].values, anomalies[6]['lat'], anomalies[6]['lon'])
normal_mask = (dists_a6 > 50) & (dists_a7 > 50)
normal_signal = df_run.loc[normal_mask, 'Channel1'].values[:20000]

# Abnormal segment around Anom 1
abnormal_signal = df_run.loc[idx_a1-10000:idx_a1+10000, 'Channel1'].values

f_norm, pxx_norm = signal.welch(normal_signal, fs=fs, nperseg=1024)
f_abn, pxx_abn = signal.welch(abnormal_signal, fs=fs, nperseg=1024)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].semilogy(f_norm, pxx_norm, label='Nominal Track Segment', color='#1f77b4', linewidth=1.5)
axes[0].semilogy(f_abn, pxx_abn, label='Anomaly 1 (Joint Gap)', color='#d62728', linewidth=1.5, alpha=0.85)
axes[0].set_xlim(0, 500)
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel(r'PSD ($g^2$/Hz)')
axes[0].set_title('Welch Power Spectral Density (Channel 1, Bogie)', weight='bold')
axes[0].legend()
axes[0].grid(True, which='both', alpha=0.5)

# Spectrogram of 20 seconds crossing multiple anomalies
seg_spec = df_run.loc[idx_a1-4000:idx_a1+36000, 'Channel1'].values
f_s, t_s, Sxx = signal.spectrogram(seg_spec, fs=fs, nperseg=512, noverlap=256)
im = axes[1].pcolormesh(t_s, f_s, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis', vmax=-10, vmin=-60)
axes[1].set_ylim(0, 500)
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Time (Seconds)')
axes[1].set_title('Short-Time Fourier Transform Spectrogram (Channel 1)', weight='bold')
cbar = fig.colorbar(im, ax=axes[1])
cbar.set_label('Power/Frequency (dB/Hz)')

plt.tight_layout()
plt.savefig('figures/04_psd_spectrogram.png', dpi=300)
plt.close()

print('Generating Figure 5: Environmental Variables Across Collection Days...')
# 5. Environmental Distributions across runs
with open('reports/dataset_inventory.json', 'r') as f:
    inv = json.load(f)

env_data = []
for r_meta in inv['runs']:
    csv_path = r_meta['csv_file']
    sub = pd.read_csv(csv_path, usecols=['Temperature', 'Humidity'])[::5000]
    sub['run_id'] = r_meta['run_id']
    sub['date'] = '2025-04-11' if '2025-04-11' in r_meta['start_timestamp'] else '2025-04-18'
    sub['direction'] = r_meta['direction']
    sub['speed'] = r_meta['target_speed_kmh']
    env_data.append(sub)

df_env = pd.concat(env_data, ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df_env, x='date', y='Temperature', hue='direction', ax=axes[0], palette='Set2')
axes[0].set_title('Ambient Temperature Distribution by Collection Date', weight='bold')
axes[0].set_ylabel('Temperature (°F)')
axes[0].set_xlabel('Collection Date')

sns.boxplot(data=df_env, x='date', y='Humidity', hue='direction', ax=axes[1], palette='Set2')
axes[1].set_title('Relative Humidity Distribution by Collection Date', weight='bold')
axes[1].set_ylabel('Relative Humidity (0.0 to 1.0)')
axes[1].set_xlabel('Collection Date')

plt.tight_layout()
plt.savefig('figures/05_environmental_distributions.png', dpi=300)
plt.close()

print('Generating Figure 6: Cross-Run Consistency Across Speeds...')
# 6. Cross-run consistency: 7.62 m track segmentation standard deviation profile
# Divide track between lat 40.2360 to 40.2319 into bins
lat_bins = np.linspace(40.2319, 40.2360, 160)
comp_runs = ['AtoB_20_1', 'AtoB_20_2', 'AtoB_22.5_1', 'BtoA_22.5_1', 'BtoA_27.5_1']

plt.figure(figsize=(14, 6))
for r in comp_runs:
    df_r = pd.read_csv(os.path.join(r, f'{r}.csv'), usecols=['Latitude', 'Channel1'])
    df_r['bin'] = pd.cut(df_r['Latitude'], bins=lat_bins)
    binned_std = df_r.groupby('bin', observed=False)['Channel1'].std().values
    bin_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    plt.plot(bin_centers, binned_std, label=f'{r}', alpha=0.75, linewidth=1.4)

# Add vertical lines for anomaly latitudes
for _, a in df_anom.iterrows():
    plt.axvline(a['lat'], color='crimson', linestyle=':', alpha=0.7)
    plt.text(a['lat'], plt.ylim()[1]*0.9, f"A{a['id']}", color='crimson', fontsize=9, weight='bold', horizontalalignment='center')

plt.xlabel('Track Latitude (°N, North to South from Left to Right)')
plt.ylabel('Bogie Z-Axis Acceleration STD (g)')
plt.title('Cross-Run Spatial Consistency of Vibration Energy Across 7.62 m Segments', fontsize=13, weight='bold')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('figures/06_cross_run_consistency_and_speed.png', dpi=300)
plt.close()

print('Generating Figure 7: Camera Frame Rates and Image Quality (Laplacian Variance)...')
# 7. Camera Frame Rates and Sharpness
sample_frame_dir = 'AtoB_20_1/Frames_Average20_Effective30'
frames = sorted(glob.glob(f'{sample_frame_dir}/*.jpg'))[:100]

laplacians = []
for f in frames:
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if im is not None:
        lap = cv2.Laplacian(im, cv2.CV_64F).var()
        laplacians.append(lap)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# Frame rate across runs from inventory
fps_list = [r['image_count'] / (r['row_count'] / 2000.0) for r in inv['runs']]
runs_labels = [r['run_id'] for r in inv['runs']]
axes[0].barh(runs_labels, fps_list, color='#1f77b4', alpha=0.8)
axes[0].axvline(100, color='red', linestyle='--', label='Hardware Target (100 FPS)')
axes[0].axvline(np.mean(fps_list), color='green', linestyle='-', label=f'Mean Achieved ({np.mean(fps_list):.1f} FPS)')
axes[0].set_xlabel('Effective Frame Rate (FPS)')
axes[0].set_title('Logging FPS Fluctuation vs Hardware Target', weight='bold')
axes[0].legend(loc='lower right')

# Laplacian variance sharpness distribution
sns.histplot(laplacians, kde=True, ax=axes[1], color='#2ca02c')
axes[1].axvline(np.mean(laplacians), color='darkgreen', linestyle='--', label=f'Sample Mean ({np.mean(laplacians):.1f})')
axes[1].set_xlabel('Laplacian Variance (Sharpness Metric)')
axes[1].set_ylabel('Frame Count')
axes[1].set_title('Image Sharpness Distribution (Variance of Laplacian)', weight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/07_camera_frame_rate_and_sharpness.png', dpi=300)
plt.close()

print('All 7 figures generated successfully.')

