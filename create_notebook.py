import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Rail-VIVID Exploratory Data Analysis & Quality Audit\n",
                "**Study:** Cross-Run Consistent Multimodal Anomaly Detection with Spatially Held-Out Evaluation on Rail-VIVID  \n",
                "**Companion Paper:** Ma et al. (2026), Nature Scientific Data, DOI: [10.1038/s41597-026-07555-y](https://doi.org/10.1038/s41597-026-07555-y)  \n",
                "**Hardware Environment:** CPU-only (16 GB RAM, 1.1 TB SSD)\n\n",
                "This notebook audits:\n",
                "1. Multimodal time-series inspection (accelerometers, GNSS, environmental data).\n",
                "2. Physical mapping of the 9 documented ground-truth anomalies.\n",
                "3. Vibration signal characteristics across bogie, axle, and car body sensors.\n",
                "4. Spectral analysis (Welch PSD and STFT spectrograms).\n",
                "5. Vision frame rate fluctuations and Laplacian sharpness distributions.\n",
                "6. Cross-run spatial consistency and confounding factor analysis."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import glob\n",
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import scipy.signal as signal\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import cv2\n\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (12, 6)\n",
                "plt.rcParams['font.size'] = 11\n\n",
                "print('Libraries successfully imported.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Dataset Inventory & Documented Ground Truth\n",
                "Loading dataset metadata and the 9 physical track anomalies from Table 2 of Ma et al. (2026)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open('../reports/dataset_inventory.json', 'r') as f:\n",
                "    inventory = json.load(f)\n\n",
                "agg = inventory['aggregate_summary']\n",
                "print(f\"Total runs: {len(inventory['runs'])}\")\n",
                "print(f\"Total 2 kHz time-series rows: {agg['total_csv_rows']:,}\")\n",
                "print(f\"Total vision frames: {agg['total_images']:,}\")\n\n",
                "# Documented ground-truth anomalies (Table 2)\n",
                "anomalies = [\n",
                "    {'id': 1, 'type': 'Joints', 'size_cm': 2.0, 'lat': 40.23553, 'lon': -77.88898},\n",
                "    {'id': 2, 'type': 'Joints', 'size_cm': 2.0, 'lat': 40.23512, 'lon': -77.88770},\n",
                "    {'id': 3, 'type': 'Joints', 'size_cm': 2.4, 'lat': 40.23493, 'lon': -77.88711},\n",
                "    {'id': 4, 'type': 'Divergence', 'size_cm': np.nan, 'lat': 40.23461, 'lon': -77.88612},\n",
                "    {'id': 5, 'type': 'Convergence', 'size_cm': np.nan, 'lat': 40.23425, 'lon': -77.88485},\n",
                "    {'id': 6, 'type': 'Joints', 'size_cm': 2.0, 'lat': 40.23388, 'lon': -77.88320},\n",
                "    {'id': 7, 'type': 'Joints', 'size_cm': 2.5, 'lat': 40.23242, 'lon': -77.87795},\n",
                "    {'id': 8, 'type': 'Joints', 'size_cm': 2.3, 'lat': 40.23205, 'lon': -77.87651},\n",
                "    {'id': 9, 'type': 'Joints', 'size_cm': 3.0, 'lat': 40.23194, 'lon': -77.87508},\n",
                "]\n",
                "df_anom = pd.DataFrame(anomalies)\n",
                "df_anom"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Tabular Data Inspection & Missingness Check\n",
                "Inspect a sample run (`AtoB_20_1.csv`) to verify column names, sampling rates, and missing values."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_sample = pd.read_csv('../AtoB_20_1/AtoB_20_1.csv', nrows=10000)\n",
                "print('Shape of sample:', df_sample.shape)\n",
                "print('Columns:', df_sample.columns.tolist())\n",
                "print('Missing values per column:\\n', df_sample.isna().sum())\n",
                "df_sample.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Acceleration Response Across Sensor Locations\n",
                "Compare unsprung mass (Axle - Ch 2/4), primary suspension (Bogie - Ch 1/3), and secondary suspension (Car Body - Ch 5/6)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ch_cols = [f'Channel{i}' for i in range(1, 7)]\n",
                "df_sample[ch_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Key Visualizations from the EDA Audit\n",
                "Displaying publication-quality visualizations generated during the audit."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import Image, display\n\n",
                "fig_files = [\n",
                "    '../figures/01_gps_trajectories_and_anomalies.png',\n",
                "    '../figures/02_acceleration_timeseries_channels.png',\n",
                "    '../figures/03_acceleration_distributions.png',\n",
                "    '../figures/04_psd_spectrogram.png',\n",
                "    '../figures/05_environmental_distributions.png',\n",
                "    '../figures/06_cross_run_consistency_and_speed.png',\n",
                "    '../figures/07_camera_frame_rate_and_sharpness.png'\n",
                "]\n\n",
                "for f in fig_files:\n",
                "    if os.path.exists(f):\n",
                "        print('Displaying:', f)\n",
                "        display(Image(filename=f, width=800))"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/01_dataset_exploration.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print('notebooks/01_dataset_exploration.ipynb created successfully.')
