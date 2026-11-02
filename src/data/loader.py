"""
Data Loading Utilities for Rail-VIVID Multimodal Dataset

Provides memory-efficient streaming, chunked access, and lazy image loading.
"""

import os
import glob
import cv2
import pandas as pd
import numpy as np
from typing import Generator, Dict, List, Optional, Tuple


def get_available_runs(root_dir: str = '.') -> List[str]:
    """Returns sorted list of all 20 run directory names."""
    return sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and (d.startswith('AtoB_') or d.startswith('BtoA_'))
    ])


def stream_run_csv(
    csv_path: str,
    chunk_size: int = 50000,
    columns: Optional[List[str]] = None
) -> Generator[pd.DataFrame, None, None]:
    """
    Streams a large 2 kHz run CSV file in manageable chunks to avoid high RAM usage.
    """
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=columns):
        yield chunk


def load_frame_image(image_path: str, grayscale: bool = True, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """
    Lazily loads a single high-resolution image frame from disk.
    Optionally resizes and converts to grayscale to minimize memory usage.
    """
    if not os.path.exists(image_path):
        return None

    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(image_path, flags)
    if img is not None and target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def get_run_image_paths(run_dir: str) -> List[str]:
    """Returns sorted list of image file paths for a given run."""
    subdirs = [s for s in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, s)) and s.startswith('Frames_')]
    if not subdirs:
        return []
    frame_dir = os.path.join(run_dir, subdirs[0])
    return sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))

