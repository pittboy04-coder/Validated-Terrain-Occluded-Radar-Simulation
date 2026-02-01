"""Load real Furuno CSV captures into numpy arrays for validation."""
import csv
import math
import os
from typing import Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

ANGLE_TICKS_PER_DEGREE = 8192.0 / 360.0


def load_furuno_csv(csv_path: str) -> Optional[object]:
    """Load a radar plotter format CSV into a 360 x N numpy array.

    Args:
        csv_path: Path to CSV file in radar_plotter format.

    Returns:
        numpy array of shape (360, num_bins) or None on failure.
    """
    if not HAS_NUMPY:
        return None

    if not os.path.isfile(csv_path):
        return None

    rows_by_bearing = {}
    num_bins = 0

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return None

        # Determine bin columns (everything after the 5 metadata columns)
        meta_cols = 5
        num_bins = len(header) - meta_cols

        for row in reader:
            if len(row) < meta_cols + 1:
                continue
            try:
                angle_ticks = float(row[4])
                bearing_deg = angle_ticks / ANGLE_TICKS_PER_DEGREE
                bearing_idx = int(round(bearing_deg)) % 360

                values = []
                for j in range(meta_cols, min(len(row), meta_cols + num_bins)):
                    values.append(float(row[j]))
                # Pad if short
                while len(values) < num_bins:
                    values.append(0.0)

                # Keep the latest sweep for each bearing
                rows_by_bearing[bearing_idx] = values
            except (ValueError, IndexError):
                continue

    if not rows_by_bearing or num_bins == 0:
        return None

    # Build 360 x num_bins array
    ppi = np.zeros((360, num_bins), dtype=np.float32)
    for bearing_idx, values in rows_by_bearing.items():
        ppi[bearing_idx, :] = values[:num_bins]

    return ppi


def ppi_to_image(ppi_array, image_size: int = 512) -> Optional[object]:
    """Convert a 360 x N PPI array to a 2D image (image_size x image_size).

    Args:
        ppi_array: numpy array of shape (360, num_bins).
        image_size: Output image dimension.

    Returns:
        numpy array of shape (image_size, image_size) with float values 0-1.
    """
    if not HAS_NUMPY:
        return None

    half = image_size // 2
    num_bins = ppi_array.shape[1]
    image = np.zeros((image_size, image_size), dtype=np.float32)

    for py in range(image_size):
        for px in range(image_size):
            dx = px - half
            dy = py - half
            r = math.sqrt(dx * dx + dy * dy)
            if r > half:
                continue
            bearing = math.degrees(math.atan2(dx, -dy)) % 360
            range_bin = int(r / half * num_bins)
            if range_bin >= num_bins:
                range_bin = num_bins - 1
            bearing_idx = int(round(bearing)) % 360
            image[py, px] = ppi_array[bearing_idx, range_bin]

    return image
