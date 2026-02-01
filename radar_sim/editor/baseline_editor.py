"""Baseline editor: load real CSV, overlay synthetic targets, export."""
import csv
import math
import os
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..validation.capture_loader import load_furuno_csv, ANGLE_TICKS_PER_DEGREE


class BaselineEditor:
    """Edit real radar captures by adding/removing synthetic targets."""

    def __init__(self):
        self._baseline: Optional[object] = None  # (360, num_bins) numpy array
        self._modifications: List[dict] = []  # Synthetic target overlays
        self._removals: List[dict] = []  # Zeroed-out regions
        self._source_path: Optional[str] = None
        self._range_scale_m: float = 11112.0  # Default 6 nm

    @property
    def is_loaded(self) -> bool:
        return self._baseline is not None

    @property
    def num_bins(self) -> int:
        if self._baseline is not None:
            return self._baseline.shape[1]
        return 512

    def load_baseline(self, csv_path: str) -> bool:
        """Load a real CSV as the background sweep data.

        Args:
            csv_path: Path to radar_plotter format CSV.

        Returns:
            True if loaded successfully.
        """
        ppi = load_furuno_csv(csv_path)
        if ppi is None:
            return False
        self._baseline = ppi
        self._source_path = csv_path
        self._modifications.clear()
        self._removals.clear()

        # Try to extract range from CSV header
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                first_row = next(reader, None)
                if first_row and len(first_row) > 2:
                    self._range_scale_m = float(first_row[2])
        except (ValueError, IndexError, OSError):
            pass

        return True

    def add_synthetic_target(self, range_m: float, bearing_deg: float,
                             rcs: float, vessel_type: str = "unknown") -> None:
        """Overlay a synthetic radar return at the given position.

        Args:
            range_m: Range to target in meters.
            bearing_deg: Bearing to target in degrees.
            rcs: Radar cross section in m^2.
            vessel_type: Type label for annotation.
        """
        self._modifications.append({
            'range_m': range_m,
            'bearing_deg': bearing_deg % 360,
            'rcs': rcs,
            'vessel_type': vessel_type,
        })

    def remove_region(self, range_range: Tuple[float, float],
                      bearing_range: Tuple[float, float]) -> None:
        """Zero out a region of the baseline data.

        Args:
            range_range: (min_range_m, max_range_m).
            bearing_range: (min_bearing_deg, max_bearing_deg).
        """
        self._removals.append({
            'range_min': range_range[0],
            'range_max': range_range[1],
            'bearing_min': bearing_range[0],
            'bearing_max': bearing_range[1],
        })

    def get_sweep_data(self, bearing_deg: float) -> List[float]:
        """Get combined baseline + modifications for a single bearing.

        Args:
            bearing_deg: Bearing in degrees.

        Returns:
            List of intensity values for each range bin.
        """
        if not HAS_NUMPY or self._baseline is None:
            return [0.0] * 512

        bearing_idx = int(round(bearing_deg)) % 360
        sweep = self._baseline[bearing_idx, :].copy()
        num_bins = len(sweep)
        bin_size = self._range_scale_m / num_bins

        # Apply removals
        for removal in self._removals:
            b_min = removal['bearing_min']
            b_max = removal['bearing_max']
            bearing = bearing_deg % 360
            # Check if this bearing is in the removal range
            if b_min <= b_max:
                in_range = b_min <= bearing <= b_max
            else:
                in_range = bearing >= b_min or bearing <= b_max
            if in_range:
                r_min_bin = max(0, int(removal['range_min'] / bin_size))
                r_max_bin = min(num_bins, int(removal['range_max'] / bin_size) + 1)
                sweep[r_min_bin:r_max_bin] = 0.0

        # Apply synthetic targets
        for mod in self._modifications:
            bearing_diff = abs((bearing_deg - mod['bearing_deg'] + 180) % 360 - 180)
            if bearing_diff > 2.0:
                continue
            beam_factor = max(0.0, 1.0 - bearing_diff / 1.2)
            range_bin = int(mod['range_m'] / bin_size)
            if 0 <= range_bin < num_bins:
                # Simple intensity model from RCS
                intensity = min(1.0, math.log10(max(1.0, mod['rcs'])) / 4.0)
                spread = max(1, int(75 / bin_size))  # ~75m pulse extent
                for di in range(-spread, spread + 1):
                    idx = range_bin + di
                    if 0 <= idx < num_bins:
                        sf = 1.0 - abs(di) / (spread + 1)
                        sweep[idx] = max(sweep[idx], intensity * beam_factor * sf)

        return sweep.tolist()

    def export(self, output_path: str) -> str:
        """Save modified data as a radar_plotter format CSV.

        Args:
            output_path: Output CSV file path.

        Returns:
            Path to the saved file.
        """
        if not HAS_NUMPY or self._baseline is None:
            return ""

        num_bins = self.num_bins
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp', 'unused', 'range_m', 'gain_code', 'angle_ticks']
            for i in range(num_bins):
                header.append(f'bin_{i}')
            writer.writerow(header)

            for bearing_idx in range(360):
                sweep = self.get_sweep_data(float(bearing_idx))
                angle_ticks = bearing_idx * ANGLE_TICKS_PER_DEGREE
                row = [f"{0.0:.3f}", 0, int(self._range_scale_m), 200,
                       f"{angle_ticks:.2f}"]
                for val in sweep:
                    clamped = val if val >= 0.03 else 0.0
                    row.append(f"{clamped:.4f}")
                writer.writerow(row)

        return output_path
