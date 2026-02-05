"""Capture analyzer: auto-configure simulator from real radar CSV captures.

Analyzes real radar captures to extract settings (gain, clutter), detect
persistent objects, and optionally load location data from GPS metadata.
"""

import csv
import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class CaptureMetadata:
    """Extracted metadata from a radar capture analysis."""
    # Settings
    range_nm: float = 6.0
    gain: float = 0.5
    sea_clutter: float = 0.3
    rain_clutter: float = 0.3

    # Detected objects (range_m, bearing_deg, estimated_rcs)
    detected_objects: List[Tuple[float, float, float]] = field(default_factory=list)

    # Location (if GPS data present)
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    location_name: Optional[str] = None

    # Source info
    source_folder: str = ""
    num_sweeps_analyzed: int = 0


class CaptureAnalyzer:
    """Analyze real radar CSV captures to auto-configure the simulator."""

    def __init__(self):
        self._ppi_data: Optional[np.ndarray] = None
        self._accumulated: Optional[np.ndarray] = None
        self._range_m: float = 11112.0  # Default 6nm
        self._num_sweeps: int = 0
        self._metadata_cache: Dict[str, Any] = {}

    def analyze_csv_folder(self, folder_path: str) -> Optional[CaptureMetadata]:
        """Analyze all CSV files in a folder to extract capture metadata.

        Args:
            folder_path: Path to folder containing radar CSV files.

        Returns:
            CaptureMetadata with extracted settings and objects, or None on failure.
        """
        if not HAS_NUMPY:
            print("numpy required for capture analysis")
            return None

        if not os.path.isdir(folder_path):
            print(f"Not a directory: {folder_path}")
            return None

        # Find all CSV files
        csv_files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith('.csv')
        ])

        if not csv_files:
            print("No CSV files found in folder")
            return None

        # Initialize accumulation arrays
        self._accumulated = None
        self._num_sweeps = 0
        self._metadata_cache = {}

        # Process each file
        for csv_path in csv_files:
            self._process_csv_file(csv_path)

        if self._accumulated is None or self._num_sweeps == 0:
            return None

        # Build metadata
        metadata = CaptureMetadata()
        metadata.source_folder = folder_path
        metadata.num_sweeps_analyzed = self._num_sweeps

        # Extract settings
        metadata.range_nm = self._range_m / 1852.0
        metadata.gain = self._estimate_gain()
        metadata.sea_clutter = self._estimate_sea_clutter()
        metadata.rain_clutter = self._estimate_rain_clutter()

        # Detect objects
        metadata.detected_objects = self._detect_objects()

        # Check for GPS metadata
        gps = self._extract_gps_metadata(csv_files[0])
        if gps:
            metadata.gps_lat, metadata.gps_lon = gps
            metadata.location_name = self._metadata_cache.get('location_name')

        return metadata

    def _process_csv_file(self, csv_path: str) -> None:
        """Process a single CSV file into the accumulation buffer."""
        from .validation.capture_loader import load_furuno_csv

        ppi = load_furuno_csv(csv_path)
        if ppi is None:
            return

        if self._accumulated is None:
            self._accumulated = ppi.astype(np.float32)
        else:
            # Accumulate maximum values for persistent object detection
            self._accumulated = np.maximum(self._accumulated, ppi)

        self._num_sweeps += 1

        # Extract range from first row
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                first_row = next(reader, None)
                if first_row and len(first_row) > 2:
                    self._range_m = float(first_row[2])
        except (ValueError, IndexError, OSError):
            pass

    def _estimate_gain(self) -> float:
        """Estimate gain setting from average echo intensity.

        Higher average intensity = higher gain setting.
        """
        if self._accumulated is None:
            return 0.5

        # Calculate mean intensity (excluding zeros/noise floor)
        mask = self._accumulated > 0.05
        if not np.any(mask):
            return 0.3

        mean_intensity = np.mean(self._accumulated[mask])

        # Map to 0-1 gain value (typical intensities: 0.1-0.7)
        gain = np.clip((mean_intensity - 0.1) / 0.5, 0.2, 0.9)
        return float(gain)

    def _estimate_sea_clutter(self) -> float:
        """Estimate sea clutter setting from near-range noise pattern.

        High clutter = more returns in first 1nm (1852m).
        """
        if self._accumulated is None:
            return 0.3

        num_bins = self._accumulated.shape[1]
        near_range_bins = int(num_bins * 1852.0 / self._range_m)
        near_range_bins = max(1, min(near_range_bins, num_bins // 4))

        # Calculate noise level in near-range bins
        near_range_data = self._accumulated[:, :near_range_bins]
        noise_level = np.mean(near_range_data > 0.1)

        # Map to clutter setting (0 = no clutter visible, 1 = heavy clutter)
        # If >30% of near-range bins have returns, likely high sea clutter
        clutter = np.clip(noise_level / 0.4, 0.1, 0.8)
        return float(clutter)

    def _estimate_rain_clutter(self) -> float:
        """Estimate rain clutter from mid-range diffuse returns.

        Rain creates scattered, non-persistent echoes at medium ranges.
        """
        if self._accumulated is None:
            return 0.3

        num_bins = self._accumulated.shape[1]

        # Mid-range: 1nm to 4nm
        start_bin = int(num_bins * 1852.0 / self._range_m)
        end_bin = int(num_bins * 4 * 1852.0 / self._range_m)
        start_bin = max(1, start_bin)
        end_bin = min(num_bins - 1, end_bin)

        if end_bin <= start_bin:
            return 0.3

        mid_range_data = self._accumulated[:, start_bin:end_bin]

        # Rain creates diffuse, low-intensity returns
        low_intensity_mask = (mid_range_data > 0.05) & (mid_range_data < 0.3)
        diffuse_fraction = np.mean(low_intensity_mask)

        # High diffuse fraction suggests rain
        rain = np.clip(diffuse_fraction / 0.3, 0.1, 0.7)
        return float(rain)

    def _detect_objects(self) -> List[Tuple[float, float, float]]:
        """Detect persistent objects using blob detection.

        Returns:
            List of (range_m, bearing_deg, estimated_rcs) tuples.
        """
        if self._accumulated is None:
            return []

        num_bins = self._accumulated.shape[1]
        bin_size = self._range_m / num_bins

        # Threshold for object detection (persistent strong returns)
        threshold = 0.4 if self._estimate_gain() > 0.5 else 0.3

        # Find connected components of high-intensity returns
        objects = []
        visited = np.zeros_like(self._accumulated, dtype=bool)

        for bearing in range(360):
            for range_bin in range(num_bins):
                if visited[bearing, range_bin]:
                    continue
                if self._accumulated[bearing, range_bin] < threshold:
                    continue

                # Flood-fill to find blob extent
                blob = self._flood_fill(bearing, range_bin, threshold, visited)

                if len(blob) < 3:  # Too small, likely noise
                    continue

                # Calculate blob center and size
                bearings = [b for b, _ in blob]
                ranges = [r for _, r in blob]

                center_bearing = np.mean(bearings) % 360
                center_range_bin = np.mean(ranges)
                center_range_m = center_range_bin * bin_size

                # Estimate RCS from blob intensity and size
                blob_intensities = [self._accumulated[b, r] for b, r in blob]
                mean_intensity = np.mean(blob_intensities)
                blob_size = len(blob)

                # RCS estimation: larger blobs with higher intensity = larger RCS
                # Typical mapping: intensity 0.3-0.8 → RCS 10-1000 m²
                estimated_rcs = blob_size * mean_intensity * 100

                objects.append((center_range_m, center_bearing, estimated_rcs))

        # Merge nearby objects and filter
        merged_objects = self._merge_nearby_objects(objects)

        return merged_objects

    def _flood_fill(self, start_bearing: int, start_bin: int,
                    threshold: float, visited: np.ndarray) -> List[Tuple[int, int]]:
        """Flood-fill to find connected blob cells."""
        blob = []
        stack = [(start_bearing, start_bin)]
        num_bins = self._accumulated.shape[1]

        while stack:
            b, r = stack.pop()
            if visited[b, r]:
                continue
            if self._accumulated[b, r] < threshold:
                continue

            visited[b, r] = True
            blob.append((b, r))

            # Check neighbors (4-connected)
            for db, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (b + db) % 360
                nr = r + dr
                if 0 <= nr < num_bins and not visited[nb, nr]:
                    stack.append((nb, nr))

        return blob

    def _merge_nearby_objects(self, objects: List[Tuple[float, float, float]],
                               range_threshold: float = 200.0,
                               bearing_threshold: float = 5.0) -> List[Tuple[float, float, float]]:
        """Merge objects that are close together."""
        if not objects:
            return []

        merged = []
        used = set()

        for i, (r1, b1, rcs1) in enumerate(objects):
            if i in used:
                continue

            # Find nearby objects to merge
            group = [(r1, b1, rcs1)]
            used.add(i)

            for j, (r2, b2, rcs2) in enumerate(objects):
                if j in used:
                    continue

                # Check proximity
                range_diff = abs(r2 - r1)
                bearing_diff = min(abs(b2 - b1), 360 - abs(b2 - b1))

                if range_diff < range_threshold and bearing_diff < bearing_threshold:
                    group.append((r2, b2, rcs2))
                    used.add(j)

            # Merge group into single object (weighted average)
            total_rcs = sum(rcs for _, _, rcs in group)
            if total_rcs > 0:
                avg_range = sum(r * rcs for r, _, rcs in group) / total_rcs
                avg_bearing = sum(b * rcs for _, b, rcs in group) / total_rcs
                merged.append((avg_range, avg_bearing % 360, total_rcs))

        # Sort by range
        merged.sort(key=lambda x: x[0])

        return merged

    def _extract_gps_metadata(self, csv_path: str) -> Optional[Tuple[float, float]]:
        """Extract GPS coordinates from CSV metadata header if present.

        Looks for comment lines like:
        # GPS_LAT: 34.0818
        # GPS_LON: -81.2169
        """
        try:
            with open(csv_path, 'r') as f:
                lines = f.readlines()

            lat = None
            lon = None

            for line in lines[:20]:  # Check first 20 lines for metadata
                line = line.strip()

                # Check for GPS_LAT/GPS_LON format
                lat_match = re.match(r'#?\s*GPS_LAT[:\s]+(-?\d+\.?\d*)', line, re.IGNORECASE)
                if lat_match:
                    lat = float(lat_match.group(1))

                lon_match = re.match(r'#?\s*GPS_LON[:\s]+(-?\d+\.?\d*)', line, re.IGNORECASE)
                if lon_match:
                    lon = float(lon_match.group(1))

                # Check for location name
                name_match = re.match(r'#?\s*LOCATION[:\s]+(.+)', line, re.IGNORECASE)
                if name_match:
                    self._metadata_cache['location_name'] = name_match.group(1).strip()

                # Check for RANGE_NM
                range_match = re.match(r'#?\s*RANGE_NM[:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
                if range_match:
                    self._range_m = float(range_match.group(1)) * 1852.0

                # Check for GAIN
                gain_match = re.match(r'#?\s*GAIN[:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
                if gain_match:
                    self._metadata_cache['gain'] = float(gain_match.group(1))

            if lat is not None and lon is not None:
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)

        except (IOError, ValueError):
            pass

        return None


def analyze_capture_folder(folder_path: str) -> Optional[CaptureMetadata]:
    """Convenience function to analyze a capture folder.

    Args:
        folder_path: Path to folder containing radar CSV files.

    Returns:
        CaptureMetadata with extracted settings and objects.
    """
    analyzer = CaptureAnalyzer()
    return analyzer.analyze_csv_folder(folder_path)
