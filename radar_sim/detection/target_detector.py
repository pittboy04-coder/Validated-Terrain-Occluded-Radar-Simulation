"""Detect targets from radar sweep data using peak detection."""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Detection:
    """A single radar detection (echo peak)."""
    bearing_deg: float
    range_ratio: float  # 0.0 to 1.0 (relative to max range)
    intensity: float    # 0.0 to 1.0
    width_bins: int     # Width of the detection in range bins


class TargetDetector:
    """Detects targets from radar sweep data using peak detection.

    Uses adaptive thresholding and peak finding to identify echoes
    that are likely to be real targets vs. noise/clutter.
    """

    def __init__(self,
                 min_intensity: float = 0.15,
                 min_width: int = 2,
                 max_width: int = 50,
                 merge_threshold: int = 3):
        """Initialize detector with threshold parameters.

        Args:
            min_intensity: Minimum echo intensity to consider (0-1).
            min_width: Minimum detection width in range bins.
            max_width: Maximum detection width (larger = clutter/land).
            merge_threshold: Bins to merge adjacent detections.
        """
        self.min_intensity = min_intensity
        self.min_width = min_width
        self.max_width = max_width
        self.merge_threshold = merge_threshold

    def detect_sweep(self, bearing_deg: float,
                      echoes: List[float]) -> List[Detection]:
        """Find target detections in a single sweep.

        Args:
            bearing_deg: Bearing of this sweep in degrees.
            echoes: List of echo intensities (0-1) from near to far.

        Returns:
            List of Detection objects found in this sweep.
        """
        if not echoes:
            return []

        num_bins = len(echoes)
        detections = []

        # Find runs of above-threshold values
        in_target = False
        start_bin = 0
        peak_intensity = 0.0

        for i, intensity in enumerate(echoes):
            if intensity >= self.min_intensity:
                if not in_target:
                    # Start of new potential target
                    in_target = True
                    start_bin = i
                    peak_intensity = intensity
                else:
                    # Continue target, track peak
                    peak_intensity = max(peak_intensity, intensity)
            else:
                if in_target:
                    # End of target
                    width = i - start_bin
                    if self.min_width <= width <= self.max_width:
                        # Valid target detection
                        center_bin = start_bin + width // 2
                        range_ratio = (center_bin + 0.5) / num_bins
                        detections.append(Detection(
                            bearing_deg=bearing_deg,
                            range_ratio=range_ratio,
                            intensity=peak_intensity,
                            width_bins=width
                        ))
                    in_target = False

        # Handle target extending to end of sweep
        if in_target:
            width = num_bins - start_bin
            if self.min_width <= width <= self.max_width:
                center_bin = start_bin + width // 2
                range_ratio = (center_bin + 0.5) / num_bins
                detections.append(Detection(
                    bearing_deg=bearing_deg,
                    range_ratio=range_ratio,
                    intensity=peak_intensity,
                    width_bins=width
                ))

        return detections

    def detect_multiple_sweeps(self,
                                sweeps: List[Tuple[float, List[float]]]) -> List[Detection]:
        """Find detections across multiple sweeps.

        Args:
            sweeps: List of (bearing_deg, echoes) tuples.

        Returns:
            List of all Detection objects found.
        """
        all_detections = []
        for bearing, echoes in sweeps:
            all_detections.extend(self.detect_sweep(bearing, echoes))
        return all_detections
