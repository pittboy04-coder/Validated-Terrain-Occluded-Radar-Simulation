"""Classify radar targets based on geometric echo characteristics."""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import math


class TargetClass(Enum):
    """Classification categories matching vessel types in simulator."""
    UNKNOWN = "unknown"
    BUOY = "buoy"           # Small point target
    SAILING = "sailing"     # Small, weak return (fiberglass/wood)
    FISHING = "fishing"     # Small-medium vessel
    PILOT = "pilot"         # Small fast vessel
    TUG = "tug"             # Medium vessel, strong return
    CARGO = "cargo"         # Large vessel
    TANKER = "tanker"       # Very large vessel
    PASSENGER = "passenger" # Large passenger vessel
    LAND = "land"           # Stationary land mass / clutter


# Classification thresholds based on radar characteristics
# Range extent is in bins (typically ~15m per bin at 6nm range with 512 bins)
# Intensity is 0.0-1.0 normalized

@dataclass
class GeometricFeatures:
    """Geometric features extracted from radar echo."""
    range_extent_bins: float      # Width in range direction
    bearing_extent_deg: float     # Width in azimuth direction
    peak_intensity: float         # Maximum echo intensity
    mean_intensity: float         # Average echo intensity
    range_ratio: float            # Distance from radar (0-1)
    detection_count: int          # Number of detections accumulated
    bearing_spread: float         # Std dev of bearing across detections
    range_spread: float           # Std dev of range across detections
    aspect_ratio: float           # bearing_extent / range_extent


class TargetClassifier:
    """Classifies radar targets based on echo geometry.

    Uses a decision tree approach based on:
    1. Size (range and bearing extent)
    2. Intensity (reflectivity indicates material/size)
    3. Stability (consistent detections = real target)
    4. Shape (aspect ratio distinguishes vessel types)
    """

    def __init__(self, range_nm: float = 6.0, num_bins: int = 512):
        """Initialize classifier with radar parameters.

        Args:
            range_nm: Radar range in nautical miles.
            num_bins: Number of range bins per sweep.
        """
        self.range_nm = range_nm
        self.num_bins = num_bins
        self.meters_per_bin = (range_nm * 1852.0) / num_bins

    def set_range(self, range_nm: float) -> None:
        """Update radar range setting."""
        self.range_nm = range_nm
        self.meters_per_bin = (range_nm * 1852.0) / self.num_bins

    def extract_features(self, tracked_target) -> GeometricFeatures:
        """Extract geometric features from a tracked target.

        Args:
            tracked_target: TrackedTarget object with detection history.

        Returns:
            GeometricFeatures dataclass with extracted measurements.
        """
        detections = tracked_target.detections

        if not detections:
            return GeometricFeatures(
                range_extent_bins=0, bearing_extent_deg=0,
                peak_intensity=0, mean_intensity=0, range_ratio=0,
                detection_count=0, bearing_spread=0, range_spread=0,
                aspect_ratio=1.0
            )

        # Collect measurements from detection history
        widths = [d.width_bins for d in detections]
        intensities = [d.intensity for d in detections]
        bearings = [d.bearing_deg for d in detections]
        ranges = [d.range_ratio for d in detections]

        # Calculate statistics
        avg_width = sum(widths) / len(widths)
        peak_intensity = max(intensities)
        mean_intensity = sum(intensities) / len(intensities)

        # Bearing spread (handling wraparound)
        bearing_spread = self._circular_std(bearings)

        # Range spread
        range_spread = self._std(ranges) if len(ranges) > 1 else 0.0

        # Estimate bearing extent from spread across detections
        # Each detection covers ~0.5-1° of beam, spread indicates target width
        bearing_extent = max(1.0, bearing_spread * 2 + 1.0)

        # Aspect ratio: angular size vs radial size
        # Convert range bins to approximate angular equivalent
        range_m = tracked_target.range_ratio * self.range_nm * 1852.0
        if range_m > 0:
            range_extent_m = avg_width * self.meters_per_bin
            range_extent_deg = math.degrees(range_extent_m / range_m)
            aspect_ratio = bearing_extent / max(0.1, range_extent_deg)
        else:
            aspect_ratio = 1.0

        return GeometricFeatures(
            range_extent_bins=avg_width,
            bearing_extent_deg=bearing_extent,
            peak_intensity=peak_intensity,
            mean_intensity=mean_intensity,
            range_ratio=tracked_target.range_ratio,
            detection_count=len(detections),
            bearing_spread=bearing_spread,
            range_spread=range_spread,
            aspect_ratio=aspect_ratio
        )

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    def _circular_std(self, angles_deg: List[float]) -> float:
        """Calculate circular standard deviation for angles."""
        if len(angles_deg) < 2:
            return 0.0

        # Convert to radians and use circular statistics
        angles_rad = [math.radians(a) for a in angles_deg]
        sin_sum = sum(math.sin(a) for a in angles_rad)
        cos_sum = sum(math.cos(a) for a in angles_rad)
        n = len(angles_rad)

        # Mean resultant length
        r = math.sqrt(sin_sum**2 + cos_sum**2) / n

        # Circular standard deviation
        if r >= 1.0:
            return 0.0
        return math.degrees(math.sqrt(-2.0 * math.log(r)))

    def classify(self, tracked_target) -> Tuple[TargetClass, float]:
        """Classify a tracked target based on its geometric features.

        Classification uses range extent in BINS (not meters) since radar
        resolution is bin-based. Typical values at 6nm/512 bins (~22m/bin):
        - Buoy: 2-4 bins
        - Small vessel: 3-8 bins
        - Medium vessel: 6-15 bins
        - Large vessel: 12-30 bins
        - Very large: 20-50+ bins

        Args:
            tracked_target: TrackedTarget object with detection history.

        Returns:
            Tuple of (TargetClass, confidence) where confidence is 0.0-1.0.
        """
        features = self.extract_features(tracked_target)

        # Need minimum detections for reliable classification
        if features.detection_count < 3:
            return TargetClass.UNKNOWN, 0.0

        # Use bin-based size (more consistent across ranges)
        size_bins = features.range_extent_bins
        intensity = features.peak_intensity
        stability = min(1.0, features.detection_count / 10.0)
        bearing_extent = features.bearing_extent_deg

        # Position stability (land is very stable)
        position_stable = features.range_spread < 0.01 and features.bearing_spread < 2.0

        # Decision tree classification based on bin size
        scores = {}

        # BUOY: Very small (2-5 bins), point-like, any intensity
        if size_bins <= 6:
            score = (
                0.4 * max(0, 1.0 - abs(size_bins - 3) / 3) +
                0.3 * (1.0 if bearing_extent < 3 else 0.5) +
                0.3 * stability
            )
            if score > 0.3:
                scores[TargetClass.BUOY] = score

        # SAILING: Small (3-10 bins), weak return (fiberglass)
        if 3 <= size_bins <= 12 and intensity < 0.45:
            score = (
                0.35 * max(0, 1.0 - abs(size_bins - 6) / 6) +
                0.35 * (1.0 - intensity / 0.45) +
                0.3 * stability
            )
            if score > 0.3:
                scores[TargetClass.SAILING] = score

        # FISHING: Small-medium (4-15 bins), moderate return
        if 4 <= size_bins <= 18 and 0.25 < intensity < 0.6:
            score = (
                0.35 * max(0, 1.0 - abs(size_bins - 10) / 10) +
                0.35 * max(0, 1.0 - abs(intensity - 0.4) / 0.25) +
                0.3 * stability
            )
            if score > 0.3:
                scores[TargetClass.FISHING] = score

        # PILOT: Small (3-8 bins), medium-strong return, moving
        if 3 <= size_bins <= 10 and intensity > 0.35:
            move_indicator = 1.0 if features.range_spread > 0.005 else 0.6
            score = (
                0.3 * max(0, 1.0 - abs(size_bins - 5) / 5) +
                0.3 * min(1.0, intensity / 0.5) +
                0.2 * stability +
                0.2 * move_indicator
            )
            if score > 0.3:
                scores[TargetClass.PILOT] = score

        # TUG: Medium (6-18 bins), strong return (metal hull)
        if 6 <= size_bins <= 20 and intensity > 0.45:
            score = (
                0.35 * max(0, 1.0 - abs(size_bins - 12) / 10) +
                0.35 * min(1.0, (intensity - 0.3) / 0.4) +
                0.3 * stability
            )
            if score > 0.3:
                scores[TargetClass.TUG] = score

        # CARGO: Large (12-35 bins), strong return
        # Upper limit prevents overlap with tanker
        if 10 <= size_bins <= 38 and intensity > 0.5:
            # Penalize if too large (likely tanker)
            size_penalty = max(0, (size_bins - 30) / 10) if size_bins > 30 else 0
            score = (
                0.35 * min(1.0, size_bins / 25) +
                0.35 * min(1.0, (intensity - 0.4) / 0.4) +
                0.3 * stability
            ) - size_penalty * 0.3
            if score > 0.3:
                scores[TargetClass.CARGO] = score

        # TANKER: Very large (35+ bins), very strong return
        # Tankers are the largest vessels with strongest returns
        if size_bins >= 32 and intensity > 0.70:
            score = (
                0.5 * min(1.0, (size_bins - 30) / 15) +
                0.3 * min(1.0, (intensity - 0.65) / 0.25) +
                0.2 * stability
            )
            if score > 0.4:
                scores[TargetClass.TANKER] = score

        # PASSENGER: Medium-large (15-35 bins), extended in bearing (ferry superstructure)
        # Distinguish from cargo by requiring bearing spread (superstructure)
        if 14 <= size_bins <= 40 and intensity > 0.5 and bearing_extent > 4.5:
            score = (
                0.25 * max(0, 1.0 - abs(size_bins - 25) / 15) +
                0.25 * min(1.0, (intensity - 0.4) / 0.4) +
                0.3 * min(1.0, (bearing_extent - 3) / 8) +
                0.2 * stability
            )
            if score > 0.4:
                scores[TargetClass.PASSENGER] = score

        # LAND: Very large (25+ bins), extremely stable, strong return
        if size_bins >= 20 and position_stable and intensity > 0.5:
            score = (
                0.25 * min(1.0, size_bins / 40) +
                0.35 * (1.0 - features.range_spread * 50) +
                0.2 * (1.0 - features.bearing_spread / 3) +
                0.2 * min(1.0, intensity / 0.7)
            )
            if score > 0.4:
                scores[TargetClass.LAND] = score

        # Select highest scoring class
        if not scores:
            return TargetClass.UNKNOWN, 0.3 * stability

        best_class = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_class])

        return best_class, confidence

    def get_label(self, tracked_target) -> str:
        """Get display label for a tracked target.

        Args:
            tracked_target: TrackedTarget object.

        Returns:
            String label like "T1:CARGO" or "T3:BUOY".
        """
        target_class, confidence = self.classify(tracked_target)

        if target_class == TargetClass.UNKNOWN or confidence < 0.3:
            return tracked_target.label

        # Short class names for display
        class_names = {
            TargetClass.BUOY: "BUOY",
            TargetClass.SAILING: "SAIL",
            TargetClass.FISHING: "FISH",
            TargetClass.PILOT: "PILOT",
            TargetClass.TUG: "TUG",
            TargetClass.CARGO: "CARGO",
            TargetClass.TANKER: "TANK",
            TargetClass.PASSENGER: "FERRY",
            TargetClass.LAND: "LAND",
        }

        class_name = class_names.get(target_class, "UNK")
        return f"{tracked_target.label}:{class_name}"

    def get_detailed_info(self, tracked_target) -> dict:
        """Get detailed classification info for a target.

        Args:
            tracked_target: TrackedTarget object.

        Returns:
            Dictionary with classification details.
        """
        features = self.extract_features(tracked_target)
        target_class, confidence = self.classify(tracked_target)

        return {
            'id': tracked_target.id,
            'label': tracked_target.label,
            'class': target_class.value,
            'confidence': confidence,
            'bearing_deg': tracked_target.bearing_deg,
            'range_ratio': tracked_target.range_ratio,
            'range_nm': tracked_target.range_ratio * self.range_nm,
            'features': {
                'size_bins': features.range_extent_bins,
                'size_m': features.range_extent_bins * self.meters_per_bin,
                'bearing_extent_deg': features.bearing_extent_deg,
                'peak_intensity': features.peak_intensity,
                'mean_intensity': features.mean_intensity,
                'detection_count': features.detection_count,
                'bearing_spread': features.bearing_spread,
                'range_spread': features.range_spread,
            }
        }
