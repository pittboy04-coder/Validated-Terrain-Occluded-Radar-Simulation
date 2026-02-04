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
    5. Range (close targets are often clutter)

    Error Prevention:
    - Land vs vessel: Land has very stable position, extended size, high hits
    - Clutter vs target: Clutter has inconsistent detections, low hit count
    - Size confusion: Uses bin count not meters (consistent across ranges)
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

        # Minimum range to consider (filters close-range clutter)
        self.min_range_ratio = 0.02  # ~0.12nm at 6nm range

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

        Uses a hierarchical decision tree with priority rules to avoid
        misclassification from overlapping feature ranges.

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

        # Filter close-range clutter (typically own-ship noise)
        if features.range_ratio < self.min_range_ratio:
            return TargetClass.UNKNOWN, 0.2

        # Use bin-based size (more consistent across ranges)
        size_bins = features.range_extent_bins
        intensity = features.peak_intensity
        stability = min(1.0, features.detection_count / 10.0)
        bearing_extent = features.bearing_extent_deg

        # Position stability metrics
        # Land: very stable (spread < 0.5°, range_spread < 0.003)
        # Vessels: moderate variation (spread 0.5-3°, range_spread 0.005-0.02)
        very_stable = features.range_spread < 0.004 and features.bearing_spread < 0.6
        is_moving = features.range_spread > 0.008 or features.bearing_spread > 1.2

        # ========== HIERARCHICAL CLASSIFICATION ==========
        # Uses a priority-based decision tree where unique features
        # (stability, intensity, bearing extent) determine class FIRST,
        # before falling through to size-based categories.

        # PRIORITY 1: LAND - Extremely stable position (unique feature)
        # Land is distinguished by near-zero position variation - no vessel
        # is this stable, not even at anchor.
        if size_bins >= 20 and very_stable:
            stability_score = (1.0 - min(1.0, features.bearing_spread / 0.6)) * \
                              (1.0 - min(1.0, features.range_spread / 0.004))
            conf = 0.70 + 0.30 * stability_score
            return TargetClass.LAND, conf

        # PRIORITY 2: BUOY - Very small (<=4 bins) AND point-like (unique feature)
        # Buoys are tiny navigation markers with minimal bearing extent
        # Check BEFORE sailing to catch small point targets
        if size_bins <= 4 and bearing_extent < 1.8:
            size_score = max(0, 1.0 - abs(size_bins - 3) / 2)
            point_score = max(0, 1.0 - bearing_extent / 1.8)
            conf = 0.62 + 0.20 * size_score + 0.18 * point_score
            return TargetClass.BUOY, conf

        # PRIORITY 3: SAILING - Low intensity (unique feature)
        # Fiberglass/wood hulls have distinctly weak radar returns
        # Upper threshold set to 0.44 to catch intensity noise
        if 3 <= size_bins <= 12 and intensity < 0.44:
            low_int_score = max(0, 1.0 - intensity / 0.44)
            size_score = max(0, 1.0 - abs(size_bins - 6) / 6)
            conf = 0.58 + 0.27 * low_int_score + 0.15 * size_score
            return TargetClass.SAILING, conf

        # PRIORITY 4: TANKER - Very large with very strong return (unique feature)
        # Only the largest vessels (VLCC, tankers) exceed 35 bins with high intensity
        if size_bins >= 35 and intensity > 0.70:
            size_score = min(1.0, (size_bins - 32) / 20)
            int_score = min(1.0, (intensity - 0.65) / 0.25)
            conf = 0.65 + 0.20 * size_score + 0.15 * int_score
            return TargetClass.TANKER, conf

        # PRIORITY 5: PILOT - Small (3-7 bins), moving, strong return
        # Fast patrol/pilot boats are small with strong metal hull returns
        # Intensity > 0.45 to avoid catching sailing boats
        if 3 <= size_bins <= 7 and intensity > 0.45 and is_moving:
            size_score = max(0, 1.0 - abs(size_bins - 5) / 4)
            move_score = min(1.0, features.range_spread / 0.014)
            conf = 0.58 + 0.22 * size_score + 0.20 * move_score
            return TargetClass.PILOT, conf

        # PRIORITY 6: FISHING vs TUG - Medium size vessels (6-17 bins)
        # Differentiate by intensity: fishing < 0.60, tug >= 0.60
        if 6 <= size_bins <= 17:
            if intensity < 0.60:  # Moderate intensity = fishing
                size_score = max(0, 1.0 - abs(size_bins - 11) / 8)
                conf = 0.56 + 0.24 * size_score + 0.20 * stability
                return TargetClass.FISHING, conf
            else:  # High intensity = tug (metal hull)
                size_score = max(0, 1.0 - abs(size_bins - 14) / 6)
                conf = 0.56 + 0.24 * size_score + 0.20 * min(1.0, (intensity - 0.55) / 0.25)
                return TargetClass.TUG, conf

        # PRIORITY 7: PASSENGER - Large (18-45 bins) with extended bearing
        # Ferries/cruise ships have high superstructures = wide angular extent
        # bearing_extent threshold lowered to 3.2 to catch ferry signatures
        if 18 <= size_bins <= 48 and bearing_extent >= 3.2:
            extent_score = min(1.0, (bearing_extent - 2.5) / 5.0)
            size_score = max(0, 1.0 - abs(size_bins - 30) / 18)
            conf = 0.58 + 0.27 * extent_score + 0.15 * size_score
            return TargetClass.PASSENGER, conf

        # PRIORITY 8: CARGO - Large vessel (18+ bins)
        # Generic large vessel classification
        if size_bins >= 18 and intensity > 0.50:
            size_score = min(1.0, (size_bins - 15) / 20)
            conf = 0.60 + 0.20 * size_score + 0.20 * min(1.0, (intensity - 0.45) / 0.35)
            return TargetClass.CARGO, conf

        # PRIORITY 9: TUG - Medium size (16-20 bins) with strong return
        # Catches tugs that are slightly larger than the 6-16 FISHING/TUG split
        if 16 <= size_bins <= 20 and intensity >= 0.55:
            size_score = max(0, 1.0 - abs(size_bins - 17) / 4)
            conf = 0.55 + 0.25 * size_score + 0.20 * min(1.0, (intensity - 0.50) / 0.25)
            return TargetClass.TUG, conf

        # FALLBACK: Best guess based on size when no specific features match
        if size_bins >= 25:
            return TargetClass.CARGO, 0.40
        elif size_bins >= 12:
            return TargetClass.TUG, 0.35
        elif size_bins >= 6:
            return TargetClass.FISHING, 0.35
        elif intensity < 0.40:
            return TargetClass.SAILING, 0.35
        elif size_bins <= 5:
            return TargetClass.BUOY, 0.35
        else:
            return TargetClass.UNKNOWN, 0.30

    def get_label(self, tracked_target, include_confidence: bool = True) -> str:
        """Get display label for a tracked target.

        Args:
            tracked_target: TrackedTarget object.
            include_confidence: If True, append confidence percentage to label.

        Returns:
            String label like "T1:CARGO 87%" or "T3:BUOY 92%".
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

        if include_confidence:
            conf_pct = int(confidence * 100)
            return f"{tracked_target.label}:{class_name} {conf_pct}%"
        else:
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
