"""Track radar targets across multiple sweeps."""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .target_detector import Detection


@dataclass
class TrackedTarget:
    """A tracked radar target with accumulated detections."""
    id: int
    bearing_deg: float      # Current estimated bearing
    range_ratio: float      # Current estimated range (0-1)
    intensity: float        # Average intensity
    label: str = ""         # Display label (T1, T2, etc.)
    hits: int = 0           # Number of detections
    misses: int = 0         # Consecutive missed detections
    last_update: float = 0  # Last bearing when updated
    detections: List[Detection] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Confidence level based on hit/miss ratio."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    @property
    def is_stable(self) -> bool:
        """True if target has enough hits to be considered stable."""
        return self.hits >= 3 and self.confidence >= 0.5


class TargetTracker:
    """Associates detections across sweeps to form stable tracks.

    Uses a simple nearest-neighbor association with gating based on
    bearing and range differences.
    """

    def __init__(self,
                 bearing_gate_deg: float = 3.0,
                 range_gate_ratio: float = 0.05,
                 max_misses: int = 5,
                 min_hits_for_label: int = 3):
        """Initialize tracker.

        Args:
            bearing_gate_deg: Max bearing difference for association.
            range_gate_ratio: Max range difference (as ratio) for association.
            max_misses: Max consecutive misses before dropping track.
            min_hits_for_label: Hits needed before displaying label.
        """
        self.bearing_gate_deg = bearing_gate_deg
        self.range_gate_ratio = range_gate_ratio
        self.max_misses = max_misses
        self.min_hits_for_label = min_hits_for_label

        self._tracks: Dict[int, TrackedTarget] = {}
        self._next_id = 1
        self._last_bearing = 0.0

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 1
        self._last_bearing = 0.0

    def update(self, detections: List[Detection], current_bearing: float) -> None:
        """Update tracks with new detections.

        Args:
            detections: List of Detection objects from current sweep(s).
            current_bearing: Current radar bearing in degrees.
        """
        # Associate detections with existing tracks
        used_detections = set()

        for track in list(self._tracks.values()):
            best_det = None
            best_dist = float('inf')

            for det in detections:
                if id(det) in used_detections:
                    continue

                # Check if detection is within gate
                bearing_diff = self._bearing_diff(det.bearing_deg, track.bearing_deg)
                range_diff = abs(det.range_ratio - track.range_ratio)

                if bearing_diff <= self.bearing_gate_deg and range_diff <= self.range_gate_ratio:
                    # Combined distance metric
                    dist = math.sqrt((bearing_diff / self.bearing_gate_deg) ** 2 +
                                     (range_diff / self.range_gate_ratio) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_det = det

            if best_det is not None:
                # Update track with new detection
                used_detections.add(id(best_det))
                self._update_track(track, best_det, current_bearing)
            else:
                # Check if we passed over this track's bearing without detecting it
                if self._bearing_passed(track.bearing_deg, self._last_bearing, current_bearing):
                    track.misses += 1
                    if track.misses > self.max_misses:
                        del self._tracks[track.id]

        # Create new tracks for unassociated detections
        for det in detections:
            if id(det) not in used_detections:
                self._create_track(det, current_bearing)

        self._last_bearing = current_bearing

    def _bearing_diff(self, b1: float, b2: float) -> float:
        """Calculate absolute bearing difference, handling wraparound."""
        diff = abs(b1 - b2)
        return min(diff, 360 - diff)

    def _bearing_passed(self, target_bearing: float,
                         old_bearing: float, new_bearing: float) -> bool:
        """Check if radar sweep passed over a bearing.

        Uses exclusive start (old_bearing < target) to avoid double-counting
        when the sweep starts exactly at the target bearing.
        """
        # Handle wraparound
        if old_bearing > new_bearing:
            # Crossed 360/0
            return target_bearing > old_bearing or target_bearing <= new_bearing
        return old_bearing < target_bearing <= new_bearing

    def _update_track(self, track: TrackedTarget, det: Detection,
                       current_bearing: float) -> None:
        """Update track with new detection."""
        # Exponential moving average for position
        alpha = 0.3
        track.bearing_deg = track.bearing_deg + alpha * self._bearing_diff_signed(
            det.bearing_deg, track.bearing_deg)
        track.bearing_deg = track.bearing_deg % 360
        track.range_ratio = track.range_ratio + alpha * (det.range_ratio - track.range_ratio)
        track.intensity = track.intensity + alpha * (det.intensity - track.intensity)

        track.hits += 1
        track.misses = 0
        track.last_update = current_bearing
        track.detections.append(det)

        # Keep only recent detections
        if len(track.detections) > 20:
            track.detections = track.detections[-20:]

    def _bearing_diff_signed(self, b1: float, b2: float) -> float:
        """Signed bearing difference (b1 - b2)."""
        diff = b1 - b2
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        return diff

    def _create_track(self, det: Detection, current_bearing: float) -> None:
        """Create a new track from a detection."""
        track = TrackedTarget(
            id=self._next_id,
            bearing_deg=det.bearing_deg,
            range_ratio=det.range_ratio,
            intensity=det.intensity,
            label=f"T{self._next_id}",
            hits=1,
            misses=0,
            last_update=current_bearing,
            detections=[det]
        )
        self._tracks[self._next_id] = track
        self._next_id += 1

    def get_stable_tracks(self) -> List[TrackedTarget]:
        """Get tracks that are stable enough to display."""
        return [t for t in self._tracks.values()
                if t.hits >= self.min_hits_for_label]

    def get_all_tracks(self) -> List[TrackedTarget]:
        """Get all current tracks."""
        return list(self._tracks.values())

    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len(self._tracks)

    def get_stable_count(self) -> int:
        """Get number of stable tracks."""
        return len(self.get_stable_tracks())
