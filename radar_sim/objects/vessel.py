"""Vessel class representing ships and other radar targets."""
import math
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class VesselType(Enum):
    OWN_SHIP = "own_ship"
    CARGO = "cargo"
    TANKER = "tanker"
    FISHING = "fishing"
    SAILING = "sailing"
    PASSENGER = "passenger"
    TUG = "tug"
    PILOT = "pilot"
    BUOY = "buoy"
    UNKNOWN = "unknown"

@dataclass
class Vessel:
    """Represents a vessel or radar target in the simulation."""

    # Identity
    id: str
    name: str = ""
    vessel_type: VesselType = VesselType.UNKNOWN

    # Position (meters, relative to origin or lat/lon if needed)
    x: float = 0.0  # East-West position (positive = East)
    y: float = 0.0  # North-South position (positive = North)

    # Motion
    course: float = 0.0  # True course in degrees (0 = North)
    speed: float = 0.0   # Speed in knots

    # Physical characteristics (for radar cross-section)
    length: float = 50.0   # Length in meters
    beam: float = 10.0     # Width in meters
    height: float = 15.0   # Height above waterline in meters
    rcs: Optional[float] = None  # Radar cross-section in m^2 (None = auto-calculate)

    # Scintillation parameters
    swerling_type: int = 1  # 1 = buoys/small, 3 = large ships
    aspect_rcs_variation_db: float = 6.0  # Peak-to-peak aspect RCS variation

    # State
    is_active: bool = True

    def __post_init__(self):
        if self.rcs is None:
            self.rcs = self._estimate_rcs()

    def _estimate_rcs(self) -> float:
        """Estimate radar cross-section from vessel dimensions."""
        if self.vessel_type == VesselType.BUOY:
            self.swerling_type = 1
            self.aspect_rcs_variation_db = 3.0
            return max(5.0, self.length * self.height * 1.5)
        elif self.vessel_type == VesselType.SAILING:
            self.swerling_type = 1
            self.aspect_rcs_variation_db = 8.0
            return max(20.0, self.length * self.height * 0.5)
        elif self.vessel_type in (VesselType.FISHING, VesselType.PILOT):
            self.swerling_type = 1
            self.aspect_rcs_variation_db = 6.0
            return max(30.0, self.length * self.height * 0.4)
        elif self.vessel_type in (VesselType.CARGO, VesselType.TANKER, VesselType.PASSENGER):
            self.swerling_type = 3
            self.aspect_rcs_variation_db = 6.0
            return max(50.0, self.length * self.height * 0.3)
        else:
            self.swerling_type = 1
            self.aspect_rcs_variation_db = 6.0
            return max(50.0, self.length * self.height * 0.3)

    def update(self, dt: float) -> None:
        """Update vessel position based on course and speed."""
        if not self.is_active or self.speed == 0:
            return

        speed_ms = self.speed * 0.514444
        course_rad = math.radians(self.course)
        vx = speed_ms * math.sin(course_rad)
        vy = speed_ms * math.cos(course_rad)

        self.x += vx * dt
        self.y += vy * dt

    def set_position(self, x: float, y: float) -> None:
        """Set vessel position."""
        self.x = x
        self.y = y

    def set_motion(self, course: float, speed: float) -> None:
        """Set vessel course and speed."""
        self.course = course % 360
        self.speed = max(0, speed)

    def get_velocity(self) -> tuple[float, float]:
        """Get velocity components in m/s (vx, vy)."""
        speed_ms = self.speed * 0.514444
        course_rad = math.radians(self.course)
        return (speed_ms * math.sin(course_rad), speed_ms * math.cos(course_rad))

    def distance_to(self, other: 'Vessel') -> float:
        """Calculate distance to another vessel in meters."""
        dx = other.x - self.x
        dy = other.y - self.y
        return math.sqrt(dx * dx + dy * dy)

    def bearing_to(self, other: 'Vessel') -> float:
        """Calculate true bearing to another vessel in degrees."""
        dx = other.x - self.x
        dy = other.y - self.y
        bearing = math.degrees(math.atan2(dx, dy))
        return bearing % 360
