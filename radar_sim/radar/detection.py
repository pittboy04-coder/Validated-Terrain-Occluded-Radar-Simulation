"""Radar detection engine for target detection calculations."""
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING
from .parameters import RadarParameters
from .antenna import Antenna
from ..objects.vessel import Vessel

if TYPE_CHECKING:
    from ..environment.occlusion import OcclusionEngine

@dataclass
class Detection:
    """Represents a radar detection/contact."""
    range_m: float
    bearing_deg: float
    intensity: float  # 0-1, signal strength
    vessel_id: Optional[str] = None  # If associated with a known vessel
    timestamp: float = 0.0

class DetectionEngine:
    """Handles radar detection calculations using proper dB-domain radar equation."""

    def __init__(self, params: RadarParameters, antenna: Antenna):
        self.params = params
        self.antenna = antenna
        self.occlusion_engine: Optional["OcclusionEngine"] = None
        # Pre-computed constant part of radar equation (dB)
        self._update_radar_constant()

    def _update_radar_constant(self) -> None:
        """Pre-compute the constant portion of the radar equation in dB."""
        p = self.params
        pt_dbw = 10.0 * math.log10(p.peak_power_kw * 1000.0)
        gt_db = p.antenna_gain_db
        gr_db = p.antenna_gain_db
        lam = p.wavelength_m
        lam_sq_db = 10.0 * math.log10(lam * lam)
        four_pi_cubed_db = 10.0 * math.log10((4.0 * math.pi) ** 3)
        # Radar equation constant: Pt + 2*G + lambda^2 - (4pi)^3 - Lsys
        self._k_db = (pt_dbw + gt_db + gr_db + lam_sq_db
                      - four_pi_cubed_db - p.system_losses_db)

    def calculate_received_power_db(self, target: Vessel, own_x: float, own_y: float,
                                     bearing_gain_db: float = 0.0) -> float:
        """Calculate received signal power in dB using radar equation.

        Pr(dB) = K + RCS_db - 40*log10(R) + beam_gain_db + scintillation + aspect

        Args:
            target: Target vessel
            own_x, own_y: Own ship position
            bearing_gain_db: Antenna beam pattern gain for this target (dB)

        Returns:
            Received power in dBW (arbitrary reference, used relatively)
        """
        dx = target.x - own_x
        dy = target.y - own_y
        range_m = math.sqrt(dx * dx + dy * dy)

        if range_m < 10:
            return -999.0

        # Range loss: R^-4
        range_loss_db = 40.0 * math.log10(range_m)

        # RCS in dB
        rcs_db = 10.0 * math.log10(max(0.01, target.rcs))

        # Swerling scintillation
        scint_db = _swerling_scintillation(
            getattr(target, 'swerling_type', 1))

        # Aspect-dependent RCS variation
        target_bearing = math.degrees(math.atan2(dx, dy)) % 360
        aspect_angle = (target_bearing - target.course) % 360
        aspect_db = _aspect_rcs_variation(
            getattr(target, 'aspect_rcs_variation_db', 6.0), aspect_angle)

        # Gain adjustment (0-1 knob mapped to ±10 dB)
        gain_offset_db = (self.params.gain - 0.5) * 20.0

        pr_db = (self._k_db + rcs_db - range_loss_db
                 + bearing_gain_db + scint_db + aspect_db + gain_offset_db)

        return pr_db

    def calculate_received_power(self, target: Vessel, own_x: float, own_y: float) -> float:
        """Calculate received signal power from a target (legacy 0-1 interface).

        Returns:
            Relative signal strength (linear power, not clamped to 1)
        """
        dx = target.x - own_x
        dy = target.y - own_y
        target_bearing = math.degrees(math.atan2(dx, dy)) % 360
        antenna_gain = self.antenna.get_gain_for_target(target_bearing)
        if antenna_gain < 0.01:
            return 0.0

        beam_gain_db = 10.0 * math.log10(max(1e-10, antenna_gain))
        pr_db = self.calculate_received_power_db(target, own_x, own_y, beam_gain_db)

        # Convert to linear, normalize so that a strong target at mid-range ~ 0.5
        # Reference: a 100 m² target at 5 km should yield ~0.5
        ref_db = self._k_db + 20.0 - 40.0 * math.log10(5000.0)
        normalized = 10.0 ** ((pr_db - ref_db) / 20.0) * 0.5
        return max(0.0, normalized)

    def detect_targets(self, targets: List[Vessel], own_x: float, own_y: float,
                      current_time: float) -> List[Detection]:
        """Process all targets and return detections."""
        detections = []
        noise_floor_db = -self.params.display_log_range_db

        for target in targets:
            if not target.is_active:
                continue

            dx = target.x - own_x
            dy = target.y - own_y
            range_m = math.sqrt(dx * dx + dy * dy)
            bearing_deg = math.degrees(math.atan2(dx, dy)) % 360

            if range_m > self.params.max_range_m:
                continue

            if self.occlusion_engine is not None:
                if self.occlusion_engine.is_target_occluded(
                    own_x, own_y, target.x, target.y,
                    target_height_m=target.height
                ):
                    continue

            signal = self.calculate_received_power(target, own_x, own_y)

            if signal < 0.005:
                continue

            range_noise = random.gauss(0, self.params.range_resolution_m * 0.5)
            bearing_noise = random.gauss(0, self.params.horizontal_beamwidth_deg * 0.3)

            detection = Detection(
                range_m=range_m + range_noise,
                bearing_deg=(bearing_deg + bearing_noise) % 360,
                intensity=min(1.0, signal),
                vessel_id=target.id,
                timestamp=current_time
            )
            detections.append(detection)

        return detections

    def generate_sweep_data(self, targets: List[Vessel], own_x: float, own_y: float,
                           current_time: float, num_range_bins: int = 512) -> List[float]:
        """Generate radar return data for the current beam position.

        Returns linear power values (pre-log-compression).
        """
        sweep_data = [0.0] * num_range_bins
        max_range = self.params.max_range_m
        bin_size = max_range / num_range_bins
        pulse_length_m = self.params.current_pulse_length_us * 1e-6 * 299792458 / 2

        current_bearing = self.antenna.get_bearing()

        for target in targets:
            if not target.is_active:
                continue

            dx = target.x - own_x
            dy = target.y - own_y
            range_m = math.sqrt(dx * dx + dy * dy)
            bearing_deg = math.degrees(math.atan2(dx, dy)) % 360

            bearing_diff = (bearing_deg - current_bearing + 180) % 360 - 180
            beam_gain = self.antenna.get_beam_pattern(bearing_diff)

            if beam_gain < 0.01 or range_m > max_range:
                continue

            if self.occlusion_engine is not None:
                if self.occlusion_engine.is_target_occluded(
                    own_x, own_y, target.x, target.y,
                    target_height_m=target.height
                ):
                    continue

            beam_gain_db = 10.0 * math.log10(max(1e-10, beam_gain))
            pr_db = self.calculate_received_power_db(target, own_x, own_y, beam_gain_db)

            # Convert to linear power (arbitrary units, will be log-compressed later)
            ref_db = self._k_db + 20.0 - 40.0 * math.log10(5000.0)
            signal = max(0.0, 10.0 ** ((pr_db - ref_db) / 10.0))

            range_bin = int(range_m / bin_size)

            if 0 <= range_bin < num_range_bins:
                # Spread signal based on pulse length, not target physical size
                spread = max(1, int(pulse_length_m / bin_size))
                for i in range(-spread, spread + 1):
                    bin_idx = range_bin + i
                    if 0 <= bin_idx < num_range_bins:
                        spread_factor = 1.0 - abs(i) / (spread + 1)
                        sweep_data[bin_idx] = max(sweep_data[bin_idx],
                                                  signal * spread_factor)

        return sweep_data


def _swerling_scintillation(swerling_type: int, rng=None) -> float:
    """Generate Swerling RCS fluctuation in dB.

    Type 1: Exponential (Rayleigh amplitude) - buoys, small targets
    Type 3: Chi-squared 4 DOF - large ships with multiple scatterers

    Returns:
        Fluctuation in dB (can be positive or negative around mean).
    """
    if swerling_type == 1:
        # Exponential: mean=1, can spike high
        x = random.expovariate(1.0)
        return 10.0 * math.log10(max(1e-10, x))
    elif swerling_type == 3:
        # Chi-squared with 4 DOF (sum of 2 exponentials), less variable
        x = random.gammavariate(2.0, 1.0)
        return 10.0 * math.log10(max(1e-10, x / 2.0))
    return 0.0


def _aspect_rcs_variation(variation_db: float, aspect_angle: float) -> float:
    """Sinusoidal RCS variation with target aspect angle.

    Args:
        variation_db: Peak-to-peak variation (e.g., 6 dB).
        aspect_angle: Angle in degrees (0=bow, 90=beam, 180=stern).

    Returns:
        RCS adjustment in dB.
    """
    # Beam aspect (90°, 270°) gives maximum RCS; bow/stern gives minimum
    return (variation_db / 2.0) * math.cos(math.radians(2.0 * aspect_angle))
