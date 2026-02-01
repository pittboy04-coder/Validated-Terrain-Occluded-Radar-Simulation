"""Sea and rain clutter simulation with K-distribution sea clutter."""
import random
import math
from typing import List


class SeaClutter:
    """Simulates sea surface clutter returns using K-distribution."""

    def __init__(self):
        self.sea_state = 3  # 0-9 Douglas sea scale
        self.wind_direction = 0.0  # degrees
        self.base_intensity = 0.1

    def set_sea_state(self, state: int) -> None:
        """Set sea state (0-9 Douglas scale)."""
        self.sea_state = max(0, min(9, state))

    def set_wind(self, direction: float, speed_knots: float = None) -> None:
        """Set wind direction (and optionally estimate sea state from wind)."""
        self.wind_direction = direction % 360
        if speed_knots is not None:
            self.sea_state = min(9, int(speed_knots / 5))

    def generate_clutter(self, num_range_bins: int, bearing: float,
                        max_range_m: float, suppression: float = 0.0) -> List[float]:
        """Generate K-distribution sea clutter for a sweep.

        K-distribution = gamma(nu) * exponential, where nu (shape) decreases
        with sea state to produce spikier clutter.
        """
        clutter = []
        bin_size = max_range_m / num_range_bins

        # K-distribution shape parameter: lower = spikier
        nu = max(0.5, 10.0 - self.sea_state * 1.2)

        # Wind direction modulation
        wind_factor = 1.0 + 0.3 * math.cos(math.radians(bearing - self.wind_direction))

        # Base clutter power from sea state
        base = self.base_intensity * (self.sea_state / 5.0)

        for i in range(num_range_bins):
            range_m = (i + 0.5) * bin_size

            # Range-dependent falloff: R^-2.5 (corrected from R^-1.5)
            if range_m < 100:
                range_factor = 1.0
            else:
                range_factor = (100.0 / range_m) ** 2.5

            # K-distribution sample: gamma(nu, 1/nu) * exponential(1)
            # Mean of the product = 1.0 (when base=1)
            gamma_sample = random.gammavariate(nu, 1.0 / nu)
            exp_sample = random.expovariate(1.0)
            k_sample = gamma_sample * exp_sample

            clutter_val = base * k_sample * range_factor * wind_factor
            clutter_val = min(1.0, clutter_val)

            # Apply suppression
            clutter_val *= (1.0 - suppression)

            clutter.append(max(0.0, clutter_val))

        return clutter


class RainClutter:
    """Simulates rain/precipitation clutter."""

    def __init__(self):
        self.rain_rate_mmh = 0.0  # mm/hour
        self.rain_cells = []  # List of (range, bearing, radius, intensity)

    def set_rain_rate(self, rate_mmh: float) -> None:
        """Set uniform rain rate in mm/hour."""
        self.rain_rate_mmh = max(0.0, rate_mmh)

    def add_rain_cell(self, range_m: float, bearing: float,
                     radius_m: float, intensity: float) -> None:
        """Add a localized rain cell."""
        self.rain_cells.append((range_m, bearing, radius_m, intensity))

    def clear_rain_cells(self) -> None:
        """Remove all rain cells."""
        self.rain_cells.clear()

    def generate_clutter(self, num_range_bins: int, bearing: float,
                        max_range_m: float, suppression: float = 0.0) -> List[float]:
        """Generate rain clutter for a sweep."""
        clutter = [0.0] * num_range_bins
        bin_size = max_range_m / num_range_bins

        # Uniform rain
        if self.rain_rate_mmh > 0:
            uniform_level = min(0.5, self.rain_rate_mmh / 100.0)
            for i in range(num_range_bins):
                clutter[i] = uniform_level * random.uniform(0.5, 1.0)

        # Rain cells
        for cell_range, cell_bearing, cell_radius, cell_intensity in self.rain_cells:
            bearing_diff = abs((bearing - cell_bearing + 180) % 360 - 180)
            if bearing_diff > 30:
                continue

            for i in range(num_range_bins):
                range_m = (i + 0.5) * bin_size

                if abs(range_m - cell_range) < cell_radius:
                    angular_dist = range_m * math.radians(bearing_diff)
                    dist_to_center = math.sqrt((range_m - cell_range)**2 + angular_dist**2)

                    if dist_to_center < cell_radius:
                        edge_factor = 1.0 - (dist_to_center / cell_radius)
                        clutter[i] = max(clutter[i],
                                        cell_intensity * edge_factor * random.uniform(0.7, 1.0))

        # Apply suppression
        clutter = [c * (1.0 - suppression) for c in clutter]

        return clutter
