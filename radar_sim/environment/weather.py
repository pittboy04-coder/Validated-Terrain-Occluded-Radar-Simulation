"""Weather effects for radar simulation."""
import math
from dataclasses import dataclass
from typing import List
from .noise import NoiseGenerator
from .clutter import SeaClutter, RainClutter

@dataclass
class WeatherConditions:
    """Current weather conditions."""
    sea_state: int = 3  # 0-9 Douglas scale
    wind_speed_knots: float = 10.0
    wind_direction: float = 0.0  # degrees
    rain_rate_mmh: float = 0.0
    visibility_nm: float = 10.0

class WeatherEffects:
    """Combines all weather-related effects on radar."""

    def __init__(self):
        self.conditions = WeatherConditions()
        self.noise_gen = NoiseGenerator()
        self.sea_clutter = SeaClutter()
        self.rain_clutter = RainClutter()

    def set_conditions(self, conditions: WeatherConditions) -> None:
        """Set weather conditions."""
        self.conditions = conditions
        self.sea_clutter.set_sea_state(conditions.sea_state)
        self.sea_clutter.set_wind(conditions.wind_direction)
        self.rain_clutter.set_rain_rate(conditions.rain_rate_mmh)

    def set_sea_state(self, state: int) -> None:
        """Set sea state."""
        self.conditions.sea_state = state
        self.sea_clutter.set_sea_state(state)

    def set_wind(self, direction: float, speed_knots: float) -> None:
        """Set wind conditions."""
        self.conditions.wind_direction = direction
        self.conditions.wind_speed_knots = speed_knots
        self.sea_clutter.set_wind(direction, speed_knots)

    def set_rain(self, rate_mmh: float) -> None:
        """Set rain rate."""
        self.conditions.rain_rate_mmh = rate_mmh
        self.rain_clutter.set_rain_rate(rate_mmh)

    def apply_to_sweep(self, sweep_data: List[float], bearing: float,
                      max_range_m: float, sea_suppression: float = 0.0,
                      rain_suppression: float = 0.0) -> List[float]:
        """Apply all weather effects to sweep data.

        Uses power addition (not max) for physically correct combination,
        and applies two-way rain attenuation per bin.
        """
        num_bins = len(sweep_data)
        bin_size = max_range_m / num_bins

        # Generate clutter
        sea = self.sea_clutter.generate_clutter(
            num_bins, bearing, max_range_m, sea_suppression
        )
        rain = self.rain_clutter.generate_clutter(
            num_bins, bearing, max_range_m, rain_suppression
        )

        # Rain attenuation: Marshall-Palmer at X-band
        # 0.01 * R^1.21 dB/km (R = rain rate in mm/h)
        rain_rate = self.conditions.rain_rate_mmh
        if rain_rate > 0:
            rain_atten_db_per_km = 0.01 * (rain_rate ** 1.21)
        else:
            rain_atten_db_per_km = 0.0

        # Combine: power addition (val + sea + rain), then attenuation + noise
        result = []
        for i in range(num_bins):
            # Power addition instead of max
            combined = sweep_data[i] + sea[i] + rain[i]

            # Two-way rain attenuation based on range
            if rain_atten_db_per_km > 0:
                range_km = ((i + 0.5) * bin_size) / 1000.0
                atten_db = rain_atten_db_per_km * range_km * 2.0  # two-way
                atten_linear = 10.0 ** (-atten_db / 10.0)
                combined *= atten_linear

            result.append(combined)

        # Add noise
        result = self.noise_gen.add_noise_to_sweep(result)

        return result

    def get_attenuation_factor(self, range_m: float) -> float:
        """Calculate atmospheric attenuation based on weather."""
        rain_rate = self.conditions.rain_rate_mmh
        if rain_rate <= 0:
            return 1.0
        rain_atten_db_per_km = 0.01 * (rain_rate ** 1.21)
        range_km = range_m / 1000.0
        total_atten_db = rain_atten_db_per_km * range_km * 2
        return 10 ** (-total_atten_db / 10)
