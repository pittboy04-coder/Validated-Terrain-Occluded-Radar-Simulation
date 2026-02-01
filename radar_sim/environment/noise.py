"""Noise generation for radar simulation with Rayleigh-envelope noise."""
import random
import math
from typing import List


class NoiseGenerator:
    """Generates Rayleigh-envelope radar noise (magnitude of complex Gaussian)."""

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        self.thermal_noise_level = 0.02
        self.receiver_noise_level = 0.01

    def generate_thermal_noise(self, num_samples: int) -> List[float]:
        """Generate Rayleigh-envelope thermal noise.

        The magnitude of complex Gaussian noise follows a Rayleigh distribution,
        which is always non-negative — matching real radar receiver output.
        """
        sigma = self.thermal_noise_level
        result = []
        for _ in range(num_samples):
            re = random.gauss(0, sigma)
            im = random.gauss(0, sigma)
            result.append(math.sqrt(re * re + im * im))
        return result

    def add_noise_to_sweep(self, sweep_data: List[float], noise_level: float = None) -> List[float]:
        """Add Rayleigh-envelope noise to sweep data.

        Args:
            sweep_data: Original sweep data (linear power).
            noise_level: Override noise level (default uses thermal_noise_level).

        Returns:
            Sweep data with added noise (power addition).
        """
        level = noise_level if noise_level is not None else self.thermal_noise_level
        result = []
        for val in sweep_data:
            re = random.gauss(0, level)
            im = random.gauss(0, level)
            noise_power = re * re + im * im
            # Add noise power to signal power
            result.append(max(0.0, val + noise_power))
        return result

    def set_noise_level(self, level: float) -> None:
        """Set the thermal noise level."""
        self.thermal_noise_level = max(0.0, min(0.5, level))
