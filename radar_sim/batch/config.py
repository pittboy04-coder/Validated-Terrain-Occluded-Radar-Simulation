"""Configuration for batch training data generation."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class BatchConfig:
    """Configuration for batch scenario generation."""

    # Vessel count range
    min_vessels: int = 2
    max_vessels: int = 8

    # Vessel type weights (relative probability)
    type_weights: Dict[str, float] = field(default_factory=lambda: {
        'cargo': 0.20,
        'tanker': 0.15,
        'fishing': 0.25,
        'sailing': 0.10,
        'tug': 0.05,
        'passenger': 0.05,
        'pilot': 0.05,
        'buoy': 0.15,
    })

    # Environmental ranges
    sea_state_range: Tuple[int, int] = (0, 6)
    rain_rate_range: Tuple[float, float] = (0.0, 20.0)
    wind_speed_range: Tuple[float, float] = (0.0, 30.0)

    # Terrain/coastline probability (0-1)
    terrain_probability: float = 0.3
    coastline_probability: float = 0.2

    # Range scales to use (nm)
    range_scales: List[float] = field(default_factory=lambda: [3.0, 6.0, 12.0])

    # Vessel placement
    min_range_m: float = 500.0
    max_range_fraction: float = 0.9  # Fraction of max display range

    # Output
    image_size: int = 512
    output_bits: int = 8
