"""Batch generator for synthetic radar training data."""
import math
import os
import random
from typing import List, Optional

from .config import BatchConfig
from ..core.simulation import Simulation
from ..objects.vessel import Vessel, VesselType
from ..environment.weather import WeatherConditions
from ..environment.terrain import create_island_terrain
from ..export.annotation import AnnotationExporter


# Map string type to VesselType enum and default dimensions
_VESSEL_SPECS = {
    'cargo': (VesselType.CARGO, 150, 20, 25),
    'tanker': (VesselType.TANKER, 200, 30, 20),
    'fishing': (VesselType.FISHING, 25, 6, 8),
    'sailing': (VesselType.SAILING, 15, 4, 15),
    'tug': (VesselType.TUG, 30, 10, 12),
    'passenger': (VesselType.PASSENGER, 100, 20, 25),
    'pilot': (VesselType.PILOT, 15, 5, 6),
    'buoy': (VesselType.BUOY, 3, 3, 4),
}


class BatchGenerator:
    """Generate batches of random radar scenarios with annotations."""

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()

    def generate(self, n: int, output_dir: str) -> List[str]:
        """Generate N random scenarios, each producing CSV + annotations.

        Args:
            n: Number of scenarios to generate.
            output_dir: Directory for output files.

        Returns:
            List of output file paths (CSV files).
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []

        for scenario_idx in range(n):
            prefix = f"scenario_{scenario_idx + 1:04d}"
            csv_path = os.path.join(output_dir, f"{prefix}.csv")
            json_path = os.path.join(output_dir, f"{prefix}_annotations.json")
            yolo_path = os.path.join(output_dir, f"{prefix}_labels.txt")

            self._generate_one(csv_path, json_path, yolo_path)
            output_files.append(csv_path)
            print(f"  Generated {prefix} ({scenario_idx + 1}/{n})")

        return output_files

    def _generate_one(self, csv_path: str, json_path: str, yolo_path: str) -> None:
        """Generate a single scenario."""
        cfg = self.config
        sim = Simulation()

        # Random range scale
        range_nm = random.choice(cfg.range_scales)
        sim.radar.set_range_scale(range_nm)
        max_range_m = sim.radar.params.max_range_m

        # Own ship at origin
        own_ship = Vessel(
            id="own_ship", name="Own Ship",
            vessel_type=VesselType.OWN_SHIP,
            x=0, y=0, course=random.uniform(0, 360), speed=random.uniform(0, 15),
            length=100, beam=15, height=20
        )
        sim.world.add_vessel(own_ship)

        # Random weather
        sea_state = random.randint(*cfg.sea_state_range)
        rain_rate = random.uniform(*cfg.rain_rate_range) if random.random() < 0.3 else 0.0
        wind_dir = random.uniform(0, 360)
        wind_speed = random.uniform(*cfg.wind_speed_range)

        sim.weather.set_conditions(WeatherConditions(
            sea_state=sea_state,
            wind_speed_knots=wind_speed,
            wind_direction=wind_dir,
            rain_rate_mmh=rain_rate,
            visibility_nm=10.0,
        ))

        # Random terrain
        if random.random() < cfg.terrain_probability:
            island_range = random.uniform(max_range_m * 0.2, max_range_m * 0.7)
            island_bearing = random.uniform(0, 360)
            ix = island_range * math.sin(math.radians(island_bearing))
            iy = island_range * math.cos(math.radians(island_bearing))
            hm = create_island_terrain(
                center_x=ix, center_y=iy,
                radius=random.uniform(300, 1500),
                peak_height=random.uniform(50, 200),
            )
            sim.add_terrain(hm)

        # Random coastline
        if random.random() < cfg.coastline_probability:
            sim.setup_harbor_coastline()

        # Random vessels
        num_vessels = random.randint(cfg.min_vessels, cfg.max_vessels)
        vessel_types = list(cfg.type_weights.keys())
        weights = [cfg.type_weights[t] for t in vessel_types]

        for vi in range(num_vessels):
            vtype_str = random.choices(vessel_types, weights=weights, k=1)[0]
            specs = _VESSEL_SPECS.get(vtype_str, _VESSEL_SPECS['cargo'])
            vtype_enum, length, beam, height = specs

            # Random polar placement
            r = random.uniform(cfg.min_range_m, max_range_m * cfg.max_range_fraction)
            b = random.uniform(0, 360)
            vx = r * math.sin(math.radians(b))
            vy = r * math.cos(math.radians(b))

            course = random.uniform(0, 360)
            speed = 0.0 if vtype_str == 'buoy' else random.uniform(0, 15)

            vessel = Vessel(
                id=f"target_{vi + 1}", name=f"{vtype_str}_{vi + 1}",
                vessel_type=vtype_enum, x=vx, y=vy,
                course=course, speed=speed,
                length=length, beam=beam, height=height,
            )
            sim.world.add_vessel(vessel)

        # Run one full rotation of sweep data and record to CSV
        sim.exporter.output_dir = os.path.dirname(csv_path) or '.'
        sim.exporter.start_recording()

        # Generate 360 sweeps (one per degree)
        for bearing in range(360):
            sweep_data = sim.get_radar_sweep_data(float(bearing))
            sim.exporter.add_sweep(
                timestamp=0.0,
                bearing_deg=float(bearing),
                range_scale_nm=range_nm,
                gain=sim.radar.params.gain,
                sea_clutter=sim.radar.params.sea_clutter,
                rain_clutter=sim.radar.params.rain_clutter,
                echo_values=sweep_data,
            )

        # Save CSV
        sim.exporter.save_to_csv(os.path.basename(csv_path))
        sim.exporter.is_recording = False
        sim.exporter.records.clear()

        # Collect and export annotations
        annotation_dicts = sim.collect_annotations()
        exporter = AnnotationExporter(
            image_size=cfg.image_size,
            range_scale_m=max_range_m,
        )
        annotations = exporter.annotations_from_dicts(annotation_dicts)

        params_dict = {
            'range_nm': range_nm,
            'sea_state': sea_state,
            'rain_rate_mmh': rain_rate,
            'wind_direction': wind_dir,
            'wind_speed_knots': wind_speed,
            'num_vessels': num_vessels,
            'terrain': len(sim.terrain_maps) > 0,
            'coastline': sim.coastline_enabled,
        }

        exporter.export_json(annotations, params_dict, json_path)
        exporter.export_yolo(annotations, cfg.image_size, yolo_path)
