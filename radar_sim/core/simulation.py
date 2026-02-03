"""Main simulation engine with terrain occlusion support."""
import time
from typing import Optional, Callable, List
from .world import World
from ..radar.system import RadarSystem
from ..radar.signal_processing import apply_stc, log_compress, quantize
from ..environment.weather import WeatherEffects, WeatherConditions
from ..environment.coastline import Coastline, create_harbor_coastline, create_island_coastline
from ..environment.terrain import HeightMap
from ..environment.occlusion import OcclusionEngine
from ..data_export import RadarDataExporter

class Simulation:
    """Main simulation controller."""

    def __init__(self):
        self.world = World()
        self.radar = RadarSystem()
        self.weather = WeatherEffects()
        self.exporter = RadarDataExporter()

        # Coastline/landmass
        self.coastlines: List[Coastline] = []
        self.coastline_enabled = False

        # Terrain / occlusion
        self.terrain_maps: List[HeightMap] = []
        self.occlusion_engine: Optional[OcclusionEngine] = None

        # Timing
        self.time_scale = 1.0  # 1.0 = real-time
        self.target_fps = 60
        self.dt = 1.0 / self.target_fps

        # State
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0

        # Annotation collection (populated each rotation for export)
        self._current_annotations: List[dict] = []

        # Callbacks
        self.on_update: Optional[Callable[[float], None]] = None

    # ------------------------------------------------------------------
    # Coastlines
    # ------------------------------------------------------------------
    def add_coastline(self, coastline: Coastline) -> None:
        """Add a coastline to the simulation."""
        self.coastlines.append(coastline)
        self.coastline_enabled = True

    def clear_coastlines(self) -> None:
        """Remove all coastlines."""
        self.coastlines.clear()
        self.coastline_enabled = False

    def setup_harbor_coastline(self) -> None:
        """Set up a default harbor coastline ahead of own ship."""
        self.clear_coastlines()
        harbor = create_harbor_coastline(
            center_x=0, center_y=8000, width=12000, depth=3000
        )
        self.add_coastline(harbor)
        island = create_island_coastline(
            center_x=4000, center_y=5000, radius=500, num_points=16
        )
        self.add_coastline(island)

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------
    def add_terrain(self, terrain: HeightMap) -> None:
        """Add a terrain height map and rebuild the occlusion engine."""
        self.terrain_maps.append(terrain)
        self._rebuild_occlusion_engine()

    def clear_terrain(self) -> None:
        """Remove all terrain maps and disable occlusion."""
        self.terrain_maps.clear()
        self.occlusion_engine = None
        self.radar.detection_engine.occlusion_engine = None

    def move_terrain(self, index: int, new_cx: float, new_cy: float) -> None:
        """Move a terrain map so its center is at (new_cx, new_cy)."""
        if 0 <= index < len(self.terrain_maps):
            self.terrain_maps[index].move_center(new_cx, new_cy)
            self._rebuild_occlusion_engine()

    def remove_terrain(self, index: int) -> None:
        """Remove a single terrain map by index."""
        if 0 <= index < len(self.terrain_maps):
            self.terrain_maps.pop(index)
            self._rebuild_occlusion_engine()

    def add_vessel_at(self, vessel_type_str: str, x: float, y: float) -> None:
        """Add a new vessel of the given type at (x, y)."""
        from ..objects.vessel import Vessel, VesselType

        type_map = {
            "cargo": (VesselType.CARGO, "Cargo Ship", 150, 20, 25),
            "tanker": (VesselType.TANKER, "Tanker", 200, 30, 20),
            "fishing": (VesselType.FISHING, "Fishing Boat", 25, 6, 8),
            "sailing": (VesselType.SAILING, "Sailing Yacht", 15, 4, 15),
            "tug": (VesselType.TUG, "Tug", 30, 10, 12),
            "passenger": (VesselType.PASSENGER, "Ferry", 100, 20, 25),
            "pilot": (VesselType.PILOT, "Pilot Boat", 15, 5, 6),
            "buoy": (VesselType.BUOY, "Buoy", 3, 3, 4),
        }

        vtype, name, length, beam, height = type_map.get(
            vessel_type_str, (VesselType.CARGO, "Ship", 100, 15, 20))

        # Generate unique id
        existing_ids = set(self.world.vessels.keys())
        idx = 1
        while f"placed_{vessel_type_str}_{idx}" in existing_ids:
            idx += 1
        vid = f"placed_{vessel_type_str}_{idx}"

        vessel = Vessel(
            id=vid, name=f"{name} {idx}",
            vessel_type=vtype, x=x, y=y,
            course=0, speed=0,
            length=length, beam=beam, height=height)
        self.world.add_vessel(vessel)

    def remove_vessel(self, vessel_id: str) -> None:
        """Remove a vessel by ID (not own ship)."""
        vessel = self.world.get_vessel(vessel_id)
        if vessel and vessel is not self.world.own_ship:
            self.world.remove_vessel(vessel_id)

    def _rebuild_occlusion_engine(self) -> None:
        """(Re)create the OcclusionEngine from current terrain maps."""
        if self.terrain_maps:
            self.occlusion_engine = OcclusionEngine(
                self.terrain_maps,
                antenna_height_m=self.radar.params.antenna_height_m,
            )
            self.radar.detection_engine.occlusion_engine = self.occlusion_engine
        else:
            self.occlusion_engine = None
            self.radar.detection_engine.occlusion_engine = None

    # ------------------------------------------------------------------
    # Default scenario
    # ------------------------------------------------------------------
    def setup_default_scenario(self) -> None:
        """Set up a default scenario with own ship and some targets."""
        from ..objects.vessel import Vessel, VesselType

        own_ship = Vessel(
            id="own_ship", name="Own Ship",
            vessel_type=VesselType.OWN_SHIP,
            x=0, y=0, course=0, speed=10,
            length=100, beam=15, height=20
        )
        self.world.add_vessel(own_ship)

        targets = [
            Vessel(id="target_1", name="Cargo Ship", vessel_type=VesselType.CARGO,
                  x=3000, y=5000, course=225, speed=12, length=150, beam=20, height=25),
            Vessel(id="target_2", name="Tanker", vessel_type=VesselType.TANKER,
                  x=-4000, y=3000, course=90, speed=8, length=200, beam=30, height=20),
            Vessel(id="target_3", name="Fishing Boat", vessel_type=VesselType.FISHING,
                  x=2000, y=-2000, course=315, speed=6, length=25, beam=6, height=8),
            Vessel(id="target_4", name="Sailing Yacht", vessel_type=VesselType.SAILING,
                  x=-1500, y=-4000, course=45, speed=5, length=15, beam=4, height=15),
        ]
        for target in targets:
            self.world.add_vessel(target)

        self.weather.set_conditions(WeatherConditions(
            sea_state=3, wind_speed_knots=15, wind_direction=45,
            rain_rate_mmh=0, visibility_nm=10
        ))

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    def collect_annotations(self) -> List[dict]:
        """Collect annotation data for all current targets."""
        own_ship = self.world.own_ship
        if own_ship is None:
            return []

        import math
        annotations = []
        for vessel in self.world.get_targets():
            dx = vessel.x - own_ship.x
            dy = vessel.y - own_ship.y
            range_m = math.sqrt(dx * dx + dy * dy)
            bearing_deg = math.degrees(math.atan2(dx, dy)) % 360

            occluded = False
            if self.occlusion_engine is not None:
                occluded = self.occlusion_engine.is_target_occluded(
                    own_ship.x, own_ship.y, vessel.x, vessel.y,
                    target_height_m=vessel.height
                )

            annotations.append({
                'vessel_id': vessel.id,
                'vessel_type': vessel.vessel_type.value,
                'range_m': range_m,
                'bearing_deg': bearing_deg,
                'rcs': vessel.rcs,
                'length': vessel.length,
                'beam': vessel.beam,
                'height': vessel.height,
                'occluded': occluded,
                'x': vessel.x,
                'y': vessel.y,
                'course': vessel.course,
                'speed': vessel.speed,
            })
        self._current_annotations = annotations
        return annotations

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------
    def update(self, dt: float = None) -> None:
        """Update simulation by one time step."""
        if self.is_paused:
            return

        dt = dt or self.dt
        scaled_dt = dt * self.time_scale

        self.world.update(scaled_dt)
        self.radar.update(self.world, scaled_dt)

        self.frame_count += 1
        if self.on_update:
            self.on_update(scaled_dt)

    def get_radar_sweep_data(self, bearing: float) -> list:
        """Get radar sweep data for a bearing, including terrain returns.

        Applies full signal chain: weather → STC → log compression → quantization.
        """
        own_ship = self.world.own_ship
        if own_ship is None:
            return [0.0] * self.radar.num_range_bins

        # Generate fresh target sweep data for this bearing
        targets = self.world.get_targets()
        own_x = own_ship.x
        own_y = own_ship.y

        sweep_data = self.radar.detection_engine.generate_sweep_data(
            targets, own_x, own_y, self.world.time,
            self.radar.num_range_bins
        )

        # Coastline returns
        if self.coastline_enabled:
            for coastline in self.coastlines:
                coast_returns = coastline.generate_returns(
                    own_x, own_y, bearing,
                    self.radar.params.horizontal_beamwidth_deg,
                    self.radar.params.max_range_m,
                    len(sweep_data)
                )
                for i, val in enumerate(coast_returns):
                    sweep_data[i] = max(sweep_data[i], val * self.radar.params.gain)

        # Terrain returns
        if self.occlusion_engine is not None:
            terrain_returns = self.occlusion_engine.generate_terrain_returns(
                own_x, own_y, bearing,
                self.radar.params.horizontal_beamwidth_deg,
                self.radar.params.max_range_m,
                len(sweep_data)
            )
            for i, val in enumerate(terrain_returns):
                sweep_data[i] = max(sweep_data[i], val * self.radar.params.gain)

        # Weather effects (adds clutter + noise in linear power domain)
        sweep_data = self.weather.apply_to_sweep(
            sweep_data, bearing,
            self.radar.params.max_range_m,
            self.radar.params.sea_clutter,
            self.radar.params.rain_clutter
        )

        # STC (near-range attenuation, applied in linear domain)
        stc_curve = self.radar.params.compute_stc_curve(len(sweep_data))
        sweep_data = apply_stc(sweep_data, stc_curve, self.radar.params.max_range_m)

        # Log compression (linear → normalized dB domain 0-1)
        sweep_data = log_compress(sweep_data, self.radar.params.display_log_range_db)

        # Quantization
        sweep_data = quantize(sweep_data, self.radar.params.output_bits)

        return sweep_data

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------
    def set_time_scale(self, scale: float) -> None:
        self.time_scale = max(0.1, min(10.0, scale))

    def pause(self) -> None:
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False

    def toggle_pause(self) -> bool:
        self.is_paused = not self.is_paused
        return self.is_paused

    def reset(self) -> None:
        self.world.clear()
        self.radar.clear_sweep_buffer()
        self.clear_coastlines()
        self.clear_terrain()
        self.frame_count = 0
        self.is_paused = False
        self._current_annotations = []

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------
    def start_recording(self) -> str:
        return self.exporter.start_recording()

    def stop_recording(self) -> str:
        return self.exporter.stop_recording()

    def is_recording(self) -> bool:
        return self.exporter.is_recording

    def get_record_count(self) -> int:
        return self.exporter.get_record_count()

    def export_current_sweep(self, bearing: float) -> str:
        sweep_data = self.get_radar_sweep_data(bearing)
        return self.exporter.export_single_sweep(
            timestamp=self.world.time,
            bearing_deg=bearing,
            range_scale_nm=self.radar.params.current_range_nm,
            gain=self.radar.params.gain,
            sea_clutter=self.radar.params.sea_clutter,
            rain_clutter=self.radar.params.rain_clutter,
            echo_values=sweep_data,
        )
