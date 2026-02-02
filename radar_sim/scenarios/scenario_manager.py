"""Scenario management for the radar simulator."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from ..objects.vessel import Vessel, VesselType
from ..environment.weather import WeatherConditions
from ..core.world import World
from ..radar.system import RadarSystem
from ..environment.weather import WeatherEffects
from ..environment.terrain import HeightMap


@dataclass
class VesselConfig:
    id: str
    name: str
    vessel_type: VesselType
    x: float
    y: float
    course: float
    speed: float
    length: float = 50.0
    beam: float = 10.0
    height: float = 15.0


@dataclass
class TerrainConfig:
    """Configuration for terrain in a scenario."""
    type: str = "island"  # "island" or "ridge"
    center_x: float = 0.0
    center_y: float = 3000.0
    radius: float = 800.0
    peak_height: float = 120.0
    grid_size: int = 64
    cell_size: float = 50.0
    # Ridge-specific
    end_x: float = 0.0
    end_y: float = 0.0
    width: float = 600.0


@dataclass
class Scenario:
    name: str
    description: str
    vessels: List[VesselConfig]
    weather: WeatherConditions = field(default_factory=WeatherConditions)
    radar_range_nm: float = 6.0
    own_ship_index: int = 0
    terrain: List[TerrainConfig] = field(default_factory=list)
    enable_coastline: bool = False


class ScenarioManager:
    def __init__(self):
        self.scenarios: Dict[str, Scenario] = {}
        self.current_scenario: Optional[str] = None
        self.on_scenario_loaded: Optional[Callable[[Scenario], None]] = None

    def register_scenario(self, scenario: Scenario) -> None:
        self.scenarios[scenario.name] = scenario

    def register_scenarios(self, scenarios: List[Scenario]) -> None:
        for scenario in scenarios:
            self.register_scenario(scenario)

    def get_scenario(self, name: str) -> Optional[Scenario]:
        return self.scenarios.get(name)

    def get_scenario_names(self) -> List[str]:
        return list(self.scenarios.keys())

    def load_scenario(self, name: str, world: World, radar: RadarSystem,
                      weather: WeatherEffects, simulation=None) -> bool:
        """Load a scenario into the simulation.

        Args:
            name: Scenario name
            world: World instance to populate
            radar: Radar system to configure
            weather: Weather effects to configure
            simulation: Optional Simulation instance for terrain/coastline setup
        """
        scenario = self.scenarios.get(name)
        if not scenario:
            return False

        world.clear()
        radar.clear_sweep_buffer()

        # Clear terrain if simulation is available
        if simulation is not None:
            simulation.clear_terrain()
            simulation.clear_coastlines()

        for i, config in enumerate(scenario.vessels):
            vessel = Vessel(
                id=config.id, name=config.name,
                vessel_type=config.vessel_type,
                x=config.x, y=config.y,
                course=config.course, speed=config.speed,
                length=config.length, beam=config.beam,
                height=config.height)
            if i == scenario.own_ship_index:
                vessel.vessel_type = VesselType.OWN_SHIP
            world.add_vessel(vessel)

        weather.set_conditions(scenario.weather)
        radar.set_range_scale(scenario.radar_range_nm)

        # Load terrain
        if simulation is not None and scenario.terrain:
            from ..environment.terrain import create_island_terrain, create_ridge_terrain
            for tc in scenario.terrain:
                if tc.type == "island":
                    hm = create_island_terrain(
                        center_x=tc.center_x, center_y=tc.center_y,
                        radius=tc.radius, peak_height=tc.peak_height,
                        grid_size=tc.grid_size, cell_size=tc.cell_size)
                elif tc.type == "ridge":
                    hm = create_ridge_terrain(
                        start_x=tc.center_x, start_y=tc.center_y,
                        end_x=tc.end_x, end_y=tc.end_y,
                        width=tc.width, peak_height=tc.peak_height,
                        grid_size=tc.grid_size, cell_size=tc.cell_size)
                else:
                    continue
                simulation.add_terrain(hm)

            # Set up coastline if requested
            if scenario.enable_coastline:
                simulation.setup_harbor_coastline()

        self.current_scenario = name
        if self.on_scenario_loaded:
            self.on_scenario_loaded(scenario)
        return True

    def load_location_file(self, filepath: str, world: World, radar: RadarSystem,
                           weather: WeatherEffects, simulation=None) -> bool:
        """Load a .radarloc location file into the simulation.

        Args:
            filepath: Path to the .radarloc file.
            world: World instance to populate.
            radar: Radar system to configure.
            weather: Weather effects instance.
            simulation: Optional Simulation instance for terrain/coastline setup.

        Returns:
            True on success.
        """
        from .location_loader import load_radarloc_file

        try:
            loc_data = load_radarloc_file(filepath)
        except Exception as e:
            print(f"Failed to load location file: {e}")
            return False

        world.clear()
        radar.clear_sweep_buffer()

        if simulation is not None:
            simulation.clear_terrain()
            simulation.clear_coastlines()

        # Place own ship at origin
        own_ship = Vessel(
            id="own_ship", name="Own Ship",
            vessel_type=VesselType.OWN_SHIP,
            x=0.0, y=0.0, course=0.0, speed=0.0)
        world.add_vessel(own_ship)

        # Add preset vessels from file
        for vessel in loc_data.vessels:
            world.add_vessel(vessel)

        # Set radar range
        radar.set_range_scale(loc_data.range_nm)

        if simulation is not None:
            # Load coastlines
            for coastline in loc_data.coastlines:
                simulation.add_coastline(coastline)

            # Load terrain
            if loc_data.terrain is not None:
                simulation.add_terrain(loc_data.terrain)

        self.current_scenario = f"LOC:{loc_data.location_name}"
        if self.on_scenario_loaded:
            # Create a minimal Scenario object for the callback
            scenario = Scenario(
                name=self.current_scenario,
                description=f"Loaded from {filepath}",
                vessels=[],
                radar_range_nm=loc_data.range_nm,
            )
            self.on_scenario_loaded(scenario)
        return True

    def get_current_scenario(self) -> Optional[Scenario]:
        if self.current_scenario:
            return self.scenarios.get(self.current_scenario)
        return None
