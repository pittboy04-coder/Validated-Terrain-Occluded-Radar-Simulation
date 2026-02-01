"""Preset scenarios for the terrain-occluded radar simulator."""
from .scenario_manager import Scenario, VesselConfig, TerrainConfig
from ..objects.vessel import VesselType
from ..environment.weather import WeatherConditions


# --- Island Occlusion Demo ---
ISLAND_OCCLUSION = Scenario(
    name="Island Occlusion",
    description="Island blocks line-of-sight to vessel behind it. Core terrain occlusion demo.",
    vessels=[
        VesselConfig(id="own_ship", name="Own Ship", vessel_type=VesselType.OWN_SHIP,
                     x=0, y=0, course=0, speed=0, length=100, beam=15, height=20),
        VesselConfig(id="hidden", name="Hidden Cargo", vessel_type=VesselType.CARGO,
                     x=0, y=5000, course=180, speed=0, length=150, beam=20, height=25),
        VesselConfig(id="visible", name="Visible Tanker", vessel_type=VesselType.TANKER,
                     x=5000, y=0, course=270, speed=0, length=200, beam=30, height=20),
    ],
    weather=WeatherConditions(sea_state=1, wind_speed_knots=5, wind_direction=0),
    radar_range_nm=6.0,
    terrain=[
        TerrainConfig(type="island", center_x=0, center_y=3000,
                      radius=800, peak_height=120, grid_size=64, cell_size=50),
    ],
)

# --- Ridge Blockage ---
RIDGE_BLOCKAGE = Scenario(
    name="Ridge Blockage",
    description="A linear ridge blocks radar coverage in one direction.",
    vessels=[
        VesselConfig(id="own_ship", name="Own Ship", vessel_type=VesselType.OWN_SHIP,
                     x=0, y=0, course=0, speed=10, length=100, beam=15, height=20),
        VesselConfig(id="cargo_1", name="MV Behind Ridge", vessel_type=VesselType.CARGO,
                     x=-2000, y=6000, course=90, speed=12, length=180, beam=25, height=30),
        VesselConfig(id="tanker_1", name="MT Clear View", vessel_type=VesselType.TANKER,
                     x=4000, y=4000, course=225, speed=10, length=250, beam=40, height=25),
        VesselConfig(id="fishing_1", name="FV Coastal", vessel_type=VesselType.FISHING,
                     x=-3000, y=2000, course=45, speed=5, length=20, beam=6, height=8),
    ],
    weather=WeatherConditions(sea_state=2, wind_speed_knots=12, wind_direction=90),
    radar_range_nm=6.0,
    terrain=[
        TerrainConfig(type="ridge", center_x=-3000, center_y=4000,
                      end_x=1000, end_y=4500, width=500, peak_height=80,
                      grid_size=64, cell_size=50),
    ],
)

# --- Calm Traffic (no terrain) ---
CALM_TRAFFIC = Scenario(
    name="Calm Traffic",
    description="Light traffic on a calm day. No terrain occlusion.",
    vessels=[
        VesselConfig(id="own_ship", name="Own Ship", vessel_type=VesselType.OWN_SHIP,
                     x=0, y=0, course=0, speed=12, length=100, beam=15, height=20),
        VesselConfig(id="cargo_1", name="MV Pacific Star", vessel_type=VesselType.CARGO,
                     x=3000, y=5000, course=225, speed=14, length=180, beam=25, height=30),
        VesselConfig(id="tanker_1", name="MT Ocean Pride", vessel_type=VesselType.TANKER,
                     x=-4000, y=6000, course=135, speed=10, length=250, beam=40, height=25),
        VesselConfig(id="fishing_1", name="FV Morning Catch", vessel_type=VesselType.FISHING,
                     x=2000, y=-3000, course=45, speed=6, length=25, beam=7, height=8),
    ],
    weather=WeatherConditions(sea_state=2, wind_speed_knots=10, wind_direction=45),
    radar_range_nm=6.0,
)

# --- Busy Channel ---
BUSY_CHANNEL = Scenario(
    name="Busy Channel",
    description="Heavy traffic in a busy shipping channel.",
    vessels=[
        VesselConfig(id="own_ship", name="Own Ship", vessel_type=VesselType.OWN_SHIP,
                     x=0, y=0, course=90, speed=15, length=120, beam=18, height=22),
        VesselConfig(id="cargo_1", name="MV Atlantic Trader", vessel_type=VesselType.CARGO,
                     x=2000, y=1000, course=270, speed=16, length=200, beam=30, height=35),
        VesselConfig(id="cargo_2", name="MV Northern Star", vessel_type=VesselType.CARGO,
                     x=-1500, y=2000, course=90, speed=14, length=175, beam=25, height=28),
        VesselConfig(id="tanker_1", name="MT Crude Carrier", vessel_type=VesselType.TANKER,
                     x=3500, y=-500, course=260, speed=12, length=300, beam=50, height=30),
        VesselConfig(id="container_1", name="MV Box Express", vessel_type=VesselType.CARGO,
                     x=-3000, y=-1500, course=75, speed=18, length=350, beam=45, height=50),
        VesselConfig(id="tug_1", name="Harbour Tug 5", vessel_type=VesselType.TUG,
                     x=500, y=800, course=180, speed=8, length=30, beam=10, height=12),
        VesselConfig(id="pilot_1", name="Pilot Boat", vessel_type=VesselType.PILOT,
                     x=-800, y=-400, course=0, speed=20, length=15, beam=5, height=6),
    ],
    weather=WeatherConditions(sea_state=3, wind_speed_knots=15, wind_direction=180),
    radar_range_nm=6.0,
)

# --- Harbor Approach with terrain and coastline ---
HARBOR_APPROACH = Scenario(
    name="Harbor Approach",
    description="Approaching harbor with island terrain, coastline, buoys, and traffic.",
    vessels=[
        VesselConfig(id="own_ship", name="Own Ship", vessel_type=VesselType.OWN_SHIP,
                     x=0, y=0, course=0, speed=8, length=80, beam=12, height=15),
        VesselConfig(id="buoy_1", name="Port Entry Buoy", vessel_type=VesselType.BUOY,
                     x=-500, y=3000, course=0, speed=0, length=3, beam=3, height=5),
        VesselConfig(id="buoy_2", name="Stbd Entry Buoy", vessel_type=VesselType.BUOY,
                     x=500, y=3000, course=0, speed=0, length=3, beam=3, height=5),
        VesselConfig(id="ferry_1", name="Harbor Ferry", vessel_type=VesselType.PASSENGER,
                     x=-1500, y=4000, course=90, speed=12, length=60, beam=15, height=18),
        VesselConfig(id="tug_1", name="Harbor Tug", vessel_type=VesselType.TUG,
                     x=800, y=6000, course=180, speed=6, length=25, beam=8, height=10),
        VesselConfig(id="hidden_1", name="MV Docked", vessel_type=VesselType.CARGO,
                     x=3500, y=5500, course=0, speed=0, length=200, beam=30, height=35),
    ],
    weather=WeatherConditions(sea_state=2, wind_speed_knots=10, wind_direction=180),
    radar_range_nm=6.0,
    terrain=[
        TerrainConfig(type="island", center_x=3000, center_y=4500,
                      radius=600, peak_height=90, grid_size=48, cell_size=50),
    ],
    enable_coastline=True,
)

# --- Storm Conditions ---
STORM_CONDITIONS = Scenario(
    name="Storm Conditions",
    description="Heavy weather with rain and high seas. Practice clutter rejection.",
    vessels=[
        VesselConfig(id="own_ship", name="Own Ship", vessel_type=VesselType.OWN_SHIP,
                     x=0, y=0, course=180, speed=8, length=100, beam=15, height=20),
        VesselConfig(id="cargo_1", name="MV Weather Rider", vessel_type=VesselType.CARGO,
                     x=-2000, y=3000, course=90, speed=10, length=200, beam=30, height=35),
        VesselConfig(id="tanker_1", name="MT Storm Runner", vessel_type=VesselType.TANKER,
                     x=4000, y=1000, course=270, speed=8, length=250, beam=40, height=28),
    ],
    weather=WeatherConditions(sea_state=6, wind_speed_knots=35, wind_direction=225,
                              rain_rate_mmh=25, visibility_nm=2),
    radar_range_nm=6.0,
)

# --- Multi-Island Archipelago ---
ARCHIPELAGO = Scenario(
    name="Archipelago",
    description="Multiple islands creating complex occlusion patterns.",
    vessels=[
        VesselConfig(id="own_ship", name="Own Ship", vessel_type=VesselType.OWN_SHIP,
                     x=0, y=0, course=0, speed=10, length=80, beam=12, height=15),
        VesselConfig(id="cargo_1", name="MV Island Hopper", vessel_type=VesselType.CARGO,
                     x=2000, y=7000, course=180, speed=12, length=150, beam=22, height=25),
        VesselConfig(id="fishing_1", name="FV Channel Fisher", vessel_type=VesselType.FISHING,
                     x=-1000, y=4000, course=90, speed=4, length=18, beam=6, height=7),
        VesselConfig(id="sailing_1", name="SY Explorer", vessel_type=VesselType.SAILING,
                     x=4000, y=2000, course=315, speed=5, length=14, beam=4, height=16),
        VesselConfig(id="tanker_1", name="MT Deep Draft", vessel_type=VesselType.TANKER,
                     x=-5000, y=5000, course=135, speed=8, length=250, beam=40, height=25),
    ],
    weather=WeatherConditions(sea_state=2, wind_speed_knots=10, wind_direction=45),
    radar_range_nm=6.0,
    terrain=[
        TerrainConfig(type="island", center_x=1500, center_y=3000,
                      radius=500, peak_height=80, grid_size=48, cell_size=40),
        TerrainConfig(type="island", center_x=-2000, center_y=5000,
                      radius=700, peak_height=150, grid_size=48, cell_size=50),
        TerrainConfig(type="island", center_x=3000, center_y=6000,
                      radius=400, peak_height=60, grid_size=32, cell_size=40),
    ],
)

# --- Close Quarters with Terrain ---
CLOSE_QUARTERS_TERRAIN = Scenario(
    name="Close Quarters + Terrain",
    description="Multiple vessels in close proximity with an island complicating the picture.",
    vessels=[
        VesselConfig(id="own_ship", name="Own Ship", vessel_type=VesselType.OWN_SHIP,
                     x=0, y=0, course=45, speed=10, length=80, beam=12, height=15),
        VesselConfig(id="crossing_1", name="MV Crossing Ship", vessel_type=VesselType.CARGO,
                     x=1500, y=-500, course=315, speed=12, length=150, beam=22, height=25),
        VesselConfig(id="overtaking_1", name="MV Fast Cargo", vessel_type=VesselType.CARGO,
                     x=-200, y=-800, course=40, speed=16, length=180, beam=25, height=30),
        VesselConfig(id="head_on_1", name="MV Oncoming", vessel_type=VesselType.CARGO,
                     x=300, y=2000, course=225, speed=14, length=160, beam=23, height=28),
        VesselConfig(id="hidden_1", name="MV Shadowed", vessel_type=VesselType.CARGO,
                     x=2500, y=4000, course=270, speed=10, length=180, beam=25, height=28),
    ],
    weather=WeatherConditions(sea_state=2, wind_speed_knots=8, wind_direction=90),
    radar_range_nm=3.0,
    terrain=[
        TerrainConfig(type="island", center_x=1500, center_y=2500,
                      radius=500, peak_height=100, grid_size=48, cell_size=40),
    ],
)

# --- Coastal Navigation ---
COASTAL_NAVIGATION = Scenario(
    name="Coastal Navigation",
    description="Coastal waters with buoys, small craft, and a ridge along the coast.",
    vessels=[
        VesselConfig(id="own_ship", name="Own Ship", vessel_type=VesselType.OWN_SHIP,
                     x=0, y=0, course=315, speed=10, length=60, beam=10, height=12),
        VesselConfig(id="buoy_1", name="Fairway Buoy", vessel_type=VesselType.BUOY,
                     x=1000, y=1500, course=0, speed=0, length=3, beam=3, height=4),
        VesselConfig(id="buoy_2", name="Channel Marker", vessel_type=VesselType.BUOY,
                     x=-800, y=2000, course=0, speed=0, length=2, beam=2, height=3),
        VesselConfig(id="sailing_1", name="SY Wind Dancer", vessel_type=VesselType.SAILING,
                     x=-1500, y=1000, course=60, speed=6, length=12, beam=4, height=18),
        VesselConfig(id="fishing_1", name="FV Local Fisher", vessel_type=VesselType.FISHING,
                     x=-500, y=-1500, course=180, speed=4, length=15, beam=5, height=6),
        VesselConfig(id="fishing_2", name="FV Day Catch", vessel_type=VesselType.FISHING,
                     x=1200, y=-2000, course=45, speed=5, length=18, beam=6, height=7),
    ],
    weather=WeatherConditions(sea_state=2, wind_speed_knots=12, wind_direction=270),
    radar_range_nm=3.0,
    terrain=[
        TerrainConfig(type="ridge", center_x=-4000, center_y=3000,
                      end_x=2000, end_y=4000, width=400, peak_height=60,
                      grid_size=64, cell_size=50),
    ],
)

# --- Search and Rescue ---
SEARCH_AND_RESCUE = Scenario(
    name="Search and Rescue",
    description="SAR operation near islands with terrain-occluded search areas.",
    vessels=[
        VesselConfig(id="own_ship", name="Own Ship (OSC)", vessel_type=VesselType.OWN_SHIP,
                     x=0, y=0, course=0, speed=10, length=80, beam=12, height=15),
        VesselConfig(id="sar_1", name="Coast Guard 1", vessel_type=VesselType.PILOT,
                     x=1500, y=2000, course=90, speed=12, length=40, beam=8, height=10),
        VesselConfig(id="sar_2", name="Coast Guard 2", vessel_type=VesselType.PILOT,
                     x=-1500, y=2000, course=270, speed=12, length=40, beam=8, height=10),
        VesselConfig(id="tug_1", name="Rescue Tug", vessel_type=VesselType.TUG,
                     x=500, y=3000, course=180, speed=8, length=35, beam=10, height=12),
        VesselConfig(id="target_1", name="Life Raft", vessel_type=VesselType.BUOY,
                     x=800, y=5500, course=0, speed=0, length=2, beam=2, height=1),
    ],
    weather=WeatherConditions(sea_state=4, wind_speed_knots=20, wind_direction=315,
                              visibility_nm=4),
    radar_range_nm=6.0,
    terrain=[
        TerrainConfig(type="island", center_x=500, center_y=4000,
                      radius=600, peak_height=100, grid_size=48, cell_size=50),
    ],
)

# --- Fishing Fleet ---
FISHING_FLEET = Scenario(
    name="Fishing Fleet",
    description="Dense fishing fleet with many small erratically-moving vessels.",
    vessels=[
        VesselConfig(id="own_ship", name="Own Ship", vessel_type=VesselType.OWN_SHIP,
                     x=0, y=0, course=30, speed=14, length=120, beam=18, height=22),
        VesselConfig(id="fish_1", name="FV Net Hauler", vessel_type=VesselType.FISHING,
                     x=2000, y=3000, course=120, speed=2, length=20, beam=6, height=7),
        VesselConfig(id="fish_2", name="FV Trawler One", vessel_type=VesselType.FISHING,
                     x=2500, y=3200, course=135, speed=3, length=25, beam=8, height=8),
        VesselConfig(id="fish_3", name="FV Trawler Two", vessel_type=VesselType.FISHING,
                     x=1800, y=3500, course=90, speed=3, length=22, beam=7, height=7),
        VesselConfig(id="fish_4", name="FV Long Liner", vessel_type=VesselType.FISHING,
                     x=3000, y=2800, course=200, speed=4, length=28, beam=8, height=9),
        VesselConfig(id="fish_5", name="FV Seiner", vessel_type=VesselType.FISHING,
                     x=2200, y=2600, course=60, speed=1, length=30, beam=9, height=9),
        VesselConfig(id="fish_6", name="FV Small Boat", vessel_type=VesselType.FISHING,
                     x=1500, y=3800, course=270, speed=5, length=12, beam=4, height=5),
        VesselConfig(id="cargo_1", name="MV Passing Cargo", vessel_type=VesselType.CARGO,
                     x=-4000, y=5000, course=90, speed=16, length=200, beam=30, height=35),
    ],
    weather=WeatherConditions(sea_state=2, wind_speed_knots=10, wind_direction=90),
    radar_range_nm=6.0,
)

# --- Strait Transit with terrain on both sides ---
STRAIT_TRANSIT = Scenario(
    name="Strait Transit",
    description="Narrow strait passage with ridges on both sides and heavy traffic.",
    vessels=[
        VesselConfig(id="own_ship", name="Own Ship", vessel_type=VesselType.OWN_SHIP,
                     x=0, y=0, course=0, speed=12, length=150, beam=22, height=25),
        VesselConfig(id="cargo_1", name="MV Northbound", vessel_type=VesselType.CARGO,
                     x=800, y=-2000, course=0, speed=14, length=200, beam=30, height=32),
        VesselConfig(id="tanker_1", name="MT Southbound", vessel_type=VesselType.TANKER,
                     x=-800, y=4000, course=180, speed=11, length=280, beam=44, height=28),
        VesselConfig(id="ferry_1", name="Cross Strait Ferry", vessel_type=VesselType.PASSENGER,
                     x=3000, y=2000, course=270, speed=18, length=100, beam=20, height=25),
        VesselConfig(id="hidden_1", name="MV Beyond Ridge", vessel_type=VesselType.CARGO,
                     x=-5000, y=3000, course=0, speed=10, length=180, beam=25, height=28),
    ],
    weather=WeatherConditions(sea_state=3, wind_speed_knots=18, wind_direction=90),
    radar_range_nm=12.0,
    terrain=[
        TerrainConfig(type="ridge", center_x=-4000, center_y=0,
                      end_x=-3500, end_y=6000, width=500, peak_height=90,
                      grid_size=64, cell_size=50),
        TerrainConfig(type="ridge", center_x=4000, center_y=0,
                      end_x=3500, end_y=6000, width=500, peak_height=70,
                      grid_size=64, cell_size=50),
    ],
)

PRESET_SCENARIOS = [
    ISLAND_OCCLUSION,
    RIDGE_BLOCKAGE,
    ARCHIPELAGO,
    CLOSE_QUARTERS_TERRAIN,
    HARBOR_APPROACH,
    STRAIT_TRANSIT,
    SEARCH_AND_RESCUE,
    CALM_TRAFFIC,
    BUSY_CHANNEL,
    STORM_CONDITIONS,
    COASTAL_NAVIGATION,
    FISHING_FLEET,
]
