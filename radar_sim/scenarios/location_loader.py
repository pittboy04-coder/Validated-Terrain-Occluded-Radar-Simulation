"""Load .radarloc location files into the radar simulator."""
import json
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from ..environment.coastline import Coastline, CoastlinePoint
from ..environment.terrain import HeightMap, TerrainConfig
from ..objects.vessel import Vessel, VesselType


@dataclass
class RadarLocationData:
    """Parsed contents of a .radarloc file."""
    location_name: str = ""
    center_lat: float = 0.0
    center_lon: float = 0.0
    range_nm: float = 6.0
    coastlines: list = field(default_factory=list)
    terrain: Optional[HeightMap] = None
    vessels: list = field(default_factory=list)


def load_radarloc_file(filepath: str) -> RadarLocationData:
    """Load and parse a .radarloc JSON file.

    Args:
        filepath: Path to the .radarloc file.

    Returns:
        RadarLocationData with parsed coastlines, terrain, and vessels.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is invalid.
        json.JSONDecodeError: If JSON is malformed.
    """
    with open(filepath, "r") as f:
        doc = json.load(f)

    version = doc.get("version", "")
    if not version.startswith("1."):
        raise ValueError(f"Unsupported .radarloc version: {version}")

    result = RadarLocationData()

    # Metadata
    meta = doc.get("metadata", {})
    result.location_name = meta.get("location_name", "Unknown Location")
    result.center_lat = meta.get("center_lat", 0.0)
    result.center_lon = meta.get("center_lon", 0.0)
    result.range_nm = meta.get("range_nm", 6.0)

    # Coastlines
    for coast_data in doc.get("coastlines", []):
        points = coast_data.get("points", [])
        if len(points) < 3:
            continue
        coastline = Coastline()
        for pt in points:
            coastline.add_point(pt["x"], pt["y"])
        # Close polygon if flagged
        if coast_data.get("closed", False) and len(points) >= 3:
            first = points[0]
            last = points[-1]
            if abs(first["x"] - last["x"]) > 0.1 or abs(first["y"] - last["y"]) > 0.1:
                coastline.add_point(first["x"], first["y"])
        result.coastlines.append(coastline)

    # Terrain
    terrain_data = doc.get("terrain", {})
    if terrain_data.get("enabled", False) and "grid" in terrain_data:
        grid_info = terrain_data["grid"]
        elevations = terrain_data.get("elevations", [])
        if elevations:
            grid_array = np.array(elevations, dtype=np.float32)
            config = TerrainConfig(
                origin_x=grid_info.get("origin_x", 0.0),
                origin_y=grid_info.get("origin_y", 0.0),
                cell_size=grid_info.get("cell_size", 50.0),
            )
            result.terrain = HeightMap(config, grid_array)

    # Vessels
    vessel_type_map = {
        "cargo": VesselType.CARGO,
        "tanker": VesselType.TANKER,
        "fishing": VesselType.FISHING,
        "sailing": VesselType.SAILING,
        "passenger": VesselType.PASSENGER,
        "tug": VesselType.TUG,
        "pilot": VesselType.PILOT,
        "buoy": VesselType.BUOY,
    }
    for v_data in doc.get("vessels", []):
        vtype = vessel_type_map.get(v_data.get("type", ""), VesselType.UNKNOWN)
        vessel = Vessel(
            id=v_data.get("id", "vessel"),
            name=v_data.get("name", ""),
            vessel_type=vtype,
            x=v_data.get("x", 0.0),
            y=v_data.get("y", 0.0),
            course=v_data.get("course", 0.0),
            speed=v_data.get("speed", 0.0),
        )
        result.vessels.append(vessel)

    return result
