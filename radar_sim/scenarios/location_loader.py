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

    # Ensure coastlines create occlusion terrain.
    # - No explicit terrain: generate 13m land from coastline polygons
    # - Explicit terrain: merge 13m minimum for land areas (use max of real vs 13m)
    if result.coastlines:
        coastline_terrain = _terrain_from_coastlines(
            result.coastlines, result.range_nm)
        if result.terrain is None:
            result.terrain = coastline_terrain
        elif coastline_terrain is not None:
            # Merge: land cells get max(explicit_elevation, 13m)
            result.terrain = _merge_terrain(result.terrain, coastline_terrain)

    return result


def _merge_terrain(explicit: HeightMap, coastline: HeightMap) -> HeightMap:
    """Merge explicit elevation data with coastline-derived terrain.

    For each cell, use the maximum of explicit elevation and coastline
    elevation (13m for land, 0 for water). This ensures land areas have
    at least 13m elevation for occlusion even if explicit data is lower.
    """
    # Resample coastline grid to match explicit grid dimensions
    ex_grid = explicit.grid
    co_grid = coastline.grid
    ex_rows, ex_cols = ex_grid.shape
    co_rows, co_cols = co_grid.shape

    # If grids match, simple element-wise max
    if ex_rows == co_rows and ex_cols == co_cols:
        merged = np.maximum(ex_grid, co_grid)
    else:
        # Resample coastline to explicit grid size
        from scipy.ndimage import zoom
        scale_r = ex_rows / co_rows
        scale_c = ex_cols / co_cols
        try:
            co_resampled = zoom(co_grid, (scale_r, scale_c), order=0)
            merged = np.maximum(ex_grid, co_resampled[:ex_rows, :ex_cols])
        except ImportError:
            # No scipy - use nearest neighbor manually
            co_resampled = np.zeros_like(ex_grid)
            for r in range(ex_rows):
                src_r = min(int(r / scale_r), co_rows - 1)
                for c in range(ex_cols):
                    src_c = min(int(c / scale_c), co_cols - 1)
                    co_resampled[r, c] = co_grid[src_r, src_c]
            merged = np.maximum(ex_grid, co_resampled)

    return HeightMap(explicit.config, merged.astype(np.float32))


def _terrain_from_coastlines(coastlines: list, range_nm: float,
                             land_height: float = 13.0,
                             grid_size: int = 128) -> HeightMap:
    """Generate a terrain height map by rasterizing coastline polygons.

    Water features (lakes, rivers, ocean) are defined by polygons that
    enclose WATER, not land. So we start with the entire grid as land,
    then carve out water areas where the polygons are.

    Args:
        coastlines: List of Coastline objects (polygons enclosing water).
        range_nm: Radar range in nautical miles (defines grid extent).
        land_height: Default land elevation in meters.
        grid_size: Grid resolution (rows and cols).

    Returns:
        HeightMap with land cells elevated, water cells at 0.
    """
    range_m = range_nm * 1852.0
    cell_size = (2 * range_m) / grid_size
    origin_x = -range_m
    origin_y = -range_m

    # Start with all land
    grid = np.full((grid_size, grid_size), land_height, dtype=np.float32)

    # Carve out water areas (inside polygons = water = 0)
    for coastline in coastlines:
        pts = coastline.points
        if len(pts) < 3:
            continue
        # Scanline fill: for each grid row, find edge crossings
        for r in range(grid_size):
            wy = origin_y + (r + 0.5) * cell_size
            crossings = []
            n = len(pts)
            for i in range(n):
                p1 = pts[i]
                p2 = pts[(i + 1) % n]
                # Check if this edge crosses the scanline
                if (p1.y <= wy < p2.y) or (p2.y <= wy < p1.y):
                    # X coordinate of intersection
                    t = (wy - p1.y) / (p2.y - p1.y)
                    ix = p1.x + t * (p2.x - p1.x)
                    crossings.append(ix)
            crossings.sort()
            # Fill between pairs (even-odd rule) with WATER (0)
            for i in range(0, len(crossings) - 1, 2):
                x_start = crossings[i]
                x_end = crossings[i + 1]
                c_start = max(0, int((x_start - origin_x) / cell_size))
                c_end = min(grid_size, int((x_end - origin_x) / cell_size) + 1)
                grid[r, c_start:c_end] = 0.0

    # Only return if there's actually some water carved out
    if np.any(grid == 0):
        config = TerrainConfig(
            origin_x=origin_x,
            origin_y=origin_y,
            cell_size=cell_size,
        )
        return HeightMap(config, grid)
    return None
