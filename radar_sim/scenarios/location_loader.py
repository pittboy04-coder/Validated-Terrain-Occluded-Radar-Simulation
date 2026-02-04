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
    _closed_indices: set = field(default_factory=set)  # Indices of closed polygons


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

    # Coastlines - track which ones are closed for terrain generation
    closed_indices = set()
    for idx, coast_data in enumerate(doc.get("coastlines", [])):
        points = coast_data.get("points", [])
        if len(points) < 3:
            continue
        coastline = Coastline()
        for pt in points:
            coastline.add_point(pt["x"], pt["y"])
        # Close polygon if flagged
        is_closed = coast_data.get("closed", False)
        if is_closed and len(points) >= 3:
            first = points[0]
            last = points[-1]
            if abs(first["x"] - last["x"]) > 0.1 or abs(first["y"] - last["y"]) > 0.1:
                coastline.add_point(first["x"], first["y"])
            closed_indices.add(len(result.coastlines))
        result.coastlines.append(coastline)

    # Store closed indices for terrain generation
    result._closed_indices = closed_indices

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
    # Only use CLOSED polygons for terrain fill (unclosed are shoreline segments)
    if result.coastlines:
        coastline_terrain = _terrain_from_coastlines(
            result.coastlines, result.range_nm,
            closed_indices=result._closed_indices)
        if result.terrain is None:
            result.terrain = coastline_terrain
        elif coastline_terrain is not None:
            # Merge: land cells get max(explicit_elevation, 13m)
            result.terrain = _merge_terrain(result.terrain, coastline_terrain)

    return result


def _merge_terrain(explicit: HeightMap, coastline: HeightMap) -> HeightMap:
    """Merge explicit elevation data with coastline-derived terrain.

    Rules:
    - Water areas (coastline=0): Always use 0m (water overrides explicit terrain)
    - Land areas (coastline>0): Use max(explicit, coastline) for occlusion
    """
    # Resample coastline grid to match explicit grid dimensions
    ex_grid = explicit.grid
    co_grid = coastline.grid
    ex_rows, ex_cols = ex_grid.shape
    co_rows, co_cols = co_grid.shape

    # Resample coastline if needed
    if ex_rows != co_rows or ex_cols != co_cols:
        from scipy.ndimage import zoom
        scale_r = ex_rows / co_rows
        scale_c = ex_cols / co_cols
        try:
            co_grid = zoom(co_grid, (scale_r, scale_c), order=0)[:ex_rows, :ex_cols]
        except ImportError:
            # No scipy - use nearest neighbor manually
            co_resampled = np.zeros_like(ex_grid)
            for r in range(ex_rows):
                src_r = min(int(r / scale_r), co_rows - 1)
                for c in range(ex_cols):
                    src_c = min(int(c / scale_c), co_cols - 1)
                    co_resampled[r, c] = coastline.grid[src_r, src_c]
            co_grid = co_resampled

    # Merge: water (coastline=0) stays 0, land uses max elevation
    merged = np.where(co_grid == 0, 0.0, np.maximum(ex_grid, co_grid))

    return HeightMap(explicit.config, merged.astype(np.float32))


def _polygon_area(points) -> float:
    """Calculate signed area of a polygon using the shoelace formula."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i].x * points[j].y
        area -= points[j].x * points[i].y
    return abs(area) / 2.0


def _point_in_polygon(px: float, py: float, points: list) -> bool:
    """Ray casting algorithm to test if point is inside polygon."""
    n = len(points)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = points[i].x, points[i].y
        xj, yj = points[j].x, points[j].y
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _scanline_fill_polygon(grid: np.ndarray, points: list,
                           origin_x: float, origin_y: float,
                           cell_size: float, fill_value: float) -> None:
    """Fill a single polygon into the grid using scanline algorithm."""
    grid_size = grid.shape[0]
    n = len(points)
    if n < 3:
        return

    for r in range(grid_size):
        wy = origin_y + (r + 0.5) * cell_size
        crossings = []

        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            y1, y2 = p1.y, p2.y
            x1, x2 = p1.x, p2.x

            if (y1 <= wy < y2) or (y2 <= wy < y1):
                if abs(y2 - y1) > 1e-10:
                    t = (wy - y1) / (y2 - y1)
                    ix = x1 + t * (x2 - x1)
                    crossings.append(ix)

        if len(crossings) >= 2:
            crossings.sort()
            for i in range(0, len(crossings) - 1, 2):
                x_start = crossings[i]
                x_end = crossings[i + 1]
                c_start = max(0, int((x_start - origin_x) / cell_size))
                c_end = min(grid_size, int((x_end - origin_x) / cell_size) + 1)
                grid[r, c_start:c_end] = fill_value


def _terrain_from_coastlines(coastlines: list, range_nm: float,
                             land_height: float = 13.0,
                             grid_size: int = 192,  # Balance of accuracy vs speed
                             closed_indices: set = None) -> HeightMap:
    """Generate a terrain height map by rasterizing coastline polygons.

    Handles two cases:
    1. Lakes/reservoirs: Many closed polygons that ENCLOSE water
       → Start with land, fill polygons as water
    2. Ocean coastlines: Mostly open ways marking shore boundary
       → Start with water at center, flood-fill land from coastlines

    Detection: If closed polygons cover significant area, use lake mode.
    Otherwise, use ocean coastline mode with flood-fill.

    Args:
        coastlines: List of Coastline objects.
        range_nm: Radar range in nautical miles (defines grid extent).
        land_height: Default land elevation in meters.
        grid_size: Grid resolution (rows and cols).
        closed_indices: Set of indices indicating which coastlines are closed.

    Returns:
        HeightMap with land cells elevated, water cells at 0.
    """
    from collections import deque

    range_m = range_nm * 1852.0
    cell_size = (2 * range_m) / grid_size
    origin_x = -range_m
    origin_y = -range_m

    if closed_indices is None:
        closed_indices = set(range(len(coastlines)))

    # Collect closed polygons with bounding boxes
    closed_polygons = []
    total_closed_area = 0.0
    for idx, coastline in enumerate(coastlines):
        if idx not in closed_indices:
            continue
        pts = coastline.points
        if len(pts) >= 3:
            xs = [p.x for p in pts]
            ys = [p.y for p in pts]
            bbox = (min(xs), max(xs), min(ys), max(ys))
            # Calculate polygon area
            area = _polygon_area(pts)
            total_closed_area += area
            closed_polygons.append((pts, bbox, area))

    # Calculate total grid area
    grid_area = (2 * range_m) ** 2

    # Decide mode: if closed polygons cover >5% of grid, use lake mode
    # Otherwise, use ocean coastline mode
    use_lake_mode = total_closed_area > grid_area * 0.05 and closed_polygons

    if use_lake_mode:
        # LAKE MODE: Closed polygons enclose water
        grid = np.full((grid_size, grid_size), land_height, dtype=np.float32)

        # Sort by area descending
        closed_polygons.sort(key=lambda x: x[2], reverse=True)

        # Fill closed polygons as water
        for r in range(grid_size):
            wy = origin_y + (r + 0.5) * cell_size
            for c in range(grid_size):
                wx = origin_x + (c + 0.5) * cell_size
                for pts, bbox, _ in closed_polygons:
                    if wx < bbox[0] or wx > bbox[1] or wy < bbox[2] or wy > bbox[3]:
                        continue
                    if _point_in_polygon(wx, wy, pts):
                        grid[r, c] = 0.0
                        break
    else:
        # OCEAN COASTLINE MODE: Water at center, land along coastlines
        # Start with water, then flood-fill land from coastline edges
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Draw all coastline segments as barriers
        def world_to_grid(wx, wy):
            c = int((wx - origin_x) / cell_size)
            r = int((wy - origin_y) / cell_size)
            return max(0, min(grid_size-1, r)), max(0, min(grid_size-1, c))

        barrier = np.zeros((grid_size, grid_size), dtype=np.uint8)

        for coastline in coastlines:
            pts = coastline.points
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                r1, c1 = world_to_grid(pts[i].x, pts[i].y)
                r2, c2 = world_to_grid(pts[i+1].x, pts[i+1].y)
                # Bresenham line
                dr, dc = abs(r2-r1), abs(c2-c1)
                sr, sc = (1 if r1 < r2 else -1), (1 if c1 < c2 else -1)
                err = dr - dc
                while True:
                    barrier[r1, c1] = 1
                    grid[r1, c1] = land_height  # Coastline itself is land
                    if r1 == r2 and c1 == c2:
                        break
                    e2 = 2 * err
                    if e2 > -dc:
                        err -= dc
                        r1 += sr
                    if e2 < dr:
                        err += dr
                        c1 += sc

        # Flood-fill land from grid edges (outside = land)
        visited = np.zeros((grid_size, grid_size), dtype=np.uint8)
        queue = deque()

        for i in range(grid_size):
            for r, c in [(0, i), (grid_size-1, i), (i, 0), (i, grid_size-1)]:
                if barrier[r, c] == 0 and visited[r, c] == 0:
                    queue.append((r, c))
                    visited[r, c] = 1
                    grid[r, c] = land_height

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    if visited[nr, nc] == 0 and barrier[nr, nc] == 0:
                        visited[nr, nc] = 1
                        grid[nr, nc] = land_height
                        queue.append((nr, nc))

    # Return terrain if there's both land and water
    has_land = np.any(grid > 0)
    has_water = np.any(grid == 0)
    if has_land and has_water:
        config = TerrainConfig(
            origin_x=origin_x,
            origin_y=origin_y,
            cell_size=cell_size,
        )
        return HeightMap(config, grid)
    return None
