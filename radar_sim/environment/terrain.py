"""Terrain height-map and elevation model for radar simulation."""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List


@dataclass
class TerrainConfig:
    """Configuration for a terrain height map."""
    origin_x: float = 0.0       # World X of grid cell (0,0)
    origin_y: float = 0.0       # World Y of grid cell (0,0)
    cell_size: float = 50.0     # Meters per grid cell
    reflectivity: float = 0.85  # Radar reflectivity of terrain surface
    roughness: float = 0.3      # Surface roughness (affects return spread)


class HeightMap:
    """Numpy-backed elevation grid with bilinear interpolation."""

    def __init__(self, config: TerrainConfig, grid: np.ndarray):
        """
        Args:
            config: Terrain configuration (origin, cell size, etc.)
            grid: 2-D float32 array of elevations in meters. Shape (rows, cols)
                  where row index increases with Y and col index increases with X.
        """
        self.config = config
        self.grid = grid.astype(np.float32)
        self.rows, self.cols = self.grid.shape

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def from_array(cls, elevations: np.ndarray,
                   origin_x: float = 0.0, origin_y: float = 0.0,
                   cell_size: float = 50.0, **kwargs) -> "HeightMap":
        """Create a HeightMap from an existing numpy elevation array.

        This is the hook for loading SRTM / NASADEM tiles.
        """
        cfg = TerrainConfig(origin_x=origin_x, origin_y=origin_y,
                            cell_size=cell_size, **kwargs)
        return cls(cfg, elevations)

    @classmethod
    def from_generator(cls, rows: int, cols: int,
                       generator: Callable[[int, int], float],
                       origin_x: float = 0.0, origin_y: float = 0.0,
                       cell_size: float = 50.0, **kwargs) -> "HeightMap":
        """Create a HeightMap from a procedural generator function.

        Args:
            rows, cols: Grid dimensions.
            generator: Callable(row, col) -> elevation_m.
        """
        grid = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                grid[r, c] = generator(r, c)
        cfg = TerrainConfig(origin_x=origin_x, origin_y=origin_y,
                            cell_size=cell_size, **kwargs)
        return cls(cfg, grid)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def get_elevation(self, wx: float, wy: float) -> float:
        """Return bilinear-interpolated elevation at world coordinates (wx, wy).

        Returns 0.0 for queries outside the grid.
        """
        # Convert world coords to fractional grid coords
        fc = (wx - self.config.origin_x) / self.config.cell_size
        fr = (wy - self.config.origin_y) / self.config.cell_size

        if fr < 0 or fc < 0 or fr >= self.rows - 1 or fc >= self.cols - 1:
            return 0.0

        r0 = int(fr)
        c0 = int(fc)
        dr = fr - r0
        dc = fc - c0

        e00 = self.grid[r0, c0]
        e01 = self.grid[r0, c0 + 1]
        e10 = self.grid[r0 + 1, c0]
        e11 = self.grid[r0 + 1, c0 + 1]

        return float(
            e00 * (1 - dr) * (1 - dc) +
            e01 * (1 - dr) * dc +
            e10 * dr * (1 - dc) +
            e11 * dr * dc
        )

    @property
    def world_extent(self):
        """Return (min_x, min_y, max_x, max_y) in world coordinates."""
        return (
            self.config.origin_x,
            self.config.origin_y,
            self.config.origin_x + (self.cols - 1) * self.config.cell_size,
            self.config.origin_y + (self.rows - 1) * self.config.cell_size,
        )


# ======================================================================
# Factory helpers
# ======================================================================

def create_island_terrain(center_x: float, center_y: float,
                          radius: float = 800.0, peak_height: float = 120.0,
                          grid_size: int = 64, cell_size: float = 50.0,
                          seed: int = 42) -> HeightMap:
    """Create a conical island terrain centred at (center_x, center_y).

    The island has a smooth falloff from peak to sea level at *radius*.
    """
    rng = np.random.RandomState(seed)
    half_extent = grid_size * cell_size / 2
    origin_x = center_x - half_extent
    origin_y = center_y - half_extent

    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r in range(grid_size):
        for c in range(grid_size):
            wx = origin_x + c * cell_size
            wy = origin_y + r * cell_size
            dist = math.sqrt((wx - center_x) ** 2 + (wy - center_y) ** 2)
            if dist < radius:
                # Smooth cosine falloff
                t = dist / radius
                elev = peak_height * (0.5 * (1 + math.cos(math.pi * t)))
                # Add a little noise for realism
                elev += rng.normal(0, peak_height * 0.02)
                grid[r, c] = max(0.0, elev)

    return HeightMap.from_array(grid, origin_x=origin_x, origin_y=origin_y,
                                cell_size=cell_size)


def create_ridge_terrain(start_x: float, start_y: float,
                         end_x: float, end_y: float,
                         width: float = 600.0, peak_height: float = 80.0,
                         grid_size: int = 64, cell_size: float = 50.0,
                         seed: int = 42) -> HeightMap:
    """Create a linear ridge between two endpoints."""
    rng = np.random.RandomState(seed)
    # Bounding box with margin
    margin = width + cell_size * 4
    min_x = min(start_x, end_x) - margin
    min_y = min(start_y, end_y) - margin

    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    ridge_dx = end_x - start_x
    ridge_dy = end_y - start_y
    ridge_len = math.sqrt(ridge_dx ** 2 + ridge_dy ** 2)
    if ridge_len < 1e-6:
        return HeightMap.from_array(grid, origin_x=min_x, origin_y=min_y,
                                    cell_size=cell_size)

    ux, uy = ridge_dx / ridge_len, ridge_dy / ridge_len  # unit along ridge

    for r in range(grid_size):
        for c in range(grid_size):
            wx = min_x + c * cell_size
            wy = min_y + r * cell_size
            # Project onto ridge axis
            dx = wx - start_x
            dy = wy - start_y
            along = dx * ux + dy * uy
            across = abs(-dx * uy + dy * ux)

            if 0 <= along <= ridge_len and across < width:
                t = across / width
                elev = peak_height * (0.5 * (1 + math.cos(math.pi * t)))
                elev += rng.normal(0, peak_height * 0.02)
                grid[r, c] = max(0.0, elev)

    return HeightMap.from_array(grid, origin_x=min_x, origin_y=min_y,
                                cell_size=cell_size)
