"""Terrain occlusion engine — LOS checks and terrain radar returns."""
import math
from typing import List, Optional
from .terrain import HeightMap


class OcclusionEngine:
    """Computes line-of-sight occlusion and terrain radar returns."""

    def __init__(self, terrain_maps: List[HeightMap],
                 antenna_height_m: float = 15.0):
        """
        Args:
            terrain_maps: List of HeightMap objects in the scene.
            antenna_height_m: Radar antenna height above sea level.
        """
        self.terrain_maps = terrain_maps
        self.antenna_height_m = antenna_height_m

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _sample_elevation(self, wx: float, wy: float) -> float:
        """Get the highest terrain elevation at a world point across all maps."""
        best = 0.0
        for hm in self.terrain_maps:
            e = hm.get_elevation(wx, wy)
            if e > best:
                best = e
        return best

    # ------------------------------------------------------------------
    # Core algorithms
    # ------------------------------------------------------------------
    def compute_occlusion_profile(self, origin_x: float, origin_y: float,
                                  bearing_deg: float, max_range_m: float,
                                  step_m: float = 50.0) -> List[float]:
        """Ray-march along a bearing and return the max elevation angle seen so far
        at each step.  Used to build shadow masks.

        Returns:
            List of max-elevation-angle (degrees) at each sample point.
        """
        ray_rad = math.radians(bearing_deg)
        dx = math.sin(ray_rad)
        dy = math.cos(ray_rad)

        profile = []
        max_angle = -90.0
        dist = step_m
        while dist <= max_range_m:
            wx = origin_x + dx * dist
            wy = origin_y + dy * dist
            elev = self._sample_elevation(wx, wy)
            # Elevation angle from antenna to terrain point
            angle = math.degrees(math.atan2(elev - self.antenna_height_m, dist))
            if angle > max_angle:
                max_angle = angle
            profile.append(max_angle)
            dist += step_m
        return profile

    def is_target_occluded(self, origin_x: float, origin_y: float,
                           target_x: float, target_y: float,
                           target_height_m: float = 10.0,
                           step_m: float = 50.0) -> bool:
        """Check whether terrain blocks line-of-sight to a target.

        Args:
            origin_x, origin_y: Radar antenna position.
            target_x, target_y: Target position.
            target_height_m: Height of target above sea level.
            step_m: Ray-march step size.

        Returns:
            True if the target is occluded by terrain.
        """
        dx = target_x - origin_x
        dy = target_y - origin_y
        target_range = math.sqrt(dx * dx + dy * dy)
        if target_range < step_m:
            return False

        # Angle from antenna to top of target
        target_angle = math.degrees(
            math.atan2(target_height_m - self.antenna_height_m, target_range)
        )

        # March along the ray, track max terrain angle
        bearing_rad = math.atan2(dx, dy)
        ux = math.sin(bearing_rad)
        uy = math.cos(bearing_rad)

        max_terrain_angle = -90.0
        dist = step_m
        while dist < target_range:
            wx = origin_x + ux * dist
            wy = origin_y + uy * dist
            elev = self._sample_elevation(wx, wy)
            angle = math.degrees(math.atan2(elev - self.antenna_height_m, dist))
            if angle > max_terrain_angle:
                max_terrain_angle = angle
            dist += step_m

        return max_terrain_angle > target_angle

    def generate_terrain_returns(self, origin_x: float, origin_y: float,
                                 bearing_deg: float, beamwidth_deg: float,
                                 max_range_m: float, num_bins: int) -> List[float]:
        """Generate radar return intensities from terrain along a bearing.

        Casts 5 sub-rays across the beamwidth. Terrain behind higher terrain
        is shadowed (no return).

        Returns:
            List of intensities (0-1) for each range bin.
        """
        returns = [0.0] * num_bins
        bin_size = max_range_m / num_bins
        half_beam = beamwidth_deg / 2.0
        num_rays = 5

        for ray_idx in range(num_rays):
            if num_rays > 1:
                offset = -half_beam + (2 * half_beam * ray_idx / (num_rays - 1))
            else:
                offset = 0.0
            ray_bearing = bearing_deg + offset
            ray_weight = 1.0 - 0.3 * abs(offset) / (half_beam + 0.01)

            ray_rad = math.radians(ray_bearing)
            ux = math.sin(ray_rad)
            uy = math.cos(ray_rad)

            max_angle = -90.0
            for b in range(num_bins):
                dist = (b + 0.5) * bin_size
                wx = origin_x + ux * dist
                wy = origin_y + uy * dist
                elev = self._sample_elevation(wx, wy)

                if elev <= 0.0:
                    continue

                angle = math.degrees(math.atan2(elev - self.antenna_height_m, dist))

                if angle >= max_angle:
                    # Visible terrain — produce a return
                    max_angle = angle
                    # Stronger return for steeper facing terrain
                    reflectivity = 0.85
                    intensity = reflectivity * ray_weight * min(1.0, elev / 50.0)
                    returns[b] = max(returns[b], intensity)
                # else: shadowed — no return for this sub-ray

        return returns

    def get_occlusion_mask(self, origin_x: float, origin_y: float,
                           bearing_deg: float, max_range_m: float,
                           num_bins: int) -> List[bool]:
        """Return a boolean shadow mask for a single bearing.

        True means the range bin is in shadow (occluded).
        """
        bin_size = max_range_m / num_bins
        ray_rad = math.radians(bearing_deg)
        ux = math.sin(ray_rad)
        uy = math.cos(ray_rad)

        mask = [False] * num_bins
        max_angle = -90.0

        for b in range(num_bins):
            dist = (b + 0.5) * bin_size
            wx = origin_x + ux * dist
            wy = origin_y + uy * dist
            elev = self._sample_elevation(wx, wy)

            angle = math.degrees(math.atan2(elev - self.antenna_height_m, dist))

            if elev > 0 and angle >= max_angle:
                max_angle = angle
                # This bin has visible terrain — not shadowed
            elif max_angle > -90.0:
                # Check if this point is below the shadow line
                shadow_elev = self.antenna_height_m + dist * math.tan(math.radians(max_angle))
                if elev < shadow_elev:
                    mask[b] = True

        return mask
