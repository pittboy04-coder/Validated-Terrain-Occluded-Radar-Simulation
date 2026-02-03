"""Top-down geographic scene view with terrain elevation rendering and drag interaction."""
import math
import pygame
from typing import List, Tuple, Optional


NM_TO_M = 1852.0

COLORS = {
    'water': (20, 40, 80),
    'land': (120, 100, 60),
    'own_ship': (255, 255, 0),
    'vessel': (255, 100, 100),
    'vessel_label': (255, 180, 180),
    'range_ring': (60, 80, 120),
    'grid': (40, 60, 100),
    'north_arrow': (255, 255, 255),
    'echo': (0, 200, 0),
    'title': (180, 200, 220),
    'terrain_low': (60, 80, 40),
    'terrain_high': (180, 160, 100),
    'occluded': (80, 20, 20),
    'selected': (255, 255, 0),
    'placement': (0, 255, 255),
}

# Hit-test radius in pixels
HIT_RADIUS_VESSEL = 14
HIT_RADIUS_TERRAIN = 18


class SceneView:
    """Top-down map view with drag-and-drop for vessels and terrain."""

    def __init__(self, size: int = 600):
        self.size = size
        self.center = (size // 2, size // 2)
        self.radius = size // 2 - 20
        self.surface = pygame.Surface((size, size))
        self.echo_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        self.echo_surface.fill((0, 0, 0, 0))
        self._font = None
        self._small_font = None

        # Offset of scene view on the main screen
        self.offset_x = 0
        self.offset_y = 0

        # Cached rendering state (set each frame by render())
        self._own_x = 0.0
        self._own_y = 0.0
        self._zoom = 1.0
        self._max_range_m = 11112.0

        # --- Interaction state ---
        # Selection: ("vessel", vessel_id) or ("terrain", terrain_index) or None
        self.selected = None
        self.is_dragging = False
        self._drag_offset_wx = 0.0  # world offset from object center to grab point
        self._drag_offset_wy = 0.0

        # Placement mode: set to a vessel type string to place on next click
        self.placement_mode: Optional[str] = None

    def set_offset(self, x: int, y: int) -> None:
        self.offset_x = x
        self.offset_y = y

    def _get_font(self, size: int = 16):
        if self._font is None:
            self._font = pygame.font.SysFont('consolas', 16)
            self._small_font = pygame.font.SysFont('consolas', 12)
        if size <= 12:
            return self._small_font
        return self._font

    def _world_to_screen(self, wx: float, wy: float,
                          own_x: float, own_y: float,
                          zoom: float) -> Tuple[int, int]:
        dx = wx - own_x
        dy = wy - own_y
        sx = self.center[0] + dx * zoom
        sy = self.center[1] - dy * zoom
        return int(sx), int(sy)

    def _screen_to_world(self, sx: int, sy: int) -> Tuple[float, float]:
        """Convert scene-local screen coordinates to world coordinates."""
        wx = self._own_x + (sx - self.center[0]) / self._zoom
        wy = self._own_y - (sy - self.center[1]) / self._zoom
        return wx, wy

    def _global_to_local(self, gx: int, gy: int) -> Tuple[int, int]:
        """Convert global screen coordinates to scene-view-local coordinates."""
        return gx - self.offset_x, gy - self.offset_y

    def _is_inside(self, gx: int, gy: int) -> bool:
        lx, ly = self._global_to_local(gx, gy)
        return 0 <= lx < self.size and 0 <= ly < self.size

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def _hit_test(self, local_x: int, local_y: int,
                  simulation) -> Optional[tuple]:
        """Find what's under the cursor. Returns ("vessel", id) or ("terrain", idx) or None."""
        if simulation is None:
            return None

        own_ship = simulation.world.own_ship

        # Check vessels first (smaller, higher priority)
        for vessel in simulation.world.get_all_vessels():
            if vessel is own_ship or not vessel.is_active:
                continue
            vx, vy = self._world_to_screen(vessel.x, vessel.y,
                                            self._own_x, self._own_y, self._zoom)
            dist = math.sqrt((local_x - vx) ** 2 + (local_y - vy) ** 2)
            if dist < HIT_RADIUS_VESSEL:
                return ("vessel", vessel.id)

        # Check terrain centers
        for i, hm in enumerate(simulation.terrain_maps):
            cx, cy = hm.center
            tx, ty = self._world_to_screen(cx, cy,
                                            self._own_x, self._own_y, self._zoom)
            dist = math.sqrt((local_x - tx) ** 2 + (local_y - ty) ** 2)
            if dist < HIT_RADIUS_TERRAIN:
                return ("terrain", i)

        return None

    # ------------------------------------------------------------------
    # Mouse event handling
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event, simulation) -> bool:
        """Handle mouse events for drag-and-drop and placement.

        Returns True if the event was consumed.
        """
        if simulation is None:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self._is_inside(event.pos[0], event.pos[1]):
                return False

            lx, ly = self._global_to_local(event.pos[0], event.pos[1])

            # Placement mode: place a new vessel
            if self.placement_mode is not None:
                wx, wy = self._screen_to_world(lx, ly)
                simulation.add_vessel_at(self.placement_mode, wx, wy)
                # Stay in placement mode so user can place multiple
                return True

            # Hit test for selection/drag
            hit = self._hit_test(lx, ly, simulation)
            if hit is not None:
                self.selected = hit
                self.is_dragging = True

                # Compute drag offset (world-space distance from object center to click)
                wx, wy = self._screen_to_world(lx, ly)
                if hit[0] == "vessel":
                    vessel = simulation.world.get_vessel(hit[1])
                    if vessel:
                        self._drag_offset_wx = vessel.x - wx
                        self._drag_offset_wy = vessel.y - wy
                elif hit[0] == "terrain":
                    hm = simulation.terrain_maps[hit[1]]
                    cx, cy = hm.center
                    self._drag_offset_wx = cx - wx
                    self._drag_offset_wy = cy - wy
                return True
            else:
                self.selected = None
                return False

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.is_dragging:
                self.is_dragging = False
                return True

        elif event.type == pygame.MOUSEMOTION:
            if self.is_dragging and self.selected:
                if not self._is_inside(event.pos[0], event.pos[1]):
                    return True

                lx, ly = self._global_to_local(event.pos[0], event.pos[1])
                wx, wy = self._screen_to_world(lx, ly)
                new_x = wx + self._drag_offset_wx
                new_y = wy + self._drag_offset_wy

                if self.selected[0] == "vessel":
                    vessel = simulation.world.get_vessel(self.selected[1])
                    if vessel:
                        vessel.x = new_x
                        vessel.y = new_y
                elif self.selected[0] == "terrain":
                    simulation.move_terrain(self.selected[1], new_x, new_y)
                return True

        # Right-click to delete selected object
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            if self.selected is not None:
                if self.selected[0] == "vessel":
                    simulation.remove_vessel(self.selected[1])
                elif self.selected[0] == "terrain":
                    simulation.remove_terrain(self.selected[1])
                self.selected = None
                self.is_dragging = False
                return True

        return False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_terrain_layer(self, terrain_maps, own_x: float, own_y: float,
                               zoom: float) -> pygame.Surface:
        surf = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))

        for hm in terrain_maps:
            min_x, min_y, max_x, max_y = hm.world_extent
            cell = hm.config.cell_size

            wy = min_y
            while wy <= max_y:
                wx = min_x
                while wx <= max_x:
                    elev = hm.get_elevation(wx, wy)
                    if elev > 0.5:
                        sx, sy = self._world_to_screen(wx, wy, own_x, own_y, zoom)
                        if 0 <= sx < self.size and 0 <= sy < self.size:
                            t = min(1.0, elev / 150.0)
                            r = int(COLORS['terrain_low'][0] + t * (COLORS['terrain_high'][0] - COLORS['terrain_low'][0]))
                            g = int(COLORS['terrain_low'][1] + t * (COLORS['terrain_high'][1] - COLORS['terrain_low'][1]))
                            b = int(COLORS['terrain_low'][2] + t * (COLORS['terrain_high'][2] - COLORS['terrain_low'][2]))
                            alpha = int(120 + 100 * t)
                            pixel_size = max(2, int(cell * zoom))
                            pygame.draw.rect(surf, (r, g, b, alpha),
                                             (sx - pixel_size // 2, sy - pixel_size // 2,
                                              pixel_size, pixel_size))
                    wx += cell
                wy += cell

        return surf

    def _draw_selection_ring(self, sx: int, sy: int, radius: int = 12) -> None:
        """Draw a pulsing selection ring around a selected object."""
        pygame.draw.circle(self.surface, COLORS['selected'], (sx, sy), radius, 2)
        pygame.draw.circle(self.surface, COLORS['selected'], (sx, sy), radius + 4, 1)

    def _draw_terrain_handle(self, hm, idx: int, own_x: float, own_y: float,
                              zoom: float) -> None:
        """Draw a draggable handle at the terrain center."""
        cx, cy = hm.center
        sx, sy = self._world_to_screen(cx, cy, own_x, own_y, zoom)

        # Diamond handle
        hs = 6
        pts = [(sx, sy - hs), (sx + hs, sy), (sx, sy + hs), (sx - hs, sy)]
        pygame.draw.polygon(self.surface, COLORS['terrain_high'], pts)
        pygame.draw.polygon(self.surface, (255, 255, 255), pts, 1)

        # Selection highlight
        if self.selected == ("terrain", idx):
            self._draw_selection_ring(sx, sy, 14)

    def render(self, own_ship, vessels, coastlines, range_nm: float,
               terrain_maps=None, occlusion_engine=None) -> pygame.Surface:
        self.surface.fill(COLORS['water'])

        max_range_m = range_nm * NM_TO_M
        zoom = self.radius / max_range_m if max_range_m > 0 else 1.0

        own_x = own_ship.x if own_ship else 0.0
        own_y = own_ship.y if own_ship else 0.0

        # Cache for coordinate conversion
        self._own_x = own_x
        self._own_y = own_y
        self._zoom = zoom
        self._max_range_m = max_range_m

        # Coastlines - draw as outlines only (polygons define water boundaries)
        # The terrain grid (green) shows actual land for occlusion
        if coastlines:
            for coastline in coastlines:
                if len(coastline.points) >= 3:
                    screen_pts = []
                    for p in coastline.points:
                        sx, sy = self._world_to_screen(p.x, p.y, own_x, own_y, zoom)
                        screen_pts.append((sx, sy))
                    if any(0 <= x < self.size and 0 <= y < self.size for x, y in screen_pts):
                        # Draw as outline, not filled - these are water boundaries
                        pygame.draw.polygon(self.surface, COLORS['land'], screen_pts, 1)

        # Terrain
        if terrain_maps:
            terrain_surf = self._render_terrain_layer(terrain_maps, own_x, own_y, zoom)
            self.surface.blit(terrain_surf, (0, 0))

        self._draw_range_rings(range_nm, max_range_m)

        # Vessels
        if vessels:
            vessel_list = vessels.values() if isinstance(vessels, dict) else vessels
            for vessel in vessel_list:
                if not vessel.is_active:
                    continue
                if vessel is own_ship:
                    continue
                vx, vy = self._world_to_screen(vessel.x, vessel.y, own_x, own_y, zoom)

                color = COLORS['vessel']
                is_occluded = False
                if occlusion_engine is not None:
                    if occlusion_engine.is_target_occluded(
                        own_x, own_y, vessel.x, vessel.y,
                        target_height_m=vessel.height
                    ):
                        color = COLORS['occluded']
                        is_occluded = True

                vessel_name = getattr(vessel, 'name', '')
                if is_occluded:
                    vessel_name = f"{vessel_name} [OCCLUDED]" if vessel_name else "[OCCLUDED]"
                self._draw_vessel_marker(vx, vy, vessel.course, color, vessel_name)

                # Selection highlight for vessels
                if self.selected == ("vessel", vessel.id):
                    self._draw_selection_ring(vx, vy, 14)

        # Terrain handles
        if terrain_maps:
            for i, hm in enumerate(terrain_maps):
                self._draw_terrain_handle(hm, i, own_x, own_y, zoom)

        # Own ship
        if own_ship:
            ox, oy = self.center
            self._draw_vessel_marker(ox, oy, own_ship.course, COLORS['own_ship'], 'OWN')

        self._draw_north_arrow()

        # Placement mode indicator
        if self.placement_mode:
            font = self._get_font()
            mode_label = font.render(f"PLACE: {self.placement_mode.upper()}", True,
                                     COLORS['placement'])
            self.surface.blit(mode_label,
                              (self.size // 2 - mode_label.get_width() // 2,
                               self.size - 22))
        else:
            font = self._get_font()
            label = font.render("SCENE VIEW", True, COLORS['title'])
            self.surface.blit(label, (self.size // 2 - label.get_width() // 2, 5))

        # Drag hint
        if self.selected and not self.is_dragging:
            font = self._get_font(12)
            hint = font.render("Drag to move | Right-click to delete", True,
                               (120, 140, 160))
            self.surface.blit(hint,
                              (self.size // 2 - hint.get_width() // 2,
                               self.size - 18))

        return self.surface

    def render_csv(self, sweep_pairs: List[Tuple[float, List[float]]],
                    range_nm: float = 6.0) -> pygame.Surface:
        max_range_m = range_nm * NM_TO_M
        zoom = self.radius / max_range_m if max_range_m > 0 else 1.0

        for bearing, data in sweep_pairs:
            bearing_rad = math.radians(bearing)
            num_bins = len(data)
            for i, intensity in enumerate(data):
                if intensity < 0.05:
                    continue
                dist_m = (i + 0.5) / num_bins * max_range_m
                dx = dist_m * math.sin(bearing_rad)
                dy = dist_m * math.cos(bearing_rad)
                sx = self.center[0] + dx * zoom
                sy = self.center[1] - dy * zoom
                if 0 <= sx < self.size and 0 <= sy < self.size:
                    green = int(min(255, intensity * 255 * 1.5))
                    alpha = int(min(255, intensity * 255))
                    pygame.draw.circle(self.echo_surface, (0, green, 0, alpha),
                                       (int(sx), int(sy)), 2)

        self.surface.fill(COLORS['water'])
        self.surface.blit(self.echo_surface, (0, 0))
        self._draw_range_rings(range_nm, max_range_m)
        self._draw_north_arrow()

        pygame.draw.circle(self.surface, COLORS['own_ship'], self.center, 4)

        font = self._get_font()
        label = font.render("SCENE VIEW (CSV)", True, COLORS['title'])
        self.surface.blit(label, (self.size // 2 - label.get_width() // 2, 5))

        return self.surface

    def clear_echoes(self):
        self.echo_surface.fill((0, 0, 0, 0))

    def _draw_range_rings(self, range_nm: float, max_range_m: float):
        num_rings = 4
        font = self._get_font(12)
        for i in range(1, num_rings + 1):
            r = int(self.radius * i / num_rings)
            pygame.draw.circle(self.surface, COLORS['range_ring'], self.center, r, 1)
            ring_nm = range_nm * i / num_rings
            label = font.render(f"{ring_nm:.1f}nm", True, COLORS['range_ring'])
            self.surface.blit(label, (self.center[0] + 4, self.center[1] - r + 2))

    def _draw_vessel_marker(self, x: int, y: int, heading: float,
                             color, name: str = ''):
        heading_rad = math.radians(heading)
        size = 8
        nose_x = x + size * math.sin(heading_rad)
        nose_y = y - size * math.cos(heading_rad)
        left_x = x + size * 0.6 * math.sin(heading_rad + 2.5)
        left_y = y - size * 0.6 * math.cos(heading_rad + 2.5)
        right_x = x + size * 0.6 * math.sin(heading_rad - 2.5)
        right_y = y - size * 0.6 * math.cos(heading_rad - 2.5)

        pts = [(int(nose_x), int(nose_y)),
               (int(left_x), int(left_y)),
               (int(right_x), int(right_y))]
        pygame.draw.polygon(self.surface, color, pts)

        if name:
            font = self._get_font(12)
            label = font.render(name, True, COLORS['vessel_label'])
            self.surface.blit(label, (x + 10, y - 6))

    def _draw_north_arrow(self):
        ax, ay = self.size - 30, 30
        pygame.draw.line(self.surface, COLORS['north_arrow'],
                         (ax, ay + 15), (ax, ay - 15), 2)
        pygame.draw.polygon(self.surface, COLORS['north_arrow'],
                            [(ax, ay - 15), (ax - 5, ay - 5), (ax + 5, ay - 5)])
        font = self._get_font(12)
        n_label = font.render("N", True, COLORS['north_arrow'])
        self.surface.blit(n_label, (ax - n_label.get_width() // 2, ay - 30))
