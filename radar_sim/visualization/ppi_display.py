"""PPI (Plan Position Indicator) radar display with beam broadening."""
import math
from typing import List, Tuple, Optional
import pygame


class PPIDisplay:
    """Renders radar data in classic PPI format."""

    def __init__(self, size: int = 600, center: Tuple[int, int] = None):
        self.size = size
        self.radius = size // 2 - 20
        self.center = center or (size // 2, size // 2)

        # Colors (Furuno-style green phosphor look)
        self.bg_color = (0, 10, 0)
        self.grid_color = (0, 60, 0)
        self.sweep_color = (0, 255, 0)
        self.echo_color = (0, 255, 0)
        self.text_color = (0, 200, 0)
        self.heading_marker_color = (0, 255, 0)

        # Display settings
        self.num_range_rings = 4
        self.show_range_rings = True
        self.show_bearing_lines = True
        self.show_heading_line = True
        self.trail_persistence = 0.95

        # Beam broadening: draw each sweep across ±beam_half_width degrees
        # For 720 spokes (0.5° steps), use slightly wider beam to ensure overlap
        self.beam_half_width_deg = 0.4  # ±0.4° = 0.8° total beamwidth
        self.beam_lines = 5  # Number of adjacent radial lines per sweep

        # Surfaces
        self.surface: Optional[pygame.Surface] = None
        self.echo_surface: Optional[pygame.Surface] = None

        # State
        self.current_bearing = 0.0
        self.range_nm = 6.0
        self.heading = 0.0

        # Mouse interaction state
        self.cursor_range_nm: Optional[float] = None
        self.cursor_bearing: Optional[float] = None
        self.ppi_offset = (0, 0)
        self.font: Optional[pygame.font.Font] = None

    def initialize(self) -> pygame.Surface:
        self.surface = pygame.Surface((self.size, self.size))
        self.echo_surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        self.echo_surface.fill((0, 0, 0, 0))
        return self.surface

    def _polar_to_screen(self, range_ratio: float, bearing_deg: float) -> Tuple[int, int]:
        bearing_rad = math.radians(bearing_deg - 90)
        r = range_ratio * self.radius
        x = self.center[0] + r * math.cos(bearing_rad)
        y = self.center[1] + r * math.sin(bearing_rad)
        return int(x), int(y)

    def draw_background(self) -> None:
        self.surface.fill(self.bg_color)

        if self.show_range_rings:
            for i in range(1, self.num_range_rings + 1):
                r = int(self.radius * i / self.num_range_rings)
                pygame.draw.circle(self.surface, self.grid_color,
                                   self.center, r, 1)

        if self.show_bearing_lines:
            for bearing in range(0, 360, 30):
                end_x, end_y = self._polar_to_screen(1.0, bearing)
                pygame.draw.line(self.surface, self.grid_color,
                                 self.center, (end_x, end_y), 1)

        if self.show_heading_line:
            hx, hy = self._polar_to_screen(1.0, self.heading)
            pygame.draw.line(self.surface, self.heading_marker_color,
                             self.center, (hx, hy), 2)

        pygame.draw.circle(self.surface, self.sweep_color, self.center, 3)

    def draw_sweep_line(self, bearing: float) -> None:
        self.current_bearing = bearing
        end_x, end_y = self._polar_to_screen(1.0, bearing)
        pygame.draw.line(self.surface, self.sweep_color,
                         self.center, (end_x, end_y), 1)

    def draw_sweep_data(self, bearing: float, data: List[float],
                        fade_old: bool = True) -> None:
        if fade_old:
            fade_surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            fade_surface.fill((0, 0, 0, int(255 * (1 - self.trail_persistence))))
            self.echo_surface.blit(fade_surface, (0, 0),
                                   special_flags=pygame.BLEND_RGBA_SUB)

        num_bins = len(data)

        # Beam broadening: draw at bearing offsets to simulate real beamwidth
        offsets = [0.0]
        if self.beam_lines >= 3:
            step = self.beam_half_width_deg / ((self.beam_lines - 1) / 2.0)
            for k in range(1, (self.beam_lines + 1) // 2):
                offsets.append(k * step)
                offsets.append(-k * step)

        for offset in offsets:
            b = bearing + offset
            bearing_rad = math.radians(b - 90)
            # Taper intensity for off-center lines (cosine taper)
            if self.beam_half_width_deg > 0:
                taper = max(0.0, math.cos(
                    (abs(offset) / self.beam_half_width_deg) * (math.pi / 2.0)))
            else:
                taper = 1.0

            for i, intensity in enumerate(data):
                if intensity < 0.02:
                    continue

                range_ratio = (i + 0.5) / num_bins
                r = range_ratio * self.radius

                x = self.center[0] + r * math.cos(bearing_rad)
                y = self.center[1] + r * math.sin(bearing_rad)

                tapered = intensity * taper
                green = int(min(255, tapered * 255 * 1.5))
                alpha = int(min(255, tapered * 255))

                pygame.draw.circle(self.echo_surface, (0, green, 0, alpha),
                                   (int(x), int(y)), 2)

    def draw_detection(self, range_ratio: float, bearing: float,
                       intensity: float = 1.0) -> None:
        x, y = self._polar_to_screen(range_ratio, bearing)
        size = max(3, int(intensity * 6))
        green = int(min(255, intensity * 255))
        pygame.draw.circle(self.echo_surface, (0, green, 0, 255),
                           (x, y), size)

    def set_range(self, range_nm: float) -> None:
        self.range_nm = range_nm

    def set_heading(self, heading: float) -> None:
        self.heading = heading

    def clear_echoes(self) -> None:
        if self.echo_surface:
            self.echo_surface.fill((0, 0, 0, 0))

    def set_ppi_offset(self, x: int, y: int) -> None:
        self.ppi_offset = (x, y)

    def screen_to_polar(self, screen_x: int, screen_y: int) -> Optional[Tuple[float, float]]:
        local_x = screen_x - self.ppi_offset[0] - self.center[0]
        local_y = screen_y - self.ppi_offset[1] - self.center[1]

        distance = math.sqrt(local_x * local_x + local_y * local_y)

        if distance > self.radius:
            return None

        range_ratio = distance / self.radius
        range_nm = range_ratio * self.range_nm

        bearing = math.degrees(math.atan2(local_x, -local_y))
        bearing = (bearing + 360) % 360

        return range_nm, bearing

    def handle_mouse_motion(self, screen_x: int, screen_y: int) -> bool:
        result = self.screen_to_polar(screen_x, screen_y)
        if result:
            self.cursor_range_nm, self.cursor_bearing = result
            return True
        else:
            self.cursor_range_nm = None
            self.cursor_bearing = None
            return False

    def draw_cursor_info(self, surface: pygame.Surface, x: int, y: int) -> None:
        if self.font is None:
            self.font = pygame.font.Font(None, 22)

        if self.cursor_range_nm is not None and self.cursor_bearing is not None:
            info_lines = [
                f"Cursor:",
                f"  Range: {self.cursor_range_nm:.2f} nm",
                f"  Bearing: {self.cursor_bearing:.1f}\u00b0",
            ]

            box_width = 140
            box_height = len(info_lines) * 18 + 10
            box_rect = pygame.Rect(x, y, box_width, box_height)

            pygame.draw.rect(surface, (20, 30, 20), box_rect)
            pygame.draw.rect(surface, (0, 100, 0), box_rect, 1)

            for i, line in enumerate(info_lines):
                text = self.font.render(line, True, self.text_color)
                surface.blit(text, (x + 5, y + 5 + i * 18))

    def draw_cursor_crosshairs(self) -> None:
        if self.cursor_range_nm is None or self.cursor_bearing is None:
            return

        range_ratio = self.cursor_range_nm / self.range_nm
        if range_ratio > 1.0:
            return

        x, y = self._polar_to_screen(range_ratio, self.cursor_bearing)

        crosshair_size = 8
        crosshair_color = (0, 255, 255)

        pygame.draw.line(self.surface, crosshair_color,
                         (x - crosshair_size, y), (x + crosshair_size, y), 1)
        pygame.draw.line(self.surface, crosshair_color,
                         (x, y - crosshair_size), (x, y + crosshair_size), 1)

    def render(self) -> pygame.Surface:
        self.draw_background()
        self.surface.blit(self.echo_surface, (0, 0))
        self.draw_cursor_crosshairs()
        self.draw_sweep_line(self.current_bearing)
        self._draw_range_labels()
        return self.surface

    def _draw_range_labels(self) -> None:
        if self.font is None:
            self.font = pygame.font.Font(None, 18)

        for i in range(1, self.num_range_rings + 1):
            range_val = self.range_nm * i / self.num_range_rings
            label = f"{range_val:.1f}"

            r = int(self.radius * i / self.num_range_rings)
            label_x = self.center[0] + 5
            label_y = self.center[1] + r - 10

            text = self.font.render(label, True, self.grid_color)
            self.surface.blit(text, (label_x, label_y))
