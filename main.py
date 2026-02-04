#!/usr/bin/env python3
"""Terrain-Occluded Radar PPI Simulator - Interactive GUI."""
import sys
import os
import platform
import pygame
from radar_sim.core.simulation import Simulation
from radar_sim.visualization.ppi_display import PPIDisplay
from radar_sim.visualization.scene_view import SceneView
from radar_sim.ui.control_panel import ControlPanel
from radar_sim.scenarios.scenario_manager import ScenarioManager
from radar_sim.scenarios.presets import PRESET_SCENARIOS
from radar_sim.detection import TargetDetector, TargetTracker

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
FPS = 60


def main():
    pygame.init()
    pygame.display.set_caption("Terrain-Occluded Radar PPI Simulator")

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    # Initialize simulation
    sim = Simulation()

    # Set export directory
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "maritime_radar_sim", "Advanced Simulator CSV Outputs")
    export_dir = os.path.normpath(export_dir)
    os.makedirs(export_dir, exist_ok=True)
    sim.exporter.output_dir = export_dir
    sim.exporter.mirror_dir = os.path.expanduser(
        "~/projects/radar-research/Advanced Simulator CSV Outputs")

    # Initialize scenario manager with presets
    scenario_manager = ScenarioManager()
    scenario_manager.register_scenarios(PRESET_SCENARIOS)

    # Load the first scenario (Island Occlusion demo)
    scenario_manager.load_scenario(
        PRESET_SCENARIOS[0].name,
        sim.world, sim.radar, sim.weather,
        simulation=sim)

    # Layout
    CONTROL_PANEL_WIDTH = 340
    MIN_PPI_SIZE = 300

    def calc_layout(win_w, win_h):
        available = win_w - CONTROL_PANEL_WIDTH - 30
        display_size = min(win_h - 40, (available - 20) // 2)
        display_size = max(MIN_PPI_SIZE, display_size)
        px = 10
        py = (win_h - display_size) // 2
        sx = display_size + 20
        sy = py
        cp_x = display_size * 2 + 30
        cp_w = win_w - cp_x - 10
        return display_size, px, py, sx, sy, cp_x, cp_w, win_h - 40

    display_size, ppi_x, ppi_y, scene_x, scene_y, cp_x, cp_w, cp_h = calc_layout(
        WINDOW_WIDTH, WINDOW_HEIGHT)

    ppi = PPIDisplay(size=display_size)
    ppi.initialize()
    ppi.set_ppi_offset(ppi_x, ppi_y)

    scene_view = SceneView(size=display_size)
    scene_view.set_offset(scene_x, scene_y)

    control_panel = ControlPanel(x=cp_x, y=20, width=cp_w, height=cp_h)
    control_panel.set_simulation(sim)
    control_panel.set_scenario_manager(scenario_manager)
    control_panel.set_scene_view(scene_view)

    # Track CSV player state to reset tracker when mode changes
    csv_was_active = False

    range_scales = sim.radar.params.range_scales_nm
    current_range_idx = (range_scales.index(sim.radar.params.current_range_nm)
                         if sim.radar.params.current_range_nm in range_scales else 5)

    # Target detection and tracking for CSV playback
    detector = TargetDetector(min_intensity=0.15, min_width=2, max_width=50)
    tracker = TargetTracker(bearing_gate_deg=3.0, range_gate_ratio=0.05,
                            max_misses=5, min_hits_for_label=3)

    running = True
    last_bearing = 0.0
    current_bearing = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if scene_view.placement_mode:
                        scene_view.placement_mode = None
                    elif scene_view.selected:
                        scene_view.selected = None
                    else:
                        running = False
                elif event.key == pygame.K_SPACE:
                    sim.toggle_pause()
                elif event.key == pygame.K_r:
                    sim.reset()
                    sim.setup_default_scenario()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    sim.set_time_scale(sim.time_scale * 1.5)
                elif event.key == pygame.K_MINUS:
                    sim.set_time_scale(sim.time_scale / 1.5)
                elif event.key == pygame.K_F5:
                    if sim.is_recording():
                        count = sim.get_record_count()
                        filepath = sim.stop_recording()
                        if filepath:
                            print(f"Recording saved ({count} sweeps): {filepath}")
                        else:
                            print(f"Recording stopped but no sweeps captured (count={count})")
                    else:
                        session_id = sim.start_recording()
                        print(f"Recording started: {session_id}")
                elif event.key == pygame.K_c:
                    if sim.coastline_enabled:
                        sim.clear_coastlines()
                    else:
                        sim.setup_harbor_coastline()
                elif event.key == pygame.K_e:
                    filepath = sim.export_current_sweep(current_bearing)
                    print(f"Sweep exported to: {filepath}")
                elif event.key == pygame.K_t:
                    # Quick-add island terrain ahead of own ship
                    from radar_sim.environment.terrain import create_island_terrain
                    own = sim.world.own_ship
                    cx = own.x if own else 0
                    cy = (own.y if own else 0) + 3000
                    hm = create_island_terrain(center_x=cx, center_y=cy,
                                               radius=800, peak_height=120)
                    sim.add_terrain(hm)
                    print("Added island terrain")
                elif event.key == pygame.K_DELETE:
                    sim.clear_terrain()
                    print("Cleared all terrain")

            elif event.type == pygame.VIDEORESIZE:
                win_w, win_h = event.w, event.h
                if win_w < 100 or win_h < 100:
                    continue
                screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
                display_size, ppi_x, ppi_y, scene_x, scene_y, cp_x, cp_w, cp_h = calc_layout(win_w, win_h)
                ppi = PPIDisplay(size=display_size)
                ppi.initialize()
                ppi.set_ppi_offset(ppi_x, ppi_y)
                scene_view = SceneView(size=display_size)
                scene_view.set_offset(scene_x, scene_y)
                control_panel = ControlPanel(x=cp_x, y=20, width=cp_w, height=cp_h)
                control_panel.set_simulation(sim)
                control_panel.set_scenario_manager(scenario_manager)
                control_panel.set_scene_view(scene_view)

            elif event.type == pygame.MOUSEMOTION:
                ppi.handle_mouse_motion(event.pos[0], event.pos[1])

            elif event.type == pygame.MOUSEWHEEL:
                mouse_pos = pygame.mouse.get_pos()
                if ppi.screen_to_polar(mouse_pos[0], mouse_pos[1]) is not None:
                    if event.y > 0 and current_range_idx > 0:
                        current_range_idx -= 1
                        sim.radar.set_range_scale(range_scales[current_range_idx])
                    elif event.y < 0 and current_range_idx < len(range_scales) - 1:
                        current_range_idx += 1
                        sim.radar.set_range_scale(range_scales[current_range_idx])

            if not scene_view.handle_event(event, sim):
                control_panel.handle_event(event)

        # Skip rendering when minimized
        if pygame.display.get_surface().get_size()[0] == 0:
            clock.tick(FPS)
            continue

        # CSV playback mode or normal simulation
        sweep_pairs = None
        tracked_targets = None
        csv_is_active = control_panel.csv_player and control_panel.csv_player.active

        # Reset tracker when CSV mode changes
        if csv_is_active and not csv_was_active:
            tracker.reset()
        csv_was_active = csv_is_active

        if csv_is_active:
            sweep_pairs = control_panel.csv_player.get_next_sweeps()
            for bearing, data in sweep_pairs:
                ppi.draw_sweep_data(bearing, data)
                current_bearing = bearing

            # Ensure PPI range is set for CSV playback (use current radar range)
            ppi.set_range(sim.radar.params.current_range_nm)

            # Detect and track targets in CSV data
            detections = detector.detect_multiple_sweeps(sweep_pairs)
            tracker.update(detections, current_bearing)
            tracked_targets = tracker.get_stable_tracks()
        else:
            sim.update()
            current_bearing = sim.radar.get_current_bearing()

            # Use 0.5° steps for 720 spokes per rotation (360/0.5 = 720)
            angular_step = 0.5
            if int(current_bearing / angular_step) != int(last_bearing / angular_step):
                sweep_data = sim.get_radar_sweep_data(current_bearing)
                ppi.draw_sweep_data(current_bearing, sweep_data)

                # Record if active
                if sim.is_recording():
                    sim.exporter.add_sweep(
                        timestamp=sim.world.time,
                        bearing_deg=current_bearing,
                        range_scale_nm=sim.radar.params.current_range_nm,
                        gain=sim.radar.params.gain,
                        sea_clutter=sim.radar.params.sea_clutter,
                        rain_clutter=sim.radar.params.rain_clutter,
                        echo_values=sweep_data)

                if sim.world.own_ship:
                    ppi.set_heading(sim.world.own_ship.course)

            ppi.set_range(sim.radar.params.current_range_nm)
            last_bearing = current_bearing

        # Render
        screen.fill((20, 20, 30))

        # PPI (with target labels if in CSV mode)
        ppi_surface = ppi.render(tracked_targets=tracked_targets)
        screen.blit(ppi_surface, (ppi_x, ppi_y))

        # Scene view (with terrain + occlusion visualization)
        if sweep_pairs is not None:
            scene_surface = scene_view.render_csv(
                sweep_pairs, sim.radar.params.current_range_nm)
        else:
            scene_surface = scene_view.render(
                sim.world.own_ship, sim.world.get_all_vessels(),
                sim.coastlines, sim.radar.params.current_range_nm,
                terrain_maps=sim.terrain_maps,
                occlusion_engine=sim.occlusion_engine)
        screen.blit(scene_surface, (scene_x, scene_y))

        # Cursor info
        ppi.draw_cursor_info(screen, ppi_x, ppi_y + display_size + 5)

        # Track info (when in CSV playback mode)
        if tracked_targets is not None:
            ppi.draw_track_info(screen, ppi_x + 150, ppi_y + display_size + 5, tracker)

        # Control panel
        control_panel.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
