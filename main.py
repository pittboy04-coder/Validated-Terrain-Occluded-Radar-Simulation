"""Demo: terrain occlusion in a radar simulation.

Creates an island with elevation between the radar and a target vessel,
runs a sweep, and verifies that the vessel behind the island is occluded.
"""
from radar_sim.core.simulation import Simulation
from radar_sim.objects.vessel import Vessel, VesselType
from radar_sim.environment.terrain import create_island_terrain
from radar_sim.environment.weather import WeatherConditions
from radar_sim.core.range_bearing import calculate_bearing


def main():
    sim = Simulation()

    # --- Own ship at origin ------------------------------------------------
    own_ship = Vessel(
        id="own_ship", name="Own Ship",
        vessel_type=VesselType.OWN_SHIP,
        x=0, y=0, course=0, speed=0,
        length=100, beam=15, height=20,
    )
    sim.world.add_vessel(own_ship)

    # --- Island terrain at 3 km north, 120 m peak -------------------------
    island = create_island_terrain(
        center_x=0, center_y=3000,
        radius=800, peak_height=120,
        grid_size=64, cell_size=50,
    )
    sim.add_terrain(island)

    # --- Target vessel directly behind the island (5 km north) -------------
    hidden_vessel = Vessel(
        id="hidden", name="Hidden Cargo",
        vessel_type=VesselType.CARGO,
        x=0, y=5000, course=180, speed=0,
        length=150, beam=20, height=25,
    )
    sim.world.add_vessel(hidden_vessel)

    # --- Visible vessel off to the side (5 km east) ------------------------
    visible_vessel = Vessel(
        id="visible", name="Visible Tanker",
        vessel_type=VesselType.TANKER,
        x=5000, y=0, course=270, speed=0,
        length=200, beam=30, height=20,
    )
    sim.world.add_vessel(visible_vessel)

    # --- Calm weather (minimal clutter) ------------------------------------
    sim.weather.set_conditions(WeatherConditions(
        sea_state=1, wind_speed_knots=5, wind_direction=0,
        rain_rate_mmh=0, visibility_nm=20,
    ))

    # --- Run one simulation update so sweep buffer is populated ------------
    sim.radar.params.set_range_scale(6)  # 6 NM
    sim.update(dt=0.5)

    # --- Check occlusion ---------------------------------------------------
    bearing_to_hidden = calculate_bearing(0, 0, hidden_vessel.x, hidden_vessel.y)
    bearing_to_visible = calculate_bearing(0, 0, visible_vessel.x, visible_vessel.y)

    hidden_occluded = sim.occlusion_engine.is_target_occluded(
        0, 0, hidden_vessel.x, hidden_vessel.y,
        target_height_m=hidden_vessel.height,
    )
    visible_occluded = sim.occlusion_engine.is_target_occluded(
        0, 0, visible_vessel.x, visible_vessel.y,
        target_height_m=visible_vessel.height,
    )

    print("=== Terrain-Occluded Radar Simulation Demo ===\n")
    print(f"Island terrain: center=(0, 3000), radius=800m, peak=120m")
    print(f"Antenna height: {sim.radar.params.antenna_height_m} m\n")

    print(f"Hidden Cargo  @ bearing {bearing_to_hidden:5.1f} deg, range 5000m")
    print(f"  Occluded: {hidden_occluded}")

    print(f"Visible Tanker @ bearing {bearing_to_visible:5.1f} deg, range 5000m")
    print(f"  Occluded: {visible_occluded}")

    # --- Sweep data --------------------------------------------------------
    sweep_hidden = sim.get_radar_sweep_data(bearing_to_hidden)
    sweep_visible = sim.get_radar_sweep_data(bearing_to_visible)

    # Find peak terrain return in the island region (bins ~55-70 for 3km at 6NM)
    num_bins = len(sweep_hidden)
    max_range_m = sim.radar.params.max_range_m
    bin_size = max_range_m / num_bins

    island_start_bin = int(2200 / bin_size)
    island_end_bin = min(num_bins, int(3800 / bin_size))
    terrain_peak = max(sweep_hidden[island_start_bin:island_end_bin])

    target_bin = int(5000 / bin_size)
    target_window = sweep_hidden[max(0, target_bin - 3):min(num_bins, target_bin + 4)]
    hidden_signal = max(target_window) if target_window else 0.0

    visible_bin = int(5000 / bin_size)
    visible_window = sweep_visible[max(0, visible_bin - 3):min(num_bins, visible_bin + 4)]
    visible_signal = max(visible_window) if visible_window else 0.0

    print(f"\nSweep data (bearing {bearing_to_hidden:.0f} deg):")
    print(f"  Peak terrain return (island): {terrain_peak:.4f}")
    print(f"  Signal at hidden target range: {hidden_signal:.4f}")

    print(f"\nSweep data (bearing {bearing_to_visible:.0f} deg):")
    print(f"  Signal at visible target range: {visible_signal:.4f}")

    # --- Assertions --------------------------------------------------------
    assert hidden_occluded, "Hidden vessel should be occluded by island"
    assert not visible_occluded, "Visible vessel should NOT be occluded"
    assert terrain_peak > 0.01, "Island should produce terrain returns"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
