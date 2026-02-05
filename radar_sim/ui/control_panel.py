"""Main control panel for the terrain-occluded radar simulator."""
import os
import pygame
from typing import Optional, List
from .widgets import Panel, Button, Slider, Label, DropDown, TextInput, COLORS
from ..core.simulation import Simulation
from ..scenarios.scenario_manager import ScenarioManager
from ..data_import import CsvPlayer


class ControlPanel:
    """Main control panel containing all radar and terrain controls."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.simulation: Optional[Simulation] = None
        self.scenario_manager: Optional[ScenarioManager] = None
        self.csv_player: Optional[CsvPlayer] = None

        # Scroll state
        self.scroll_offset = 0
        self.max_scroll = 0
        self.is_scrollbar_dragging = False
        self.scroll_drag_start_y = 0
        self.scroll_drag_start_offset = 0

        # Validation state
        self._validation_report = None
        self._baseline_editor = None

        self._create_panels()

    def _create_panels(self) -> None:
        panel_width = self.rect.width - 20
        y_offset = 10

        # Scenario panel
        self.scenario_panel = Panel(10, y_offset, panel_width, 135, "SCENARIO")
        self._scenario_names = ["Default"]
        self.scenario_slider = Slider(
            15, 45, panel_width - 70, 20,
            min_val=0.0, max_val=0.0, value=0.0,
            label="SCENARIO",
            callback=self._on_scenario_slider)
        self.scenario_name_label = Label(20, 75, "Default", 18)
        self.load_location_button = Button(
            15, 100, panel_width - 30, 28, "LOAD LOCATION",
            callback=self._on_load_location)
        self.scenario_panel.add_widget(self.scenario_slider)
        self.scenario_panel.add_widget(self.scenario_name_label)
        self.scenario_panel.add_widget(self.load_location_button)
        y_offset += 145

        # Range control panel
        self.range_panel = Panel(10, y_offset, panel_width, 110, "RANGE")
        self.range_input = TextInput(
            15, 35, panel_width - 30, 25,
            text="6.0", label="Range (nm)",
            callback=self._on_range_input)
        self.range_dropdown = DropDown(
            15, 75, panel_width - 30, 25,
            ["0.25 nm", "0.5 nm", "0.75 nm", "1.5 nm", "3 nm", "6 nm",
             "12 nm", "24 nm", "48 nm"],
            selected=5,
            callback=self._on_range_change)
        self.range_panel.add_widget(self.range_input)
        self.range_panel.add_widget(self.range_dropdown)
        y_offset += 120

        # Gain controls panel
        self.gain_panel = Panel(10, y_offset, panel_width, 160, "GAIN CONTROLS")
        self.gain_slider = Slider(
            15, 45, panel_width - 70, 20,
            min_val=0.0, max_val=1.0, value=0.5,
            label="GAIN", callback=self._on_gain_change)
        self.sea_slider = Slider(
            15, 85, panel_width - 70, 20,
            min_val=0.0, max_val=1.0, value=0.3,
            label="SEA", callback=self._on_sea_change)
        self.rain_slider = Slider(
            15, 125, panel_width - 70, 20,
            min_val=0.0, max_val=1.0, value=0.3,
            label="RAIN", callback=self._on_rain_change)
        self.gain_panel.add_widget(self.gain_slider)
        self.gain_panel.add_widget(self.sea_slider)
        self.gain_panel.add_widget(self.rain_slider)
        y_offset += 170

        # Simulation controls panel
        self.sim_panel = Panel(10, y_offset, panel_width, 120, "SIMULATION")
        self.pause_button = Button(
            15, 35, (panel_width - 40) // 2, 30, "PAUSE",
            callback=self._on_pause)
        self.reset_button = Button(
            (panel_width - 40) // 2 + 25, 35, (panel_width - 40) // 2, 30, "RESET",
            callback=self._on_reset)
        self.speed_slider = Slider(
            15, 85, panel_width - 70, 20,
            min_val=0.1, max_val=5.0, value=1.0,
            label="SPEED", callback=self._on_speed_change)
        self.sim_panel.add_widget(self.pause_button)
        self.sim_panel.add_widget(self.reset_button)
        self.sim_panel.add_widget(self.speed_slider)
        y_offset += 130

        # Terrain panel
        self.terrain_panel = Panel(10, y_offset, panel_width, 140, "TERRAIN")
        btn_w3 = (panel_width - 50) // 3
        self.add_island_button = Button(
            15, 35, btn_w3, 28, "ISLAND",
            callback=self._on_add_island)
        self.add_ridge_button = Button(
            20 + btn_w3, 35, btn_w3, 28, "RIDGE",
            callback=self._on_add_ridge)
        self.clear_terrain_button = Button(
            25 + btn_w3 * 2, 35, btn_w3, 28, "CLEAR",
            callback=self._on_clear_terrain)
        self.terrain_label = Label(20, 70, "No terrain", 18)
        self.coastline_button = Button(
            15, 95, panel_width - 30, 28, "COASTLINE ON",
            callback=self._on_coastline_toggle)
        self.terrain_panel.add_widget(self.add_island_button)
        self.terrain_panel.add_widget(self.add_ridge_button)
        self.terrain_panel.add_widget(self.clear_terrain_button)
        self.terrain_panel.add_widget(self.terrain_label)
        self.terrain_panel.add_widget(self.coastline_button)
        y_offset += 150

        # Place Objects panel
        self.place_panel = Panel(10, y_offset, panel_width, 200, "PLACE OBJECTS")
        obj_btn_w = (panel_width - 50) // 3
        obj_row_h = 32
        obj_types = [
            ("CARGO", "cargo"), ("TANKER", "tanker"), ("FISHING", "fishing"),
            ("SAILING", "sailing"), ("TUG", "tug"), ("FERRY", "passenger"),
            ("PILOT", "pilot"), ("BUOY", "buoy"),
        ]
        self._place_buttons = []
        for i, (label, vtype) in enumerate(obj_types):
            row = i // 3
            col = i % 3
            bx = 15 + col * (obj_btn_w + 5)
            by = 35 + row * obj_row_h
            btn = Button(bx, by, obj_btn_w, 26, label,
                         callback=lambda vt=vtype: self._on_place_object(vt))
            self.place_panel.add_widget(btn)
            self._place_buttons.append(btn)
        self.cancel_place_button = Button(
            15, 35 + 3 * obj_row_h, panel_width - 30, 26, "CANCEL PLACEMENT",
            callback=self._on_cancel_placement)
        self.place_panel.add_widget(self.cancel_place_button)
        self.place_label = Label(20, 35 + 3 * obj_row_h + 30, "Click scene to place", 16)
        self.place_panel.add_widget(self.place_label)
        y_offset += 210

        # Weather panel
        self._sea_state_names = ["Calm (0)", "Light (2)", "Moderate (4)", "Rough (6)", "Severe (8)"]
        self._sea_state_values = [0, 2, 4, 6, 8]
        self.weather_panel = Panel(10, y_offset, panel_width, 140, "WEATHER")
        self.sea_state_slider = Slider(
            15, 45, panel_width - 70, 20,
            min_val=0.0, max_val=4.0, value=1.0,
            label="SEA STATE", callback=self._on_sea_state_slider)
        self.sea_state_label = Label(20, 70, "Light (2)", 18)
        self.rain_rate_slider = Slider(
            15, 105, panel_width - 70, 20,
            min_val=0.0, max_val=50.0, value=0.0,
            label="RAIN mm/h", callback=self._on_rain_rate_change)
        self.weather_panel.add_widget(self.sea_state_slider)
        self.weather_panel.add_widget(self.sea_state_label)
        self.weather_panel.add_widget(self.rain_rate_slider)
        y_offset += 150

        # Data Export panel
        self.export_panel = Panel(10, y_offset, panel_width, 210, "DATA EXPORT")
        btn_width = (panel_width - 50) // 3
        self.record_button = Button(
            15, 35, btn_width, 28, "RECORD",
            callback=self._on_record)
        self.save_button = Button(
            20 + btn_width, 35, btn_width, 28, "SAVE",
            callback=self._on_save)
        self.csv_button = Button(
            25 + btn_width * 2, 35, btn_width, 28, "LOAD CSV",
            callback=self._on_load_csv)
        self.record_label = Label(20, 70, "Not recording", 18)
        self.file_label = Label(20, 90, "", 16)
        # Capture analysis button
        self.load_capture_button = Button(
            15, 115, panel_width - 30, 28, "LOAD CAPTURE",
            callback=self._on_load_capture)
        self.capture_label = Label(20, 150, "", 16)
        self.export_panel.add_widget(self.record_button)
        self.export_panel.add_widget(self.save_button)
        self.export_panel.add_widget(self.csv_button)
        self.export_panel.add_widget(self.record_label)
        self.export_panel.add_widget(self.file_label)
        self.export_panel.add_widget(self.load_capture_button)
        self.export_panel.add_widget(self.capture_label)
        y_offset += 220

        # --- Validation panel ---
        self.validation_panel = Panel(10, y_offset, panel_width, 200, "VALIDATION")
        val_btn_w = (panel_width - 40) // 2
        self.load_ref_button = Button(
            15, 35, val_btn_w, 28, "LOAD REF",
            callback=self._on_load_reference)
        self.compare_button = Button(
            20 + val_btn_w, 35, val_btn_w, 28, "COMPARE",
            callback=self._on_compare)
        self.val_overall_label = Label(20, 70, "Overall: --", 18)
        self.val_blob_label = Label(20, 90, "Blob: --", 16)
        self.val_intensity_label = Label(20, 108, "Intensity: --", 16)
        self.val_clutter_label = Label(20, 126, "Clutter: --", 16)
        self.val_noise_label = Label(20, 144, "Noise: --", 16)
        self.val_sparsity_label = Label(20, 162, "Sparsity: --", 16)
        self.validation_panel.add_widget(self.load_ref_button)
        self.validation_panel.add_widget(self.compare_button)
        self.validation_panel.add_widget(self.val_overall_label)
        self.validation_panel.add_widget(self.val_blob_label)
        self.validation_panel.add_widget(self.val_intensity_label)
        self.validation_panel.add_widget(self.val_clutter_label)
        self.validation_panel.add_widget(self.val_noise_label)
        self.validation_panel.add_widget(self.val_sparsity_label)
        y_offset += 210

        # --- Baseline Editor panel ---
        self.editor_panel = Panel(10, y_offset, panel_width, 130, "BASELINE EDITOR")
        ed_btn_w = (panel_width - 50) // 3
        self.load_baseline_button = Button(
            15, 35, ed_btn_w, 28, "LOAD",
            callback=self._on_load_baseline)
        self.add_target_button = Button(
            20 + ed_btn_w, 35, ed_btn_w, 28, "ADD TGT",
            callback=self._on_add_editor_target)
        self.export_baseline_button = Button(
            25 + ed_btn_w * 2, 35, ed_btn_w, 28, "EXPORT",
            callback=self._on_export_baseline)
        self.editor_label = Label(20, 70, "No baseline loaded", 16)
        self.editor_panel.add_widget(self.load_baseline_button)
        self.editor_panel.add_widget(self.add_target_button)
        self.editor_panel.add_widget(self.export_baseline_button)
        self.editor_panel.add_widget(self.editor_label)
        y_offset += 140

        # Info panel
        self.info_panel = Panel(10, y_offset, panel_width, 100, "INFO")
        self.time_label = Label(20, 35, "Time: 0.0s", 20)
        self.targets_label = Label(20, 55, "Targets: 0", 20)
        self.terrain_info_label = Label(20, 75, "Terrain: none", 18)
        self.info_panel.add_widget(self.time_label)
        self.info_panel.add_widget(self.targets_label)
        self.info_panel.add_widget(self.terrain_info_label)

        self.content_height = y_offset + 100

        self.panels = [
            self.scenario_panel,
            self.range_panel,
            self.gain_panel,
            self.sim_panel,
            self.terrain_panel,
            self.place_panel,
            self.weather_panel,
            self.export_panel,
            self.validation_panel,
            self.editor_panel,
            self.info_panel,
        ]

        # Scene view reference (for placement mode)
        self.scene_view = None

        # Validation reference data
        self._reference_ppi = None
        self._reference_path = None

    def set_scene_view(self, scene_view) -> None:
        self.scene_view = scene_view

    def set_simulation(self, sim: Simulation) -> None:
        self.simulation = sim

    def set_scenario_manager(self, manager: ScenarioManager) -> None:
        self.scenario_manager = manager
        self._scenario_names = manager.get_scenario_names()
        if self._scenario_names:
            self.scenario_slider.max_val = float(len(self._scenario_names) - 1)
            self.scenario_slider.value = 0.0
            self.scenario_name_label.set_text(self._scenario_names[0])

    # --- Callbacks ---

    def _on_scenario_slider(self, value: float) -> None:
        if not self.scenario_manager or not self.simulation:
            return
        index = int(round(value))
        index = max(0, min(index, len(self._scenario_names) - 1))
        self.scenario_slider.value = float(index)
        name = self._scenario_names[index]
        self.scenario_name_label.set_text(name)
        self.scenario_manager.load_scenario(
            name, self.simulation.world, self.simulation.radar,
            self.simulation.weather, simulation=self.simulation)
        self.pause_button.text = "PAUSE"

    def _on_load_location(self) -> None:
        """Open file dialog to load a .radarloc location file."""
        if not self.scenario_manager or not self.simulation:
            return
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(
            title="Select Location File",
            filetypes=[("Radar Location files", "*.radarloc"), ("All files", "*.*")])
        root.destroy()
        if filepath:
            success = self.scenario_manager.load_location_file(
                filepath, self.simulation.world, self.simulation.radar,
                self.simulation.weather, simulation=self.simulation)
            if success:
                # Update label with location name
                from ..scenarios.location_loader import load_radarloc_file
                try:
                    loc_data = load_radarloc_file(filepath)
                    name = loc_data.location_name
                    if len(name) > 40:
                        name = name[:37] + "..."
                    self.scenario_name_label.set_text(name)
                except Exception:
                    self.scenario_name_label.set_text("Location loaded")
            else:
                self.scenario_name_label.set_text("Load failed")
            self.pause_button.text = "PAUSE"

    def _on_range_input(self, text: str) -> None:
        if self.simulation:
            try:
                range_nm = float(text.strip().replace('nm', '').strip())
                if 0.1 <= range_nm <= 200:
                    self.simulation.radar.params.current_range_nm = range_nm
            except ValueError:
                pass

    def _on_range_change(self, index: int, value: str) -> None:
        if self.simulation:
            range_nm = float(value.split()[0])
            self.simulation.radar.set_range_scale(range_nm)
            self.range_input.text = str(range_nm)

    def _on_gain_change(self, value: float) -> None:
        if self.simulation:
            self.simulation.radar.set_gain(value)

    def _on_sea_change(self, value: float) -> None:
        if self.simulation:
            self.simulation.radar.set_sea_clutter(value)

    def _on_rain_change(self, value: float) -> None:
        if self.simulation:
            self.simulation.radar.set_rain_clutter(value)

    def _on_pause(self) -> None:
        if self.simulation:
            is_paused = self.simulation.toggle_pause()
            self.pause_button.text = "RESUME" if is_paused else "PAUSE"

    def _on_reset(self) -> None:
        if self.simulation:
            self.simulation.reset()
            self.simulation.setup_default_scenario()
            self.pause_button.text = "PAUSE"

    def _on_speed_change(self, value: float) -> None:
        if self.simulation:
            self.simulation.set_time_scale(value)

    def _on_add_island(self) -> None:
        if not self.simulation:
            return
        from ..environment.terrain import create_island_terrain
        own = self.simulation.world.own_ship
        cx = own.x if own else 0
        cy = (own.y if own else 0) + 3000
        hm = create_island_terrain(center_x=cx, center_y=cy,
                                   radius=800, peak_height=120)
        self.simulation.add_terrain(hm)

    def _on_add_ridge(self) -> None:
        if not self.simulation:
            return
        from ..environment.terrain import create_ridge_terrain
        own = self.simulation.world.own_ship
        cx = (own.x if own else 0) - 2000
        cy = (own.y if own else 0) + 3000
        hm = create_ridge_terrain(start_x=cx, start_y=cy,
                                  end_x=cx + 4000, end_y=cy + 500,
                                  width=500, peak_height=80)
        self.simulation.add_terrain(hm)

    def _on_clear_terrain(self) -> None:
        if self.simulation:
            self.simulation.clear_terrain()

    def _on_place_object(self, vessel_type: str) -> None:
        if self.scene_view is not None:
            self.scene_view.placement_mode = vessel_type
            self.place_label.set_text(f"Placing: {vessel_type.upper()}")

    def _on_cancel_placement(self) -> None:
        if self.scene_view is not None:
            self.scene_view.placement_mode = None
            self.place_label.set_text("Click scene to place")

    def _on_coastline_toggle(self) -> None:
        if self.simulation:
            if self.simulation.coastline_enabled:
                self.simulation.clear_coastlines()
            else:
                self.simulation.setup_harbor_coastline()

    def _on_sea_state_slider(self, value: float) -> None:
        index = int(round(value))
        index = max(0, min(index, len(self._sea_state_values) - 1))
        self.sea_state_slider.value = float(index)
        self.sea_state_label.set_text(self._sea_state_names[index])
        if self.simulation:
            self.simulation.weather.set_sea_state(self._sea_state_values[index])

    def _on_rain_rate_change(self, value: float) -> None:
        if self.simulation:
            self.simulation.weather.set_rain(value)

    def _on_record(self) -> None:
        if self.simulation:
            if self.simulation.is_recording():
                filepath = self.simulation.stop_recording()
                self.record_button.text = "RECORD"
                if filepath:
                    filename = os.path.basename(filepath)
                    self.record_label.set_text("Stopped")
                    self.file_label.set_text(f"Saved: {filename}")
                else:
                    self.record_label.set_text("No data recorded")
                    self.file_label.set_text("")
            else:
                session_id = self.simulation.start_recording()
                self.record_button.text = "STOP"
                self.record_label.set_text("Recording...")
                self.file_label.set_text(f"Session: {session_id}")

    def _on_save(self) -> None:
        if self.simulation:
            bearing = self.simulation.radar.get_current_bearing()
            filepath = self.simulation.export_current_sweep(bearing)
            if filepath:
                filename = os.path.basename(filepath)
                self.file_label.set_text(f"Saved: {filename}")

    def _on_load_csv(self) -> None:
        if self.csv_player and self.csv_player.active:
            self.csv_player.stop()
            self.csv_player = None
            self.csv_button.text = "LOAD CSV"
            self.file_label.set_text("CSV stopped")
        else:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            folder = filedialog.askdirectory(title="Select CSV Data Folder")
            root.destroy()
            if folder:
                self.csv_player = CsvPlayer(folder)
                if self.csv_player.active:
                    self.csv_button.text = "STOP CSV"
                    self.file_label.set_text(f"CSV: {os.path.basename(folder)}")
                else:
                    self.csv_player = None
                    self.file_label.set_text("No CSV files found")

    def _on_load_capture(self) -> None:
        """Analyze a capture folder and auto-configure the simulator."""
        if not self.scenario_manager or not self.simulation:
            return

        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Select Capture Folder (CSV files)")
        root.destroy()

        if not folder:
            return

        self.capture_label.set_text("Analyzing...")

        try:
            from ..baseline_editor import CaptureAnalyzer
            analyzer = CaptureAnalyzer()
            metadata = analyzer.analyze_csv_folder(folder)

            if metadata is None:
                self.capture_label.set_text("Analysis failed")
                return

            # Load into simulator
            success = self.scenario_manager.load_from_capture(
                metadata,
                self.simulation.world,
                self.simulation.radar,
                self.simulation.weather,
                simulation=self.simulation
            )

            if success:
                # Update UI controls to reflect loaded settings
                self.gain_slider.value = metadata.gain
                self.sea_slider.value = metadata.sea_clutter
                self.rain_slider.value = metadata.rain_clutter

                # Find matching range dropdown index
                range_options = [0.25, 0.5, 0.75, 1.5, 3, 6, 12, 24, 48]
                closest_idx = 5  # default 6nm
                min_diff = float('inf')
                for i, r in enumerate(range_options):
                    diff = abs(r - metadata.range_nm)
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = i
                self.range_dropdown.selected = closest_idx
                self.range_input.text = f"{metadata.range_nm:.1f}"

                # Update labels
                num_objects = len(metadata.detected_objects)
                location_info = ""
                if metadata.location_name:
                    location_info = metadata.location_name
                elif metadata.gps_lat is not None:
                    location_info = f"{metadata.gps_lat:.4f}, {metadata.gps_lon:.4f}"

                if location_info:
                    self.capture_label.set_text(f"{num_objects} objects, {location_info}")
                else:
                    self.capture_label.set_text(f"Loaded: {num_objects} objects")

                if metadata.location_name and len(metadata.location_name) <= 40:
                    self.scenario_name_label.set_text(metadata.location_name)
                elif location_info:
                    self.scenario_name_label.set_text(f"Capture: {location_info[:35]}")
                else:
                    self.scenario_name_label.set_text("Capture loaded")

                self.pause_button.text = "PAUSE"

                # If GPS data available, optionally trigger location generation
                if metadata.gps_lat is not None and metadata.gps_lon is not None:
                    self._trigger_location_from_gps(metadata.gps_lat, metadata.gps_lon,
                                                    metadata.range_nm)
            else:
                self.capture_label.set_text("Load failed")

        except Exception as e:
            self.capture_label.set_text(f"Error: {str(e)[:30]}")

    def _trigger_location_from_gps(self, lat: float, lon: float, range_nm: float) -> None:
        """Attempt to generate and load a location file from GPS coordinates.

        This is a best-effort operation - if it fails, the capture still loads.
        """
        try:
            import subprocess
            import tempfile
            import os

            # Try to find the location generator
            generator_paths = [
                os.path.expanduser("~/marine-radar-location-generator/generate_location.py"),
                os.path.join(os.path.dirname(__file__), "..", "..", "..",
                            "marine-radar-location-generator", "generate_location.py"),
            ]

            generator_path = None
            for p in generator_paths:
                if os.path.isfile(p):
                    generator_path = p
                    break

            if generator_path is None:
                # Generator not found - skip location loading
                return

            # Generate location file
            with tempfile.NamedTemporaryFile(suffix='.radarloc', delete=False) as tmp:
                tmp_path = tmp.name

            result = subprocess.run(
                ["python", generator_path, f"{lat},{lon}",
                 "--range", str(range_nm), "-o", tmp_path],
                capture_output=True,
                timeout=60
            )

            if result.returncode == 0 and os.path.isfile(tmp_path):
                # Load the generated location file
                self.scenario_manager.load_location_file(
                    tmp_path,
                    self.simulation.world,
                    self.simulation.radar,
                    self.simulation.weather,
                    simulation=self.simulation
                )
                self.capture_label.set_text(self.capture_label.text + " +loc")

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        except Exception:
            # Location generation failed - not critical, capture still loaded
            pass

    # --- Validation callbacks ---

    def _on_load_reference(self) -> None:
        """Load a real Furuno CSV as reference for validation."""
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(
            title="Select Reference CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        root.destroy()
        if filepath:
            from ..validation.capture_loader import load_furuno_csv
            ppi = load_furuno_csv(filepath)
            if ppi is not None:
                self._reference_ppi = ppi
                self._reference_path = filepath
                self.val_overall_label.set_text(f"Loaded: {os.path.basename(filepath)}")
            else:
                self.val_overall_label.set_text("Failed to load CSV")

    def _on_compare(self) -> None:
        """Run validation comparison between reference and current synthetic data."""
        if self._reference_ppi is None:
            self.val_overall_label.set_text("No reference loaded")
            return
        if not self.simulation:
            return

        try:
            import numpy as np
        except ImportError:
            self.val_overall_label.set_text("numpy required")
            return

        # Build synthetic PPI from current simulation state
        num_bins = self._reference_ppi.shape[1]
        synth_ppi = np.zeros((360, num_bins), dtype=np.float32)
        for bearing in range(360):
            sweep = self.simulation.get_radar_sweep_data(float(bearing))
            # Resample if bin counts differ
            if len(sweep) != num_bins:
                ratio = len(sweep) / num_bins
                resampled = []
                for i in range(num_bins):
                    src_idx = min(int(i * ratio), len(sweep) - 1)
                    resampled.append(sweep[src_idx])
                sweep = resampled
            synth_ppi[bearing, :] = sweep[:num_bins]

        from ..validation.comparator import compare
        report = compare(self._reference_ppi, synth_ppi)
        self._validation_report = report

        self.val_overall_label.set_text(f"Overall: {report.overall_score:.3f}")
        self.val_blob_label.set_text(f"Blob: {report.blob_score:.3f}")
        self.val_intensity_label.set_text(f"Intensity: {report.intensity_score:.3f}")
        self.val_clutter_label.set_text(f"Clutter: {report.clutter_score:.3f}")
        self.val_noise_label.set_text(f"Noise: {report.noise_score:.3f}")
        self.val_sparsity_label.set_text(f"Sparsity: {report.sparsity_score:.3f}")

    # --- Editor callbacks ---

    def _on_load_baseline(self) -> None:
        """Load a real CSV as baseline for editing."""
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(
            title="Select Baseline CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        root.destroy()
        if filepath:
            from ..editor.baseline_editor import BaselineEditor
            self._baseline_editor = BaselineEditor()
            if self._baseline_editor.load_baseline(filepath):
                self.editor_label.set_text(f"Loaded: {os.path.basename(filepath)}")
            else:
                self._baseline_editor = None
                self.editor_label.set_text("Failed to load baseline")

    def _on_add_editor_target(self) -> None:
        """Add a synthetic target to the baseline editor at a default position."""
        if self._baseline_editor is None or not self._baseline_editor.is_loaded:
            self.editor_label.set_text("Load baseline first")
            return
        # Add at 3000m range, 45° bearing with moderate RCS
        self._baseline_editor.add_synthetic_target(3000.0, 45.0, 100.0, "cargo")
        self.editor_label.set_text("Target added (3km, 045°)")

    def _on_export_baseline(self) -> None:
        """Export the edited baseline."""
        if self._baseline_editor is None or not self._baseline_editor.is_loaded:
            self.editor_label.set_text("No baseline to export")
            return
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.asksaveasfilename(
            title="Export Edited Baseline",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")])
        root.destroy()
        if filepath:
            result = self._baseline_editor.export(filepath)
            if result:
                self.editor_label.set_text(f"Exported: {os.path.basename(filepath)}")
            else:
                self.editor_label.set_text("Export failed")

    def _update_max_scroll(self) -> None:
        self.max_scroll = max(0, self.content_height - self.rect.height)

    def handle_event(self, event: pygame.event.Event) -> bool:
        self._update_max_scroll()

        # Scrollbar dragging
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.max_scroll > 0:
                sb_x = self.rect.x + self.rect.width - 12
                sb_rect = pygame.Rect(sb_x, self.rect.y, 12, self.rect.height)
                if sb_rect.collidepoint(event.pos):
                    self.is_scrollbar_dragging = True
                    self.scroll_drag_start_y = event.pos[1]
                    self.scroll_drag_start_offset = self.scroll_offset
                    return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.is_scrollbar_dragging:
                self.is_scrollbar_dragging = False
                return True

        if event.type == pygame.MOUSEMOTION and self.is_scrollbar_dragging:
            dy = event.pos[1] - self.scroll_drag_start_y
            ratio = dy / self.rect.height
            self.scroll_offset = self.scroll_drag_start_offset + ratio * self.content_height
            self.scroll_offset = max(0, min(self.max_scroll, self.scroll_offset))
            return True

        if event.type == pygame.MOUSEWHEEL:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll_offset -= event.y * 30
                self.scroll_offset = max(0, min(self.max_scroll, self.scroll_offset))
                return True

        # Temporarily offset widget positions for event handling
        for panel in self.panels:
            panel.rect.x += self.rect.x
            panel.rect.y += self.rect.y - int(self.scroll_offset)
            for widget in panel.widgets:
                widget.rect.x += self.rect.x
                widget.rect.y += self.rect.y - int(self.scroll_offset)

        handled = False
        for panel in reversed(self.panels):
            if panel.handle_event(event):
                handled = True
                break

        for panel in self.panels:
            panel.rect.x -= self.rect.x
            panel.rect.y -= self.rect.y - int(self.scroll_offset)
            for widget in panel.widgets:
                widget.rect.x -= self.rect.x
                widget.rect.y -= self.rect.y - int(self.scroll_offset)

        return handled

    def update(self) -> None:
        if not self.simulation:
            return
        self.time_label.set_text(f"Time: {self.simulation.world.time:.1f}s")
        self.targets_label.set_text(f"Targets: {len(self.simulation.world.get_targets())}")

        # Terrain info
        n_maps = len(self.simulation.terrain_maps)
        if n_maps > 0:
            self.terrain_info_label.set_text(f"Terrain: {n_maps} map(s)")
            self.terrain_label.set_text(f"{n_maps} terrain map(s) active")
        else:
            self.terrain_info_label.set_text("Terrain: none")
            self.terrain_label.set_text("No terrain")

        if self.simulation.is_recording():
            count = self.simulation.get_record_count()
            self.record_label.set_text(f"Recording: {count} sweeps")

        if self.simulation.coastline_enabled:
            self.coastline_button.text = "COASTLINE OFF"
        else:
            self.coastline_button.text = "COASTLINE ON"

    def draw(self, surface: pygame.Surface) -> None:
        self.update()
        self._update_max_scroll()

        bg_rect = pygame.Rect(
            self.rect.x - 10, self.rect.y - 10,
            self.rect.width + 20, self.rect.height + 20)
        pygame.draw.rect(surface, COLORS['bg'], bg_rect)

        surface.set_clip(self.rect)
        scroll_y = int(self.scroll_offset)

        for panel in self.panels:
            original_x = panel.rect.x
            original_y = panel.rect.y
            panel.rect.x += self.rect.x
            panel.rect.y += self.rect.y - scroll_y
            for widget in panel.widgets:
                widget.rect.x += self.rect.x
                widget.rect.y += self.rect.y - scroll_y
            panel.draw(surface)
            panel.rect.x = original_x
            panel.rect.y = original_y
            for widget in panel.widgets:
                widget.rect.x -= self.rect.x
                widget.rect.y -= self.rect.y - scroll_y

        surface.set_clip(None)

        if self.max_scroll > 0:
            sb_x = self.rect.x + self.rect.width - 10
            sb_track = pygame.Rect(sb_x, self.rect.y, 8, self.rect.height)
            pygame.draw.rect(surface, COLORS['panel'], sb_track)
            thumb_ratio = self.rect.height / self.content_height
            thumb_h = max(20, int(self.rect.height * thumb_ratio))
            thumb_y = self.rect.y + int((self.rect.height - thumb_h) * self.scroll_offset / self.max_scroll)
            thumb_rect = pygame.Rect(sb_x, thumb_y, 8, thumb_h)
            color = COLORS['highlight'] if self.is_scrollbar_dragging else COLORS['slider_knob']
            pygame.draw.rect(surface, color, thumb_rect, border_radius=4)
