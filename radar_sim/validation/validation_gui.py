#!/usr/bin/env python3
"""
Radar Validation Tool GUI

Compare real radar captures against simulator output with comprehensive
metrics and visualizations. Supports single file and folder comparisons.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from radar_sim.validation.validator import (
    RadarValidator, ValidationVisualizer,
    write_summary_report, write_batch_report
)


class ValidationApp(tk.Tk):
    """Main validation tool GUI."""

    def __init__(self, simulation=None):
        super().__init__()
        self.title("Radar Validation Tool")
        self.geometry("700x600")
        self.resizable(True, True)

        self.simulation = simulation
        self.validator = RadarValidator()
        self.visualizer = None

        self._create_widgets()

    def _create_widgets(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab 1: Single File Comparison
        self.single_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.single_frame, text='Single File')
        self._create_single_tab()

        # Tab 2: Folder Comparison
        self.folder_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.folder_frame, text='Folder Comparison')
        self._create_folder_tab()

        # Tab 3: Live Simulator Comparison
        self.live_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.live_frame, text='vs Live Simulator')
        self._create_live_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief='sunken', anchor='w')
        status_bar.pack(fill='x', side='bottom', padx=10, pady=5)

    def _create_single_tab(self):
        """Create single file comparison tab."""
        frame = self.single_frame
        pad = {'padx': 10, 'pady': 5}

        # Real CSV
        row = 0
        ttk.Label(frame, text="Real CSV:").grid(row=row, column=0, sticky='e', **pad)
        self.real_csv_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.real_csv_var, width=50).grid(row=row, column=1, **pad)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_file(self.real_csv_var)).grid(row=row, column=2, **pad)

        # Sim CSV
        row += 1
        ttk.Label(frame, text="Sim CSV:").grid(row=row, column=0, sticky='e', **pad)
        self.sim_csv_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.sim_csv_var, width=50).grid(row=row, column=1, **pad)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_file(self.sim_csv_var)).grid(row=row, column=2, **pad)

        # Output dir
        row += 1
        ttk.Label(frame, text="Output Dir:").grid(row=row, column=0, sticky='e', **pad)
        self.single_output_var = tk.StringVar(value="validation_output")
        ttk.Entry(frame, textvariable=self.single_output_var, width=50).grid(row=row, column=1, **pad)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_dir(self.single_output_var)).grid(row=row, column=2, **pad)

        # Thresholds
        row += 1
        thresh_frame = ttk.Frame(frame)
        thresh_frame.grid(row=row, column=0, columnspan=3, **pad)
        ttk.Label(thresh_frame, text="Coastline threshold:").pack(side='left')
        self.coast_thresh_var = tk.StringVar(value="0.3")
        ttk.Entry(thresh_frame, textvariable=self.coast_thresh_var, width=6).pack(side='left', padx=(0, 20))
        ttk.Label(thresh_frame, text="Target threshold:").pack(side='left')
        self.target_thresh_var = tk.StringVar(value="0.5")
        ttk.Entry(thresh_frame, textvariable=self.target_thresh_var, width=6).pack(side='left')

        # Buttons
        row += 1
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=15)
        self.single_run_btn = ttk.Button(btn_frame, text="Run Comparison", command=self._run_single)
        self.single_run_btn.pack(side='left', padx=10)
        ttk.Button(btn_frame, text="Open Results", command=lambda: self._open_folder(self.single_output_var.get())).pack(side='left', padx=10)

        # Results display
        row += 1
        ttk.Label(frame, text="Results:").grid(row=row, column=0, sticky='ne', **pad)
        self.single_results = tk.Text(frame, height=15, width=60, state='disabled')
        self.single_results.grid(row=row, column=1, columnspan=2, **pad)

    def _create_folder_tab(self):
        """Create folder comparison tab."""
        frame = self.folder_frame
        pad = {'padx': 10, 'pady': 5}

        # Real folder
        row = 0
        ttk.Label(frame, text="Real CSV Folder:").grid(row=row, column=0, sticky='e', **pad)
        self.real_folder_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.real_folder_var, width=50).grid(row=row, column=1, **pad)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_dir(self.real_folder_var)).grid(row=row, column=2, **pad)

        # Sim folder
        row += 1
        ttk.Label(frame, text="Sim CSV Folder:").grid(row=row, column=0, sticky='e', **pad)
        self.sim_folder_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.sim_folder_var, width=50).grid(row=row, column=1, **pad)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_dir(self.sim_folder_var)).grid(row=row, column=2, **pad)

        # Output dir
        row += 1
        ttk.Label(frame, text="Output Dir:").grid(row=row, column=0, sticky='e', **pad)
        self.folder_output_var = tk.StringVar(value="validation_output_batch")
        ttk.Entry(frame, textvariable=self.folder_output_var, width=50).grid(row=row, column=1, **pad)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_dir(self.folder_output_var)).grid(row=row, column=2, **pad)

        # Progress bar
        row += 1
        self.folder_progress = ttk.Progressbar(frame, mode='indeterminate')
        self.folder_progress.grid(row=row, column=0, columnspan=3, sticky='ew', **pad)

        # Buttons
        row += 1
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=15)
        self.folder_run_btn = ttk.Button(btn_frame, text="Run Batch Comparison", command=self._run_folder)
        self.folder_run_btn.pack(side='left', padx=10)
        ttk.Button(btn_frame, text="Open Results", command=lambda: self._open_folder(self.folder_output_var.get())).pack(side='left', padx=10)

        # Results display
        row += 1
        ttk.Label(frame, text="Results:").grid(row=row, column=0, sticky='ne', **pad)
        self.folder_results = tk.Text(frame, height=15, width=60, state='disabled')
        self.folder_results.grid(row=row, column=1, columnspan=2, **pad)

    def _create_live_tab(self):
        """Create live simulator comparison tab."""
        frame = self.live_frame
        pad = {'padx': 10, 'pady': 5}

        # Status
        row = 0
        sim_status = "Connected" if self.simulation else "Not connected"
        self.sim_status_var = tk.StringVar(value=f"Simulator: {sim_status}")
        ttk.Label(frame, textvariable=self.sim_status_var, font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=3, **pad)

        # Real CSV/folder
        row += 1
        ttk.Label(frame, text="Compare:").grid(row=row, column=0, sticky='e', **pad)
        self.live_mode_var = tk.StringVar(value="file")
        ttk.Radiobutton(frame, text="Single File", variable=self.live_mode_var, value="file").grid(row=row, column=1, sticky='w')
        ttk.Radiobutton(frame, text="Folder", variable=self.live_mode_var, value="folder").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(frame, text="Real CSV/Folder:").grid(row=row, column=0, sticky='e', **pad)
        self.live_source_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.live_source_var, width=50).grid(row=row, column=1, **pad)
        ttk.Button(frame, text="Browse...", command=self._browse_live_source).grid(row=row, column=2, **pad)

        # Output dir
        row += 1
        ttk.Label(frame, text="Output Dir:").grid(row=row, column=0, sticky='e', **pad)
        self.live_output_var = tk.StringVar(value="validation_output_live")
        ttk.Entry(frame, textvariable=self.live_output_var, width=50).grid(row=row, column=1, **pad)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_dir(self.live_output_var)).grid(row=row, column=2, **pad)

        # Buttons
        row += 1
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=15)
        self.live_run_btn = ttk.Button(btn_frame, text="Compare vs Simulator", command=self._run_live)
        self.live_run_btn.pack(side='left', padx=10)
        ttk.Button(btn_frame, text="Open Results", command=lambda: self._open_folder(self.live_output_var.get())).pack(side='left', padx=10)

        # Results display
        row += 1
        ttk.Label(frame, text="Results:").grid(row=row, column=0, sticky='ne', **pad)
        self.live_results = tk.Text(frame, height=15, width=60, state='disabled')
        self.live_results.grid(row=row, column=1, columnspan=2, **pad)

    def _browse_file(self, var):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            var.set(path)

    def _browse_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _browse_live_source(self):
        if self.live_mode_var.get() == "file":
            path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        else:
            path = filedialog.askdirectory()
        if path:
            self.live_source_var.set(path)

    def _open_folder(self, folder):
        if not os.path.isdir(folder):
            messagebox.showinfo("No folder", f"Output folder does not exist:\n{folder}")
            return
        if sys.platform == 'win32':
            os.startfile(folder)
        else:
            import subprocess
            subprocess.Popen(['xdg-open', folder])

    def _set_status(self, msg):
        self.status_var.set(msg)
        self.update_idletasks()

    def _update_results(self, text_widget, text):
        text_widget.config(state='normal')
        text_widget.delete('1.0', tk.END)
        text_widget.insert('1.0', text)
        text_widget.config(state='disabled')

    def _run_single(self):
        """Run single file comparison."""
        real_csv = self.real_csv_var.get().strip()
        sim_csv = self.sim_csv_var.get().strip()
        output_dir = self.single_output_var.get().strip()

        if not real_csv or not sim_csv:
            messagebox.showwarning("Missing input", "Please select both Real and Sim CSV files.")
            return

        try:
            coast_t = float(self.coast_thresh_var.get())
            target_t = float(self.target_thresh_var.get())
        except ValueError:
            messagebox.showwarning("Invalid threshold", "Thresholds must be numbers.")
            return

        self.single_run_btn.config(state='disabled')
        self._set_status("Running comparison...")

        def worker():
            try:
                validator = RadarValidator(coastline_threshold=coast_t, target_threshold=target_t)
                result = validator.validate_file(real_csv, sim_csv)

                # Generate visualizations
                self.visualizer = ValidationVisualizer(output_dir)
                real_grid, _ = validator._load_and_normalize(real_csv)
                sim_grid, _ = validator._load_and_normalize(sim_csv)
                self.visualizer.generate_all(real_grid, sim_grid, result)

                # Write report
                write_summary_report(result, os.path.join(output_dir, "metrics_summary.txt"))

                # Display results
                text = self._format_result(result)
                self.after(0, lambda: self._update_results(self.single_results, text))
                self.after(0, lambda: self._set_status("Comparison complete."))

            except Exception as e:
                self.after(0, lambda: self._set_status(f"Error: {e}"))
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self.single_run_btn.config(state='normal'))

        threading.Thread(target=worker, daemon=True).start()

    def _run_folder(self):
        """Run folder batch comparison."""
        real_folder = self.real_folder_var.get().strip()
        sim_folder = self.sim_folder_var.get().strip()
        output_dir = self.folder_output_var.get().strip()

        if not real_folder or not sim_folder:
            messagebox.showwarning("Missing input", "Please select both Real and Sim folders.")
            return

        self.folder_run_btn.config(state='disabled')
        self.folder_progress.start()
        self._set_status("Running batch comparison...")

        def progress_cb(msg):
            self.after(0, lambda: self._set_status(msg))

        def worker():
            try:
                validator = RadarValidator()
                result = validator.validate_folder(real_folder, sim_folder, progress_callback=progress_cb)

                # Write batch report
                os.makedirs(output_dir, exist_ok=True)
                write_batch_report(result, os.path.join(output_dir, "batch_report.txt"))

                # Display results
                text = self._format_batch_result(result)
                self.after(0, lambda: self._update_results(self.folder_results, text))
                self.after(0, lambda: self._set_status("Batch comparison complete."))

            except Exception as e:
                self.after(0, lambda: self._set_status(f"Error: {e}"))
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self.folder_run_btn.config(state='normal'))
                self.after(0, lambda: self.folder_progress.stop())

        threading.Thread(target=worker, daemon=True).start()

    def _run_live(self):
        """Run comparison against live simulator."""
        if not self.simulation:
            messagebox.showwarning("No simulator", "Simulator not connected.\nRun this from within the simulator.")
            return

        source = self.live_source_var.get().strip()
        output_dir = self.live_output_var.get().strip()
        mode = self.live_mode_var.get()

        if not source:
            messagebox.showwarning("Missing input", "Please select a CSV file or folder.")
            return

        self.live_run_btn.config(state='disabled')
        self._set_status("Running live comparison...")

        def worker():
            try:
                validator = RadarValidator()

                if mode == "file":
                    result = validator.validate_against_simulator(source, self.simulation)

                    # Generate visualizations
                    self.visualizer = ValidationVisualizer(output_dir)
                    real_grid, _ = validator._load_and_normalize(source)
                    sim_grid = validator._get_simulator_grid(self.simulation)
                    self.visualizer.generate_all(real_grid, sim_grid, result)
                    write_summary_report(result, os.path.join(output_dir, "metrics_summary.txt"))

                    text = self._format_result(result)
                else:
                    result = validator.validate_folder_against_simulator(
                        source, self.simulation,
                        progress_callback=lambda msg: self.after(0, lambda: self._set_status(msg))
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    write_batch_report(result, os.path.join(output_dir, "batch_report.txt"))
                    text = self._format_batch_result(result)

                self.after(0, lambda: self._update_results(self.live_results, text))
                self.after(0, lambda: self._set_status("Live comparison complete."))

            except Exception as e:
                self.after(0, lambda: self._set_status(f"Error: {e}"))
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self.live_run_btn.config(state='normal'))

        threading.Thread(target=worker, daemon=True).start()

    def _format_result(self, result) -> str:
        """Format single result for display."""
        rpb = result.range_per_bin_m
        lines = [
            "=== Validation Results ===",
            "",
            f"Overall RMSE:        {result.overall_rmse:.4f}",
            f"Pearson Correlation: {result.pearson_r:.4f}",
        ]
        if result.ssim_value is not None:
            lines.append(f"SSIM:                {result.ssim_value:.4f}")

        lines.extend([
            "",
            f"Coverage (real):     {result.coverage_real:.1%}",
            f"Coverage (sim):      {result.coverage_sim:.1%}",
            "",
            "--- Coastline Error ---",
            f"Mean: {result.coastline_mean_error_bins:.1f} bins ({result.coastline_mean_error_bins * rpb:.1f} m)",
            f"RMS:  {result.coastline_rms_error_bins:.1f} bins ({result.coastline_rms_error_bins * rpb:.1f} m)",
            "",
            "--- Target Matching ---",
            f"Real targets:  {result.real_target_count}",
            f"Sim targets:   {result.sim_target_count}",
            f"Matched:       {len(result.target_matches)}",
            f"Missed:        {result.missed_targets}",
            f"False alarms:  {result.false_alarm_targets}",
        ])

        return "\n".join(lines)

    def _format_batch_result(self, result) -> str:
        """Format batch result for display."""
        lines = [
            "=== Batch Validation Results ===",
            "",
            f"Files processed: {result.files_processed}",
            f"Files failed:    {result.files_failed}",
            "",
            "--- Aggregate Metrics ---",
            f"Mean RMSE:    {result.mean_rmse:.4f}",
            f"Mean Pearson: {result.mean_pearson:.4f}",
        ]
        if result.mean_ssim is not None:
            lines.append(f"Mean SSIM:    {result.mean_ssim:.4f}")

        lines.extend([
            "",
            "--- Target Statistics ---",
            f"Total matched:      {result.total_matched_targets}",
            f"Total missed:       {result.total_missed_targets}",
            f"Total false alarms: {result.total_false_alarms}",
        ])

        return "\n".join(lines)

    def set_simulation(self, simulation):
        """Connect to simulator instance."""
        self.simulation = simulation
        status = "Connected" if simulation else "Not connected"
        self.sim_status_var.set(f"Simulator: {status}")


def launch_validator(simulation=None):
    """Launch the validation GUI."""
    app = ValidationApp(simulation)
    app.mainloop()


if __name__ == "__main__":
    launch_validator()
