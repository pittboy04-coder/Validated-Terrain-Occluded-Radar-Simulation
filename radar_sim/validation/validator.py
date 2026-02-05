"""
Advanced Radar Validation Framework

Compare real radar captures against simulator output with comprehensive metrics,
visualizations, and batch processing support.
"""

import csv
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy import ndimage
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Constants
NUM_BEARINGS = 360
NUM_RANGE_BINS = 512


@dataclass
class Target:
    """Detected radar target."""
    centroid_bearing: float  # degrees
    centroid_range_bin: float
    peak_intensity: float
    area_bins: int
    extent_bearing: float
    extent_range: float


@dataclass
class TargetMatch:
    """Matched target pair between real and sim."""
    real_idx: int
    sim_idx: int
    range_offset_bins: float
    bearing_offset_deg: float
    distance: float


@dataclass
class ValidationResult:
    """Comprehensive validation metrics."""
    # Core metrics
    overall_rmse: float = 0.0
    pearson_r: float = 0.0
    ssim_value: Optional[float] = None

    # Per-bearing analysis
    per_bearing_rmse: Optional[np.ndarray] = None

    # Coastline metrics
    coastline_mean_error_bins: float = 0.0
    coastline_max_error_bins: float = 0.0
    coastline_rms_error_bins: float = 0.0

    # Target matching
    target_matches: List[TargetMatch] = field(default_factory=list)
    missed_targets: int = 0
    false_alarm_targets: int = 0
    real_target_count: int = 0
    sim_target_count: int = 0

    # Coverage
    coverage_real: float = 0.0
    coverage_sim: float = 0.0

    # Metadata
    range_m: float = 11112.0
    range_per_bin_m: float = 21.7
    source_real: str = ""
    source_sim: str = ""
    timestamp: str = ""


@dataclass
class BatchResult:
    """Results from batch folder comparison."""
    individual_results: List[ValidationResult] = field(default_factory=list)
    mean_rmse: float = 0.0
    mean_pearson: float = 0.0
    mean_ssim: Optional[float] = None
    total_matched_targets: int = 0
    total_missed_targets: int = 0
    total_false_alarms: int = 0
    files_processed: int = 0
    files_failed: int = 0


class RadarValidator:
    """Advanced radar data validation engine."""

    def __init__(self, coastline_threshold: float = 0.3,
                 target_threshold: float = 0.5,
                 target_min_area: int = 3):
        self.coastline_threshold = coastline_threshold
        self.target_threshold = target_threshold
        self.target_min_area = target_min_area

    def validate_file(self, real_csv: str, sim_csv: str) -> ValidationResult:
        """Compare a single real CSV against a simulator CSV."""
        if not HAS_NUMPY:
            raise RuntimeError("numpy required for validation")

        # Load and normalize both datasets
        real_grid, real_range = self._load_and_normalize(real_csv)
        sim_grid, sim_range = self._load_and_normalize(sim_csv)

        range_m = max(real_range, sim_range) if real_range > 0 else 11112.0

        return self._compare_grids(real_grid, sim_grid, range_m,
                                   source_real=real_csv, source_sim=sim_csv)

    def validate_against_simulator(self, real_csv: str,
                                   simulation) -> ValidationResult:
        """Compare a real CSV against current simulator state."""
        if not HAS_NUMPY:
            raise RuntimeError("numpy required for validation")

        # Load real data
        real_grid, real_range = self._load_and_normalize(real_csv)

        # Get simulator data
        sim_grid = self._get_simulator_grid(simulation)
        range_m = simulation.radar.params.current_range_nm * 1852.0

        return self._compare_grids(real_grid, sim_grid, range_m,
                                   source_real=real_csv,
                                   source_sim="[Live Simulator]")

    def validate_folder(self, real_folder: str, sim_folder: str,
                        progress_callback: Callable = None) -> BatchResult:
        """Compare all CSVs in two folders."""
        result = BatchResult()

        # Find matching CSV files
        real_files = sorted(Path(real_folder).glob("*.csv"))
        sim_files = sorted(Path(sim_folder).glob("*.csv"))

        # Match by filename or index
        pairs = self._match_files(real_files, sim_files)

        for i, (real_path, sim_path) in enumerate(pairs):
            if progress_callback:
                progress_callback(f"Processing {i+1}/{len(pairs)}: {real_path.name}")

            try:
                individual = self.validate_file(str(real_path), str(sim_path))
                result.individual_results.append(individual)
                result.files_processed += 1
            except Exception as e:
                print(f"Failed to process {real_path.name}: {e}")
                result.files_failed += 1

        # Aggregate statistics
        if result.individual_results:
            result.mean_rmse = np.mean([r.overall_rmse for r in result.individual_results])
            result.mean_pearson = np.mean([r.pearson_r for r in result.individual_results])

            ssim_vals = [r.ssim_value for r in result.individual_results if r.ssim_value is not None]
            if ssim_vals:
                result.mean_ssim = np.mean(ssim_vals)

            result.total_matched_targets = sum(len(r.target_matches) for r in result.individual_results)
            result.total_missed_targets = sum(r.missed_targets for r in result.individual_results)
            result.total_false_alarms = sum(r.false_alarm_targets for r in result.individual_results)

        return result

    def validate_folder_against_simulator(self, real_folder: str,
                                          simulation,
                                          progress_callback: Callable = None) -> BatchResult:
        """Compare all CSVs in a folder against current simulator state."""
        result = BatchResult()

        # Get simulator grid once
        sim_grid = self._get_simulator_grid(simulation)
        range_m = simulation.radar.params.current_range_nm * 1852.0

        real_files = sorted(Path(real_folder).glob("*.csv"))

        for i, real_path in enumerate(real_files):
            if progress_callback:
                progress_callback(f"Processing {i+1}/{len(real_files)}: {real_path.name}")

            try:
                real_grid, _ = self._load_and_normalize(str(real_path))
                individual = self._compare_grids(real_grid, sim_grid, range_m,
                                                source_real=str(real_path),
                                                source_sim="[Live Simulator]")
                result.individual_results.append(individual)
                result.files_processed += 1
            except Exception as e:
                print(f"Failed to process {real_path.name}: {e}")
                result.files_failed += 1

        # Aggregate
        if result.individual_results:
            result.mean_rmse = np.mean([r.overall_rmse for r in result.individual_results])
            result.mean_pearson = np.mean([r.pearson_r for r in result.individual_results])

        return result

    def _load_and_normalize(self, csv_path: str) -> Tuple[np.ndarray, float]:
        """Load CSV and normalize to (360, 512) grid."""
        data, range_m = self._load_csv(csv_path)
        grid = self._normalize_to_grid(data)
        return grid, range_m

    def _load_csv(self, csv_path: str) -> Tuple[List, float]:
        """Load radar CSV with auto-format detection."""
        sweeps = []
        range_m = 0.0

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return sweeps, range_m

            header_lower = [h.lower().strip() for h in header]

            # Detect format
            if 'angle' in header_lower:
                # Format: Status,Scale,Range,Gain,Angle,EchoValues...
                angle_col = header_lower.index('angle')
                meta_cols = angle_col + 1

                # Read first row to get actual column count
                first_row = next(reader, None)
                if first_row:
                    num_bins = len(first_row) - meta_cols
                    angle_scale = self._detect_angle_scale(csv_path, angle_col)

                    # Process first row
                    self._process_row(first_row, angle_col, meta_cols, angle_scale, sweeps)

                    # Extract range
                    if 'range' in header_lower:
                        range_col = header_lower.index('range')
                        try:
                            range_code = int(first_row[range_col])
                            range_m = self._range_code_to_meters(range_code)
                        except (ValueError, IndexError):
                            pass

                    # Process remaining rows
                    for row in reader:
                        self._process_row(row, angle_col, meta_cols, angle_scale, sweeps)
            else:
                # Furuno format or simulator format
                angle_col = 4
                meta_cols = 5
                angle_scale = 8192.0 / 360.0

                for row in reader:
                    self._process_row(row, angle_col, meta_cols, angle_scale, sweeps)

        return sweeps, range_m

    def _process_row(self, row, angle_col, meta_cols, angle_scale, sweeps):
        """Process a single CSV row into sweep data."""
        if len(row) <= meta_cols:
            return
        try:
            angle_raw = float(row[angle_col])
            bearing_deg = (angle_raw / angle_scale) % 360
            echoes = [float(v) for v in row[meta_cols:]]

            # Normalize to 0-1 if needed
            max_val = max(echoes) if echoes else 1.0
            if max_val > 1.5:
                echoes = [v / 252.0 for v in echoes]

            sweeps.append((bearing_deg, echoes))
        except (ValueError, IndexError):
            pass

    def _detect_angle_scale(self, csv_path: str, angle_col: int) -> float:
        """Detect angle tick scaling."""
        angles = []
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for i, row in enumerate(reader):
                    if i > 1000:
                        break
                    try:
                        angles.append(float(row[angle_col]))
                    except (ValueError, IndexError):
                        continue
        except IOError:
            return 1.0

        if not angles:
            return 1.0

        max_angle = max(angles)
        if max_angle > 7000:
            return 8192.0 / 360.0
        elif max_angle > 3500:
            return 4096.0 / 360.0
        elif max_angle > 360:
            return max_angle / 360.0
        return 1.0

    def _range_code_to_meters(self, code: int) -> float:
        """Convert Furuno range code to meters."""
        furuno_ranges_nm = [0.125, 0.25, 0.5, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 72.0, 96.0]
        if 0 <= code < len(furuno_ranges_nm):
            return furuno_ranges_nm[code] * 1852.0
        return 6.0 * 1852.0

    def _normalize_to_grid(self, sweeps: List) -> np.ndarray:
        """Normalize sweep data to (360, 512) grid."""
        grid = np.zeros((NUM_BEARINGS, NUM_RANGE_BINS), dtype=np.float32)
        counts = np.zeros(NUM_BEARINGS, dtype=np.int32)

        for bearing_deg, echoes in sweeps:
            bearing_idx = int(round(bearing_deg)) % NUM_BEARINGS

            # Resample to NUM_RANGE_BINS
            if len(echoes) == 0:
                continue
            if len(echoes) == NUM_RANGE_BINS:
                resampled = np.array(echoes)
            else:
                x_src = np.linspace(0, 1, len(echoes))
                x_dst = np.linspace(0, 1, NUM_RANGE_BINS)
                resampled = np.interp(x_dst, x_src, echoes)

            if counts[bearing_idx] == 0:
                grid[bearing_idx] = resampled
            else:
                grid[bearing_idx] = np.maximum(grid[bearing_idx], resampled)
            counts[bearing_idx] += 1

        # Interpolate missing bearings
        valid = counts > 0
        if valid.any() and not valid.all():
            valid_idx = np.where(valid)[0]
            missing_idx = np.where(~valid)[0]
            for col in range(NUM_RANGE_BINS):
                vals = grid[valid_idx, col]
                grid[missing_idx, col] = np.interp(missing_idx, valid_idx, vals)

        return np.clip(grid, 0.0, 1.0)

    def _get_simulator_grid(self, simulation) -> np.ndarray:
        """Extract current radar data from simulator."""
        grid = np.zeros((NUM_BEARINGS, NUM_RANGE_BINS), dtype=np.float32)

        for bearing in range(NUM_BEARINGS):
            sweep = simulation.get_radar_sweep_data(float(bearing))
            if len(sweep) == NUM_RANGE_BINS:
                grid[bearing] = sweep
            else:
                x_src = np.linspace(0, 1, len(sweep))
                x_dst = np.linspace(0, 1, NUM_RANGE_BINS)
                grid[bearing] = np.interp(x_dst, x_src, sweep)

        return np.clip(grid, 0.0, 1.0)

    def _compare_grids(self, real_grid: np.ndarray, sim_grid: np.ndarray,
                       range_m: float, source_real: str = "",
                       source_sim: str = "") -> ValidationResult:
        """Compute all comparison metrics between two grids."""
        result = ValidationResult()
        result.range_m = range_m
        result.range_per_bin_m = range_m / NUM_RANGE_BINS
        result.source_real = source_real
        result.source_sim = source_sim
        result.timestamp = datetime.now().isoformat()

        # Per-bearing RMSE
        result.per_bearing_rmse = np.sqrt(np.mean((real_grid - sim_grid) ** 2, axis=1))

        # Overall RMSE
        result.overall_rmse = float(np.sqrt(np.mean((real_grid - sim_grid) ** 2)))

        # Pearson correlation
        r_flat = real_grid.flatten()
        s_flat = sim_grid.flatten()
        if np.std(r_flat) > 0 and np.std(s_flat) > 0:
            if HAS_SCIPY:
                result.pearson_r, _ = pearsonr(r_flat, s_flat)
            else:
                result.pearson_r = float(np.corrcoef(r_flat, s_flat)[0, 1])

        # SSIM
        if HAS_SKIMAGE:
            result.ssim_value = float(ssim(real_grid, sim_grid, data_range=1.0))

        # Extract features
        real_coastline = self._extract_coastline(real_grid)
        sim_coastline = self._extract_coastline(sim_grid)
        real_targets = self._extract_targets(real_grid)
        sim_targets = self._extract_targets(sim_grid)

        # Coastline errors
        both_valid = ~np.isnan(real_coastline) & ~np.isnan(sim_coastline)
        if both_valid.any():
            errors = np.abs(real_coastline[both_valid] - sim_coastline[both_valid])
            result.coastline_mean_error_bins = float(errors.mean())
            result.coastline_max_error_bins = float(errors.max())
            result.coastline_rms_error_bins = float(np.sqrt(np.mean(errors ** 2)))

        # Target matching
        result.real_target_count = len(real_targets)
        result.sim_target_count = len(sim_targets)
        result.target_matches, result.missed_targets, result.false_alarm_targets = \
            self._match_targets(real_targets, sim_targets)

        # Coverage
        sig_threshold = 0.05
        result.coverage_real = float(np.mean(np.any(real_grid > sig_threshold, axis=1)))
        result.coverage_sim = float(np.mean(np.any(sim_grid > sig_threshold, axis=1)))

        return result

    def _extract_coastline(self, grid: np.ndarray) -> np.ndarray:
        """Find coastline boundary per bearing."""
        coastline = np.full(NUM_BEARINGS, np.nan)

        for b in range(NUM_BEARINGS):
            row = grid[b]
            above = row > self.coastline_threshold
            run_start = None
            run_len = 0

            for r in range(NUM_RANGE_BINS):
                if above[r]:
                    if run_start is None:
                        run_start = r
                    run_len += 1
                    if run_len >= 5:
                        coastline[b] = run_start
                        break
                else:
                    run_start = None
                    run_len = 0

        return coastline

    def _extract_targets(self, grid: np.ndarray) -> List[Target]:
        """Detect target blobs above threshold."""
        if not HAS_SCIPY:
            return []

        binary = grid > self.target_threshold
        labeled, n_features = ndimage.label(binary)
        targets = []

        for i in range(1, n_features + 1):
            region = labeled == i
            area = region.sum()
            if area < self.target_min_area:
                continue

            coords = np.argwhere(region)
            bearings = coords[:, 0]
            ranges = coords[:, 1]
            values = grid[region]

            targets.append(Target(
                centroid_bearing=float(bearings.mean()),
                centroid_range_bin=float(ranges.mean()),
                peak_intensity=float(values.max()),
                area_bins=int(area),
                extent_bearing=float(bearings.max() - bearings.min()),
                extent_range=float(ranges.max() - ranges.min()),
            ))

        return targets

    def _match_targets(self, real_targets: List[Target],
                       sim_targets: List[Target]) -> Tuple[List[TargetMatch], int, int]:
        """Match real targets to simulated targets."""
        matches = []
        matched_sim = set()
        max_match_distance = 50  # bins

        for ri, rt in enumerate(real_targets):
            best_dist = float('inf')
            best_si = -1

            for si, st in enumerate(sim_targets):
                if si in matched_sim:
                    continue

                db = rt.centroid_bearing - st.centroid_bearing
                dr = rt.centroid_range_bin - st.centroid_range_bin
                dist = math.sqrt(db ** 2 + dr ** 2)

                if dist < best_dist:
                    best_dist = dist
                    best_si = si

            if best_si >= 0 and best_dist < max_match_distance:
                matched_sim.add(best_si)
                st = sim_targets[best_si]
                matches.append(TargetMatch(
                    real_idx=ri,
                    sim_idx=best_si,
                    range_offset_bins=rt.centroid_range_bin - st.centroid_range_bin,
                    bearing_offset_deg=rt.centroid_bearing - st.centroid_bearing,
                    distance=best_dist,
                ))

        missed = len(real_targets) - len(matches)
        false_alarms = len(sim_targets) - len(matches)

        return matches, missed, false_alarms

    def _match_files(self, real_files: List[Path],
                     sim_files: List[Path]) -> List[Tuple[Path, Path]]:
        """Match files between two folders by name or index."""
        pairs = []

        # Try matching by name first
        sim_by_name = {f.stem: f for f in sim_files}
        matched_sim = set()

        for real_f in real_files:
            if real_f.stem in sim_by_name:
                pairs.append((real_f, sim_by_name[real_f.stem]))
                matched_sim.add(real_f.stem)

        # Fall back to index matching for unmatched files
        unmatched_real = [f for f in real_files if f.stem not in sim_by_name]
        unmatched_sim = [f for f in sim_files if f.stem not in matched_sim]

        for real_f, sim_f in zip(unmatched_real, unmatched_sim):
            pairs.append((real_f, sim_f))

        return pairs


class ValidationVisualizer:
    """Generate validation visualizations."""

    def __init__(self, output_dir: str = "validation_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, real_grid: np.ndarray, sim_grid: np.ndarray,
                     result: ValidationResult, prefix: str = "") -> Dict[str, str]:
        """Generate all visualizations and return file paths."""
        if not HAS_MATPLOTLIB:
            return {}

        paths = {}

        paths['comparison_ppi'] = self._plot_comparison_ppi(
            real_grid, sim_grid, result.range_m, prefix)
        paths['difference_heatmap'] = self._plot_difference_heatmap(
            real_grid, sim_grid, prefix)
        paths['intensity_profiles'] = self._plot_intensity_profiles(
            real_grid, sim_grid, result.range_m, prefix)
        paths['dashboard'] = self._plot_dashboard(
            real_grid, sim_grid, result, prefix)

        return paths

    def _polar_meshgrid(self, n_bearings: int, n_range: int):
        theta = np.linspace(0, 2 * np.pi, n_bearings + 1)
        r = np.linspace(0, 1, n_range + 1)
        return np.meshgrid(theta, r)

    def _plot_ppi(self, ax, grid: np.ndarray, title: str, range_m: float):
        theta, r = self._polar_meshgrid(*grid.shape)
        c = ax.pcolormesh(theta, r, grid.T, cmap='viridis', vmin=0, vmax=1, shading='flat')
        for frac in [0.25, 0.5, 0.75, 1.0]:
            circle = plt.Circle((0, 0), frac, fill=False, color='white', linewidth=0.3, alpha=0.5)
            ax.add_patch(circle)
        ax.set_title(title, fontsize=9)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        return c

    def _plot_comparison_ppi(self, real_grid, sim_grid, range_m, prefix) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 5))
        self._plot_ppi(ax1, real_grid, 'Real (Reference)', range_m)
        c = self._plot_ppi(ax2, sim_grid, 'Simulator', range_m)
        fig.colorbar(c, ax=[ax1, ax2], shrink=0.6, label='Intensity')
        fig.tight_layout()
        path = str(self.output_dir / f"{prefix}comparison_ppi.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

    def _plot_difference_heatmap(self, real_grid, sim_grid, prefix) -> str:
        diff = real_grid - sim_grid
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 6))
        theta, r = self._polar_meshgrid(*diff.shape)
        vmax = max(abs(diff.min()), abs(diff.max()), 0.01)
        c = ax.pcolormesh(theta, r, diff.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='flat')
        ax.set_title('Difference (Real - Sim)', fontsize=10)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        fig.colorbar(c, ax=ax, shrink=0.7, label='Intensity difference')
        fig.tight_layout()
        path = str(self.output_dir / f"{prefix}difference_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

    def _plot_intensity_profiles(self, real_grid, sim_grid, range_m, prefix,
                                  bearings=(0, 90, 180, 270)) -> str:
        n_range = real_grid.shape[1]
        ranges = np.linspace(0, range_m, n_range)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, b in zip(axes.flat, bearings):
            b_idx = int(b) % 360
            ax.plot(ranges, real_grid[b_idx], 'b-', linewidth=1, label='Real')
            ax.plot(ranges, sim_grid[b_idx], 'r--', linewidth=1, label='Sim')
            ax.set_title(f'Bearing {b}°', fontsize=9)
            ax.set_xlabel('Range (m)', fontsize=8)
            ax.set_ylabel('Intensity', fontsize=8)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = str(self.output_dir / f"{prefix}intensity_profiles.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

    def _plot_dashboard(self, real_grid, sim_grid, result: ValidationResult, prefix) -> str:
        fig = plt.figure(figsize=(16, 10))

        # Row 1: PPI real, PPI sim, difference
        ax1 = fig.add_subplot(2, 3, 1, projection='polar')
        self._plot_ppi(ax1, real_grid, 'Real (Reference)', result.range_m)

        ax2 = fig.add_subplot(2, 3, 2, projection='polar')
        self._plot_ppi(ax2, sim_grid, 'Simulator', result.range_m)

        ax3 = fig.add_subplot(2, 3, 3, projection='polar')
        diff = real_grid - sim_grid
        theta, r = self._polar_meshgrid(*diff.shape)
        vmax = max(abs(diff.min()), abs(diff.max()), 0.01)
        ax3.pcolormesh(theta, r, diff.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='flat')
        ax3.set_title('Difference', fontsize=9)
        ax3.set_yticklabels([])
        ax3.set_xticklabels([])

        # Row 2: RMSE by bearing, intensity profile, metrics text
        ax4 = fig.add_subplot(2, 3, 4)
        if result.per_bearing_rmse is not None:
            ax4.plot(range(360), result.per_bearing_rmse, 'b-', linewidth=0.5)
            ax4.set_xlabel('Bearing (°)')
            ax4.set_ylabel('RMSE')
            ax4.set_title('RMSE by Bearing')
            ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(2, 3, 5)
        ranges = np.linspace(0, result.range_m, NUM_RANGE_BINS)
        ax5.plot(ranges, real_grid[0], 'b-', label='Real')
        ax5.plot(ranges, sim_grid[0], 'r--', label='Sim')
        ax5.set_title('Profile @ 0°', fontsize=9)
        ax5.set_xlabel('Range (m)')
        ax5.set_ylabel('Intensity')
        ax5.legend(fontsize=7)
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        rpb = result.range_per_bin_m
        text = (
            f"Overall RMSE: {result.overall_rmse:.4f}\n"
            f"Pearson r: {result.pearson_r:.4f}\n"
        )
        if result.ssim_value is not None:
            text += f"SSIM: {result.ssim_value:.4f}\n"
        text += (
            f"\nCoastline Error:\n"
            f"  Mean: {result.coastline_mean_error_bins:.1f} bins ({result.coastline_mean_error_bins * rpb:.1f} m)\n"
            f"  RMS:  {result.coastline_rms_error_bins:.1f} bins ({result.coastline_rms_error_bins * rpb:.1f} m)\n"
            f"\nTargets:\n"
            f"  Real: {result.real_target_count}, Sim: {result.sim_target_count}\n"
            f"  Matched: {len(result.target_matches)}\n"
            f"  Missed: {result.missed_targets}, False alarms: {result.false_alarm_targets}\n"
            f"\nCoverage: Real {result.coverage_real:.1%} / Sim {result.coverage_sim:.1%}"
        )
        ax6.text(0.1, 0.9, text, transform=ax6.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax6.set_title('Metrics Summary', fontsize=9)

        fig.suptitle('Radar Validation Dashboard', fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = str(self.output_dir / f"{prefix}dashboard.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path


def write_summary_report(result: ValidationResult, path: str) -> None:
    """Write human-readable metrics summary."""
    rpb = result.range_per_bin_m
    with open(path, 'w') as f:
        f.write("=== Radar Validation Report ===\n\n")
        f.write(f"Timestamp: {result.timestamp}\n")
        f.write(f"Real source: {result.source_real}\n")
        f.write(f"Sim source: {result.source_sim}\n")
        f.write(f"Range: {result.range_m:.0f} m ({result.range_m/1852:.2f} nm)\n\n")

        f.write("--- Core Metrics ---\n")
        f.write(f"Overall RMSE:        {result.overall_rmse:.4f}\n")
        f.write(f"Pearson Correlation: {result.pearson_r:.4f}\n")
        if result.ssim_value is not None:
            f.write(f"SSIM:                {result.ssim_value:.4f}\n")

        f.write(f"\nCoverage (real):     {result.coverage_real:.1%}\n")
        f.write(f"Coverage (sim):      {result.coverage_sim:.1%}\n")

        f.write(f"\n--- Coastline Boundary Error ---\n")
        f.write(f"Mean:  {result.coastline_mean_error_bins:.1f} bins ({result.coastline_mean_error_bins * rpb:.1f} m)\n")
        f.write(f"Max:   {result.coastline_max_error_bins:.1f} bins ({result.coastline_max_error_bins * rpb:.1f} m)\n")
        f.write(f"RMS:   {result.coastline_rms_error_bins:.1f} bins ({result.coastline_rms_error_bins * rpb:.1f} m)\n")

        f.write(f"\n--- Target Matching ---\n")
        f.write(f"Real targets:  {result.real_target_count}\n")
        f.write(f"Sim targets:   {result.sim_target_count}\n")
        f.write(f"Matched:       {len(result.target_matches)}\n")
        f.write(f"Missed:        {result.missed_targets}\n")
        f.write(f"False alarms:  {result.false_alarm_targets}\n")

        if result.target_matches:
            ro = [m.range_offset_bins for m in result.target_matches]
            bo = [m.bearing_offset_deg for m in result.target_matches]
            f.write(f"Mean range offset:   {np.mean(ro):.1f} bins ({np.mean(ro) * rpb:.1f} m)\n")
            f.write(f"Mean bearing offset: {np.mean(bo):.1f} deg\n")


def write_batch_report(result: BatchResult, path: str) -> None:
    """Write batch comparison summary."""
    with open(path, 'w') as f:
        f.write("=== Batch Validation Report ===\n\n")
        f.write(f"Files processed: {result.files_processed}\n")
        f.write(f"Files failed: {result.files_failed}\n\n")

        f.write("--- Aggregate Metrics ---\n")
        f.write(f"Mean RMSE:    {result.mean_rmse:.4f}\n")
        f.write(f"Mean Pearson: {result.mean_pearson:.4f}\n")
        if result.mean_ssim is not None:
            f.write(f"Mean SSIM:    {result.mean_ssim:.4f}\n")

        f.write(f"\n--- Target Statistics ---\n")
        f.write(f"Total matched:      {result.total_matched_targets}\n")
        f.write(f"Total missed:       {result.total_missed_targets}\n")
        f.write(f"Total false alarms: {result.total_false_alarms}\n")

        f.write(f"\n--- Individual Results ---\n")
        for i, r in enumerate(result.individual_results):
            f.write(f"\n[{i+1}] {Path(r.source_real).name}\n")
            f.write(f"    RMSE: {r.overall_rmse:.4f}, Pearson: {r.pearson_r:.4f}\n")
            f.write(f"    Targets: {len(r.target_matches)} matched, {r.missed_targets} missed\n")
