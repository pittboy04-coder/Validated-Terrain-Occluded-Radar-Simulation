"""CSV data import for replaying real Furuno radar captures."""
import csv
import os
import glob
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SweepRecord:
    bearing_deg: float
    echoes: List[float]


class CsvRotation:
    def __init__(self, filepath: str):
        self.sweeps: List[SweepRecord] = []
        self._load(filepath)

    def _load(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return
            for row in reader:
                if len(row) < 6:
                    continue
                angle_ticks = int(row[4])
                bearing_deg = (angle_ticks / 8192.0) * 360.0
                echoes = [min(1.0, int(v) / 252.0) for v in row[5:]]
                self.sweeps.append(SweepRecord(bearing_deg=bearing_deg, echoes=echoes))


class CsvPlayer:
    def __init__(self, folder: str):
        self.folder = folder
        self.active = True
        self._rotations: List[str] = []
        self._current_rotation_idx = 0
        self._current_sweep_idx = 0
        self._current_rotation: Optional[CsvRotation] = None
        self._sweeps_per_frame = 10

        self._rotations = sorted(glob.glob(os.path.join(folder, '*.csv')))
        if not self._rotations:
            self.active = False
            return
        self._load_rotation(0)

    def _load_rotation(self, idx: int) -> None:
        if 0 <= idx < len(self._rotations):
            self._current_rotation = CsvRotation(self._rotations[idx])
            self._current_rotation_idx = idx
            self._current_sweep_idx = 0
        else:
            self.active = False

    def get_next_sweeps(self) -> List[Tuple[float, List[float]]]:
        if not self.active or self._current_rotation is None:
            return []
        result = []
        for _ in range(self._sweeps_per_frame):
            rot = self._current_rotation
            if self._current_sweep_idx >= len(rot.sweeps):
                next_idx = self._current_rotation_idx + 1
                if next_idx >= len(self._rotations):
                    next_idx = 0
                self._load_rotation(next_idx)
                if not self.active:
                    break
                rot = self._current_rotation
            sweep = rot.sweeps[self._current_sweep_idx]
            result.append((sweep.bearing_deg, sweep.echoes))
            self._current_sweep_idx += 1
        return result

    def stop(self) -> None:
        self.active = False
