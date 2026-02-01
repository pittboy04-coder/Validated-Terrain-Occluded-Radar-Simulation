# Future Roadmap — Radar Object Detection & Geometry Pipeline

## 1. Drawable Landmasses on PPI/Scene View

**Goal:** Interactive polygon drawing for custom coastline/landmass shapes.

**Changes needed:**
- `scene_view.py` — Add `drawing_mode = "coastline"` state. Mouse clicks accumulate `CoastlinePoint` objects, double-click/right-click closes the polygon and passes it to `Simulation.add_coastline()`. Render in-progress polygon with dashed lines.
- `coastline.py` — Add `simplify_polygon()` utility (Ramer-Douglas-Peucker) to reduce vertex count on hand-drawn shapes.
- `control_panel.py` — Add "DRAW LAND", "UNDO POINT", and "FINISH" buttons in terrain panel.

**Complexity:** Low. Polygon infrastructure already exists.

---

## 2. Geometric Shape Classification and Labeling

**Goal:** Classify detected radar blobs by geometric shape (point target, vessel, land, rain, clutter).

**New file: `radar_sim/detection/classifier.py`**

**Feature extraction per blob (extend `detect_blobs`):**
- **Eccentricity** — major/minor axis ratio from second-order moments. Buoys ≈ 1.0, coastline arcs >> 1.0
- **Radial extent** — range bins spanned. Pulse-limited targets = 1-2 bins; extended targets = many
- **Angular extent** — bearing degrees. Land = 10-90°; vessels = 1-3°
- **Intensity profile** — peak-to-edge gradient. Metal hulls = steep; rain cells = soft
- **Range position** — close-range high-eccentricity = sea clutter; mid-range compact = vessel

**Classification rules (threshold-based, no ML initially):**

| Shape Class       | Eccentricity | Radial Extent | Angular Extent | Intensity        |
|-------------------|-------------|---------------|----------------|------------------|
| Point (buoy)      | ~1.0        | 1-2 bins      | <2°            | Variable         |
| Small vessel      | ~1.2        | 2-4 bins      | 1-3°           | Moderate-high    |
| Large vessel      | >1.5        | 4-10 bins     | 2-5°           | High, stable     |
| Land/coastline    | >>2.0       | Many bins     | >10°           | High, uniform    |
| Rain cell         | ~1.0        | Many bins     | >20°           | Low, soft edges  |
| Clutter spike     | ~1.0        | 1 bin         | <1°            | High, isolated   |

**Later stage:** Replace threshold rules with a trained classifier (random forest or small MLP) using synthetic ground-truth data.

---

## 3. Real Capture CSV → Geometry → Labels → Annotated Image

**Goal:** Load real Furuno CSV, detect all objects, classify them, output annotated image + labels.

### Pipeline

```
Real CSV → load_furuno_csv()
    → 360×N PPI array
    → CFAR adaptive thresholding
    → detect_blobs() with extended features
    → classify_blobs() → shape labels
    → convert to TargetAnnotation with bboxes
    → render PPI image + overlay bounding boxes + class labels
    → export annotated image (PNG) + YOLO/COCO labels
```

### New Files

**`radar_sim/detection/__init__.py`** — Package init.

**`radar_sim/detection/cfar.py`** — CA-CFAR (Constant False Alarm Rate) adaptive threshold:
- For each range-bearing cell, compute threshold from surrounding reference window mean × threshold factor.
- Window: ~16 range bins × 5 bearing cells, 2-bin guard band.
- Replaces fixed threshold in `detect_blobs`, reduces false alarms from clutter while preserving real targets.

**`radar_sim/detection/auto_detector.py`** — End-to-end detector:
```python
class AutoDetector:
    def detect(self, ppi_array) -> List[DetectedObject]:
        # 1. Estimate noise floor
        # 2. CFAR adaptive threshold
        # 3. Connected component detection
        # 4. Feature extraction per blob
        # 5. Classification
        # 6. Convert to DetectedObject with bbox, class, confidence

class DetectedObject:
    class_name: str          # "buoy", "vessel_small", "vessel_large", "land", "rain", "clutter"
    confidence: float        # 0-1
    bearing_deg: float
    range_m: float
    bbox: List[float]        # [cx, cy, w, h] in pixel space
    area_bins: int
    peak_intensity: float
    eccentricity: float
```

**`radar_sim/detection/image_renderer.py`** — PPI to annotated PNG:
- Polar → Cartesian image (green phosphor or grayscale colormap)
- Draw bounding boxes with per-class colors
- Draw class labels + confidence text
- Save as PNG (Pillow) or PPM (no dependency)

**`radar_sim/detection/pipeline.py`** — Full CSV-to-output pipeline:
```python
class DetectionPipeline:
    def process_csv(self, csv_path, output_dir):
        ppi = load_furuno_csv(csv_path)
        detections = self.detector.detect(ppi)
        annotations = self._to_annotations(detections)
        render_annotated_ppi(ppi, detections, output_path="annotated.png")
        exporter.export_yolo(annotations, image_size, "labels.txt")
        exporter.export_coco(annotations, image_size, "coco.json")
```

**`detect_from_csv.py`** — CLI entry point:
```
python detect_from_csv.py -i capture.csv -o output/
```

---

## 4. Accuracy Improvements for Automatic Detection

### CFAR Adaptive Thresholding (Critical)
Fixed thresholds fail on real data where noise/clutter varies with range and sea state. CA-CFAR computes a local threshold per cell — the standard approach in maritime radar signal processing.

### Multi-Scan Integration
Single sweeps are noisy. Buffer last N PPI arrays (N=3-5). Targets appearing in 3/5 consecutive rotations are real; single-scan spikes are clutter. Score: `persistence = appearances / N`, threshold at ~0.6.

### Training Pipeline (Synthetic → Real)
1. `generate_training_data.py -n 5000` → 5000 scenarios with ground-truth YOLO labels
2. Run `AutoDetector` on synthetic CSVs → detected blobs
3. Match detected blobs to ground-truth annotations (Hungarian matching by range/bearing)
4. Build feature matrix: [eccentricity, radial_extent, angular_extent, peak_intensity, range, ...] → class label
5. Train random forest or gradient-boosted classifier (scikit-learn)
6. Save model weights → load in `AutoDetector` for real-data inference

This is where synthetic data bridges to real-world detection — the model learns what each target type looks like using physically-modeled synthetic ground truth.

---

## 5. File Summary

### New Files to Create

| File | Purpose | Priority |
|------|---------|----------|
| `radar_sim/detection/__init__.py` | Package | — |
| `radar_sim/detection/cfar.py` | CA-CFAR adaptive threshold | High |
| `radar_sim/detection/classifier.py` | Shape feature extraction + classification | High |
| `radar_sim/detection/auto_detector.py` | End-to-end blob → labeled detection | High |
| `radar_sim/detection/image_renderer.py` | PPI → annotated PNG | Medium |
| `radar_sim/detection/pipeline.py` | CSV → detect → annotate → export | Medium |
| `detect_from_csv.py` | CLI entry point | Medium |

### Existing Files to Modify

| File | Change |
|------|--------|
| `geometry_extractor.py` | Add moment-based features (eccentricity, extent) to `detect_blobs` |
| `annotation.py` | Accept `DetectedObject` inputs (not just simulation dicts) |
| `control_panel.py` | "DRAW LAND" button, "DETECT" button for auto-detection |
| `scene_view.py` | Polygon drawing mode for coastlines |

### Dependencies

- **No new required dependencies** for threshold-based classification
- **Optional:** `scikit-learn` for trained classifier, `Pillow` for PNG export
- Everything else uses numpy (already required) and stdlib

### Critical Path for Accuracy

**CFAR → feature extraction → synthetic training → classifier**

Drawable landmasses and image rendering are independent UI features that can be built in parallel.
