"""Annotation export in JSON, COCO, and YOLO formats."""
import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


@dataclass
class TargetAnnotation:
    """Annotation for a single radar target."""
    vessel_type: str
    range_m: float
    bearing_deg: float
    rcs: float
    length: float
    beam: float
    height: float
    occluded: bool
    bbox: List[float] = field(default_factory=list)  # [x_center, y_center, width, height] in pixels
    vessel_id: str = ""
    x: float = 0.0
    y: float = 0.0
    course: float = 0.0
    speed: float = 0.0


# Class mapping for YOLO format
VESSEL_CLASS_MAP = {
    'buoy': 0,
    'fishing': 1,
    'sailing': 2,
    'pilot': 3,
    'tug': 4,
    'cargo': 5,
    'tanker': 6,
    'passenger': 7,
    'unknown': 8,
    'own_ship': 9,
}


class AnnotationExporter:
    """Export radar target annotations in multiple formats."""

    def __init__(self, image_size: int = 512, range_scale_m: float = 11112.0):
        self.image_size = image_size
        self.range_scale_m = range_scale_m

    def annotations_from_dicts(self, annotation_dicts: List[dict]) -> List[TargetAnnotation]:
        """Convert annotation dicts (from Simulation.collect_annotations) to TargetAnnotation objects."""
        annotations = []
        half = self.image_size / 2.0
        radius = half  # PPI fills the image

        for d in annotation_dicts:
            range_m = d.get('range_m', 0)
            bearing_deg = d.get('bearing_deg', 0)

            # Skip targets outside display range
            if range_m > self.range_scale_m:
                continue

            # Convert polar to pixel coordinates (PPI image space)
            range_ratio = range_m / self.range_scale_m
            bearing_rad = math.radians(bearing_deg - 90)
            px = half + range_ratio * radius * math.cos(bearing_rad)
            py = half + range_ratio * radius * math.sin(bearing_rad)

            # Estimate bbox size from vessel dimensions and range
            # At closer range, targets appear larger on PPI
            angular_extent = math.degrees(
                math.atan2(d.get('length', 50) / 2.0, max(100, range_m)))
            bbox_w = max(4, angular_extent / 360.0 * self.image_size * math.pi)
            bbox_h = max(4, bbox_w * 0.8)  # Slightly shorter than wide

            ann = TargetAnnotation(
                vessel_type=d.get('vessel_type', 'unknown'),
                range_m=range_m,
                bearing_deg=bearing_deg,
                rcs=d.get('rcs', 0),
                length=d.get('length', 0),
                beam=d.get('beam', 0),
                height=d.get('height', 0),
                occluded=d.get('occluded', False),
                bbox=[px, py, bbox_w, bbox_h],
                vessel_id=d.get('vessel_id', ''),
                x=d.get('x', 0),
                y=d.get('y', 0),
                course=d.get('course', 0),
                speed=d.get('speed', 0),
            )
            annotations.append(ann)

        return annotations

    def export_json(self, annotations: List[TargetAnnotation],
                    params: Optional[dict], filepath: str) -> str:
        """Export full metadata JSON.

        Args:
            annotations: List of target annotations.
            params: Radar parameters dict (or None).
            filepath: Output path.

        Returns:
            Path to saved file.
        """
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        data = {
            'image_size': self.image_size,
            'range_scale_m': self.range_scale_m,
            'radar_params': params or {},
            'annotations': [asdict(a) for a in annotations],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return filepath

    def export_coco(self, annotations: List[TargetAnnotation],
                    image_size: int, filepath: str,
                    image_id: int = 0, image_filename: str = "") -> str:
        """Export COCO format bounding boxes.

        Args:
            annotations: List of target annotations.
            image_size: Image dimension (square).
            filepath: Output path.
            image_id: Image ID for COCO format.
            image_filename: Image filename.

        Returns:
            Path to saved file.
        """
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        categories = [{'id': v, 'name': k} for k, v in VESSEL_CLASS_MAP.items()]
        images = [{
            'id': image_id,
            'file_name': image_filename,
            'width': image_size,
            'height': image_size,
        }]

        coco_anns = []
        for i, ann in enumerate(annotations):
            if ann.occluded:
                continue
            if len(ann.bbox) < 4:
                continue
            cx, cy, w, h = ann.bbox
            # COCO uses top-left x,y,w,h
            x1 = max(0, cx - w / 2)
            y1 = max(0, cy - h / 2)
            class_id = VESSEL_CLASS_MAP.get(ann.vessel_type, 8)
            coco_anns.append({
                'id': i,
                'image_id': image_id,
                'category_id': class_id,
                'bbox': [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                'area': round(w * h, 1),
                'iscrowd': 0,
            })

        coco = {
            'images': images,
            'annotations': coco_anns,
            'categories': categories,
        }
        with open(filepath, 'w') as f:
            json.dump(coco, f, indent=2)
        return filepath

    def export_yolo(self, annotations: List[TargetAnnotation],
                    image_size: int, filepath: str) -> str:
        """Export YOLO format (class cx cy w h), normalized 0-1.

        Args:
            annotations: List of target annotations.
            image_size: Image dimension (square).
            filepath: Output path.

        Returns:
            Path to saved file.
        """
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        lines = []
        for ann in annotations:
            if ann.occluded:
                continue
            if len(ann.bbox) < 4:
                continue
            cx, cy, w, h = ann.bbox
            # Normalize to 0-1
            ncx = cx / image_size
            ncy = cy / image_size
            nw = w / image_size
            nh = h / image_size
            # Clamp
            ncx = max(0.0, min(1.0, ncx))
            ncy = max(0.0, min(1.0, ncy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            class_id = VESSEL_CLASS_MAP.get(ann.vessel_type, 8)
            lines.append(f"{class_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
            if lines:
                f.write('\n')
        return filepath
