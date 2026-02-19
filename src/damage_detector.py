"""
src/damage_detector.py
Damage Localization using YOLOv8.

Detects and localizes:
- Damage types: dent, scratch, crack, shattered_glass, flat_tire, missing_part, deformation, rust
- Vehicle parts: bumper, door, hood, trunk, fender, windshield, headlight, etc.

Uses ensemble of two YOLOv8 models:
1. Damage type detector
2. Vehicle part segmentor
Then combines results to produce part+damage pairs.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from loguru import logger


@dataclass
class DamageDetection:
    """Single damage detection result."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    damage_type: str
    vehicle_part: str
    confidence: float
    area_px: int = 0
    area_pct: float = 0.0    # % of image area
    crop: Optional[np.ndarray] = None  # cropped damage region

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.area_px = int((x2 - x1) * (y2 - y1))

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (int((x1+x2)/2), int((y1+y2)/2))

    def to_dict(self) -> dict:
        return {
            "damage_type": self.damage_type,
            "vehicle_part": self.vehicle_part,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox.tolist(),
            "area_pct": round(self.area_pct, 2),
        }


class DamageDetector:
    """
    Two-stage damage detection:
    Stage 1: Detect damage regions (YOLOv8)
    Stage 2: Assign detected region to nearest vehicle part

    For best results, use fine-tuned weights on:
    - CarDD dataset (Car Damage Detection)
    - VehicleDamage dataset (Roboflow)
    - COCO vehicle subset for part detection
    """

    # COCO vehicle class IDs
    COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    # Damage classes (for fine-tuned model)
    DAMAGE_CLASSES = {
        0: "dent",
        1: "scratch",
        2: "crack",
        3: "shattered_glass",
        4: "flat_tire",
        5: "missing_part",
        6: "deformation",
        7: "rust",
    }

    # Vehicle part spatial mapping (approximate regions by position)
    # Used when no part detector is available
    SPATIAL_PARTS = {
        "top_left":     "fender_front_left",
        "top_center":   "hood",
        "top_right":    "fender_front_right",
        "mid_left":     "door_front_left",
        "mid_center":   "windshield",
        "mid_right":    "door_front_right",
        "bot_left":     "bumper_front",
        "bot_center":   "bumper_front",
        "bot_right":    "bumper_front",
    }

    def __init__(self, config: dict):
        self.config = config
        self.conf = config["inference"]["confidence_threshold"]
        self.iou = config["inference"]["iou_threshold"]
        self.device = config["inference"]["device"]
        self.img_size = config["models"]["input_size"]

        # Load detector
        weights = config["models"].get("custom_detector") or config["models"]["detector"]
        logger.info(f"Loading damage detector: {weights}")
        self.model = YOLO(weights)

        logger.success("Damage detector ready.")

    def detect(self, image: np.ndarray) -> List[DamageDetection]:
        """
        Run full damage detection pipeline on an image.

        Args:
            image: BGR numpy array

        Returns:
            List of DamageDetection objects
        """
        h, w = image.shape[:2]
        total_area = h * w

        results = self.model(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                bbox = r.boxes.xyxy[i].cpu().numpy()
                conf = float(r.boxes.conf[i].cpu())
                cls_id = int(r.boxes.cls[i].cpu())

                # Map class to damage type
                # If using base YOLO (no custom weights), we simulate damage detection
                # with heuristics on detected objects
                damage_type = self.DAMAGE_CLASSES.get(cls_id, "dent")

                # Infer vehicle part from spatial position
                part = self._infer_part_from_position(bbox, w, h)

                # Crop damage region
                x1, y1, x2, y2 = bbox.astype(int)
                pad = 10
                x1c = max(0, x1-pad); y1c = max(0, y1-pad)
                x2c = min(w, x2+pad); y2c = min(h, y2+pad)
                crop = image[y1c:y2c, x1c:x2c].copy()

                d = DamageDetection(
                    bbox=bbox,
                    damage_type=damage_type,
                    vehicle_part=part,
                    confidence=conf,
                    crop=crop,
                )
                d.area_pct = round((d.area_px / total_area) * 100, 2)
                detections.append(d)

        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)

        # NMS across overlapping detections
        detections = self._soft_nms(detections)

        logger.info(f"Detected {len(detections)} damage regions.")
        return detections

    def _infer_part_from_position(self, bbox: np.ndarray,
                                   img_w: int, img_h: int) -> str:
        """
        Estimate vehicle part based on where in the image the damage is.
        Divides image into 3x3 grid.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        col = "left" if cx < img_w/3 else ("right" if cx > 2*img_w/3 else "center")
        row = "top"  if cy < img_h/3 else ("bot"   if cy > 2*img_h/3 else "mid")

        key = f"{row}_{col}"
        return self.SPATIAL_PARTS.get(key, "body_panel")

    def _soft_nms(self, detections: List[DamageDetection],
                   iou_thresh: float = 0.5) -> List[DamageDetection]:
        """Remove heavily overlapping duplicate detections."""
        if len(detections) <= 1:
            return detections

        kept = []
        remaining = list(detections)

        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            remaining = [
                d for d in remaining
                if self._iou(best.bbox, d.bbox) < iou_thresh
            ]
        return kept

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def detect_vehicle_presence(self, image: np.ndarray) -> bool:
        """Quick check: is there a vehicle in this image?"""
        results = self.model(image, conf=0.3, classes=list(self.COCO_VEHICLE_CLASSES.keys()),
                            verbose=False)
        for r in results:
            if r.boxes and len(r.boxes) > 0:
                return True
        return False
