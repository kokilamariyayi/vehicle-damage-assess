"""
utils/visualizer.py
Annotates vehicle images with damage detections, severity, and cost info.
"""

import cv2
import numpy as np
from typing import List, Tuple
from src.damage_detector import DamageDetection
from src.cost_estimator import TotalEstimate


SEVERITY_COLORS = {
    "minor":    (0, 200, 0),      # Green
    "moderate": (0, 140, 255),    # Orange
    "severe":   (0, 0, 220),      # Red
}

DAMAGE_COLORS = {
    "dent":           (255, 100, 0),
    "scratch":        (0, 200, 255),
    "crack":          (0, 0, 255),
    "shattered_glass":(180, 0, 255),
    "flat_tire":      (0, 100, 255),
    "missing_part":   (255, 0, 100),
    "deformation":    (0, 50, 255),
    "rust":           (0, 100, 200),
}


class DamageVisualizer:

    def annotate(
        self,
        image: np.ndarray,
        detections: List[DamageDetection],
        severities: List[str],
        estimate: TotalEstimate,
    ) -> np.ndarray:
        """Draw all damage annotations on image."""
        out = image.copy()

        for i, (det, sev) in enumerate(zip(detections, severities)):
            color = SEVERITY_COLORS.get(sev, (128, 128, 128))
            x1, y1, x2, y2 = det.bbox.astype(int)

            # Box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Damage number circle
            cv2.circle(out, (x1+12, y1-12), 12, color, -1)
            cv2.putText(out, str(i+1), (x1+7, y1-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            # Label
            label = f"{det.damage_type} | {sev}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, y1-28), (x1+lw+6, y1), color, -1)
            cv2.putText(out, label, (x1+3, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            # Part label below box
            cv2.putText(out, det.vehicle_part.replace("_"," "),
                       (x1, y2+16), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                       color, 1)

        # Summary panel
        self._draw_summary(out, detections, severities, estimate)
        return out

    def _draw_summary(self, frame, detections, severities, estimate):
        h, w = frame.shape[:2]
        panel_w = 260
        panel_h = min(h, 200)
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-panel_w, 0), (w, panel_h), (20,20,20), -1)
        cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)

        sev = estimate.overall_severity
        sev_color = SEVERITY_COLORS.get(sev, (128,128,128))

        texts = [
            ("DAMAGE SUMMARY", (255,255,255), 0.5, True),
            (f"Overall: {sev.upper()}", sev_color, 0.5, True),
            (f"Damages found: {len(detections)}", (200,200,200), 0.42, False),
            (f"Estimate: ${estimate.grand_total_min:.0f}-${estimate.grand_total_max:.0f}",
             (100,255,100), 0.42, False),
        ]
        y = 22
        for text, color, scale, bold in texts:
            thickness = 2 if bold else 1
            cv2.putText(frame, text, (w-panel_w+8, y),
                       cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
            y += 28

        # Severity bar
        y += 5
        for i, (label, color) in enumerate([("Minor",(0,200,0)),
                                             ("Moderate",(0,140,255)),
                                             ("Severe",(0,0,220))]):
            count = sum(1 for s in severities if s == label.lower())
            cv2.putText(frame, f"{label}: {count}", (w-panel_w+8, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
            y += 18

    def draw_severity_heatmap(self, image: np.ndarray,
                               detections: List[DamageDetection],
                               severities: List[str]) -> np.ndarray:
        """Generate a heatmap overlay showing damage intensity."""
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        sev_weights = {"minor": 0.3, "moderate": 0.65, "severe": 1.0}

        for det, sev in zip(detections, severities):
            x1, y1, x2, y2 = det.bbox.astype(int)
            weight = sev_weights.get(sev, 0.5)
            heatmap[y1:y2, x1:x2] += weight

        # Normalize and colorize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Gaussian blur for smooth heatmap
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        blended = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
        return blended
