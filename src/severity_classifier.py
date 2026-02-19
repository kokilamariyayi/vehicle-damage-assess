"""
src/severity_classifier.py
Severity Classification using EfficientNet-B4 (fine-tuned).

Classifies each detected damage crop into:
  - minor    (surface-level, cosmetic)
  - moderate (structural damage, needs repair)
  - severe   (major structural damage, safety risk)

Architecture:
  EfficientNet-B4 pretrained on ImageNet
  → Global Average Pooling
  → Dropout(0.4)
  → Dense(256, ReLU)
  → Dropout(0.3)
  → Dense(3, Softmax)

Alternative: Vision Transformer (ViT-B/16) for higher accuracy
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Optional
from loguru import logger
import timm


class SeverityClassifier:
    """
    EfficientNet-B4 based severity classifier.
    Falls back to rule-based if no trained weights available.
    """

    CLASSES = ["minor", "moderate", "severe"]
    CLASS_DESCRIPTIONS = {
        "minor": "Surface-level damage. Cosmetic repair recommended.",
        "moderate": "Significant damage. Professional repair required.",
        "severe": "Major structural damage. Immediate repair or replacement needed.",
    }

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config["inference"]["device"])
        self.img_size = config["models"]["classifier_input_size"]
        self.model: Optional[nn.Module] = None

        weights_path = config["models"].get("classifier_weights")

        if weights_path:
            self._load_model(weights_path)
        else:
            logger.warning("No classifier weights — using rule-based severity estimation.")

        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self, weights_path: str):
        """Load fine-tuned EfficientNet-B4."""
        try:
            backbone = config["models"]["classifier"]
            self.model = timm.create_model(backbone, pretrained=False, num_classes=3)
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.eval().to(self.device)
            logger.success(f"Severity classifier loaded: {weights_path}")
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            self.model = None

    def classify(self, damage_crop: np.ndarray,
                  damage_type: str,
                  area_pct: float) -> Tuple[str, float, List[float]]:
        """
        Classify severity of a damage crop.

        Args:
            damage_crop: BGR numpy array of cropped damage region
            damage_type: Type of damage (dent, scratch, etc.)
            area_pct: Percentage of image area this damage covers

        Returns:
            (severity_label, confidence, [minor_prob, moderate_prob, severe_prob])
        """
        if self.model is not None:
            return self._nn_classify(damage_crop)
        else:
            return self._rule_classify(damage_crop, damage_type, area_pct)

    def _nn_classify(self, crop: np.ndarray) -> Tuple[str, float, List[float]]:
        """Neural network classification."""
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        cls_id = int(np.argmax(probs))
        return self.CLASSES[cls_id], float(probs[cls_id]), probs.tolist()

    def _rule_classify(self, crop: np.ndarray,
                        damage_type: str,
                        area_pct: float) -> Tuple[str, float, List[float]]:
        """
        Heuristic severity estimation based on:
        1. Damage area percentage
        2. Color deviation (rust/deep damage = more color change)
        3. Edge density (more edges = more structural damage)
        4. Damage type weight
        """
        score = 0.0

        # ── Area score (0-40 points) ─────────────────────────────────────────
        if area_pct < 2.0:
            score += 10
        elif area_pct < 8.0:
            score += 25
        else:
            score += 40

        # ── Edge density score (0-30 points) ─────────────────────────────────
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (crop.shape[0] * crop.shape[1] + 1e-6)
        score += min(30, edge_density * 200)

        # ── Color deviation score (0-20 points) ──────────────────────────────
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat_std = float(np.std(hsv[:, :, 1]))
        val_std = float(np.std(hsv[:, :, 2]))
        score += min(20, (sat_std + val_std) / 10)

        # ── Damage type multiplier ────────────────────────────────────────────
        type_weights = {
            "scratch": 0.7,
            "dent": 1.0,
            "rust": 1.1,
            "crack": 1.2,
            "deformation": 1.3,
            "missing_part": 1.4,
            "flat_tire": 1.3,
            "shattered_glass": 1.5,
        }
        score *= type_weights.get(damage_type, 1.0)

        # ── Map score to class ────────────────────────────────────────────────
        if score < 35:
            cls = "minor"
            probs = [0.75, 0.20, 0.05]
            conf = 0.75
        elif score < 65:
            cls = "moderate"
            probs = [0.15, 0.70, 0.15]
            conf = 0.70
        else:
            cls = "severe"
            probs = [0.05, 0.20, 0.75]
            conf = 0.75

        return cls, conf, probs

    def classify_batch(self, crops_and_info: list) -> list:
        """Classify multiple damage regions."""
        return [
            self.classify(crop, dtype, area)
            for crop, dtype, area in crops_and_info
        ]


# ── Model Architecture (for training) ────────────────────────────────────────
class EfficientNetSeverity(nn.Module):
    """
    EfficientNet-B4 fine-tuned for severity classification.
    Use this for training on CarDD / custom datasets.
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True,
                 dropout: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,          # Remove classification head
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class ViTSeverity(nn.Module):
    """
    Vision Transformer (ViT-B/16) for severity classification.
    Higher accuracy than EfficientNet but slower.
    Use for offline/batch processing.
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
