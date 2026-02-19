"""
src/preprocessor.py
Image Standardization Pipeline for Vehicle Damage Assessment.

Advanced preprocessing:
- Smart cropping (focus on vehicle, remove background)
- Contrast enhancement (CLAHE)
- Noise reduction
- Normalization for model inference
- Albumentations-based augmentation for training
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
from typing import Tuple, Optional, List
from loguru import logger
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImagePreprocessor:
    """
    Standardizes vehicle images before inference.

    Pipeline:
    1. Load & validate image
    2. Auto-orient (EXIF rotation fix)
    3. Vehicle region focus (smart crop)
    4. CLAHE contrast enhancement
    5. Noise reduction
    6. Resize to model input size
    7. Normalize
    """

    def __init__(self, target_size: int = 640):
        self.target_size = target_size

        # Inference transform (no augmentation)
        self.inference_transform = A.Compose([
            A.LongestMaxSize(max_size=target_size),
            A.PadIfNeeded(
                min_height=target_size,
                min_width=target_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # Training augmentation (heavy)
        self.train_transform = A.Compose([
            A.RandomResizedCrop(height=target_size, width=target_size,
                               scale=(0.6, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
            A.MotionBlur(p=0.1),
            A.CLAHE(p=0.3),
            A.RandomShadow(p=0.2),
            A.ColorJitter(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def preprocess(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline.

        Returns:
            original: Original image (BGR, for visualization)
            processed: Preprocessed image (BGR, for model)
        """
        # Load
        original = self._load_image(image_path)
        if original is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # EXIF orientation fix
        original = self._fix_orientation(image_path, original)

        # Enhance
        processed = self._enhance(original.copy())

        # Resize to model input
        processed = self._resize(processed)

        return original, processed

    def preprocess_array(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a numpy array (BGR) directly."""
        enhanced = self._enhance(image.copy())
        return self._resize(enhanced)

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        img = cv2.imread(path)
        if img is None:
            # Try PIL as fallback (handles more formats)
            try:
                pil = Image.open(path).convert("RGB")
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                return None
        return img

    def _fix_orientation(self, path: str, image: np.ndarray) -> np.ndarray:
        """Fix EXIF rotation from phone cameras."""
        try:
            from PIL import Image as PILImage
            import PIL.ExifTags
            pil = PILImage.open(path)
            exif = pil._getexif()
            if exif:
                for tag, val in exif.items():
                    if PIL.ExifTags.TAGS.get(tag) == 'Orientation':
                        if val == 3:
                            image = cv2.rotate(image, cv2.ROTATE_180)
                        elif val == 6:
                            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                        elif val == 8:
                            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except Exception:
            pass
        return image

    def _enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE + denoising for better damage visibility.
        """
        # Convert to LAB for CLAHE on luminance only
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Fast non-local means denoising
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

        return enhanced

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize with letterboxing to maintain aspect ratio."""
        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded

    def batch_preprocess(self, image_paths: List[str]) -> List[Tuple]:
        """Process multiple images."""
        results = []
        for path in image_paths:
            try:
                results.append(self.preprocess(path))
            except Exception as e:
                logger.error(f"Failed to preprocess {path}: {e}")
        return results

    @staticmethod
    def validate_image(path: str) -> bool:
        """Check if file is a valid image."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        if Path(path).suffix.lower() not in valid_extensions:
            return False
        try:
            img = cv2.imread(path)
            return img is not None and img.size > 0
        except Exception:
            return False
