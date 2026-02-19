"""
models/train.py
Training scripts for:
1. YOLOv8 damage detector (fine-tuning)
2. EfficientNet-B4 severity classifier

Recommended datasets:
  - CarDD (Car Damage Detection): https://github.com/CarDD-Dataset/CarDD
  - Vehicle Damage Detection (Roboflow): https://universe.roboflow.com/car-damage-kwmkc/vehicle-damage-v2
  - Stanford Cars Dataset (for vehicle recognition)
  - COCO (vehicle parts)
  - CompCars (Comprehensive Cars)
"""

import os
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from loguru import logger
from ultralytics import YOLO


# ── 1. YOLOv8 Damage Detector Training ───────────────────────────────────────

def create_dataset_yaml(data_root: str, output_path: str = "configs/damage_dataset.yaml") -> str:
    """Create YOLO dataset config from directory structure."""
    dataset = {
        "path": str(Path(data_root).resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc": 8,
        "names": [
            "dent", "scratch", "crack", "shattered_glass",
            "flat_tire", "missing_part", "deformation", "rust"
        ]
    }
    with open(output_path, "w") as f:
        yaml.dump(dataset, f)
    logger.info(f"Dataset YAML: {output_path}")
    return output_path


def train_detector(
    data_yaml: str,
    model_size: str = "yolov8m",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
):
    """
    Fine-tune YOLOv8 for vehicle damage detection.

    Advanced training features:
    - Mosaic + MixUp augmentation
    - Cosine LR schedule
    - Label smoothing
    - Multi-scale training
    """
    model = YOLO(f"{model_size}.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project="runs/damage_detector",
        name="exp",

        # Augmentation
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,

        # Training
        optimizer="AdamW",
        lr0=1e-3, lrf=0.01,
        cos_lr=True,
        label_smoothing=0.1,
        patience=20,
        save=True,
        plots=True,
        val=True,
    )

    best = "runs/damage_detector/exp/weights/best.pt"
    if os.path.exists(best):
        logger.success(f"Detector trained! Best weights: {best}")
        # Export ONNX
        YOLO(best).export(format="onnx", imgsz=imgsz)
    return results


# ── 2. EfficientNet Severity Classifier Training ──────────────────────────────

class SeverityDataset(Dataset):
    """
    Dataset for severity classification.
    Directory structure:
        data/severity/
            train/
                minor/   ← damage crop images
                moderate/
                severe/
            val/
                minor/
                moderate/
                severe/
    """
    CLASSES = ["minor", "moderate", "severe"]

    def __init__(self, root: str, split: str = "train", img_size: int = 380):
        self.samples = []
        self.transform = self._get_transform(split, img_size)

        for cls_id, cls_name in enumerate(self.CLASSES):
            cls_dir = Path(root) / split / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*.[jp][pn]*"):
                self.samples.append((str(img_path), cls_id))

        logger.info(f"SeverityDataset [{split}]: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label

    def _get_transform(self, split: str, img_size: int):
        if split == "train":
            return T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(p=0.1),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                T.RandomRotation(15),
                T.RandomGrayscale(p=0.05),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            return T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])


def train_severity_classifier(
    data_dir: str = "data/severity/",
    epochs: int = 50,
    batch: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
    backbone: str = "efficientnet_b4",
):
    """Train EfficientNet-B4 severity classifier with advanced training."""
    import timm
    from src.severity_classifier import EfficientNetSeverity

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    train_ds = SeverityDataset(data_dir, "train")
    val_ds   = SeverityDataset(data_dir, "val")

    # Class weights for imbalanced data
    class_counts = [sum(1 for _, c in train_ds.samples if c == i) for i in range(3)]
    total = sum(class_counts)
    weights = torch.tensor([total / (3 * c + 1) for c in class_counts]).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch, num_workers=4)

    model = EfficientNetSeverity(num_classes=3, pretrained=True).to(device)

    # Optimizer: different LR for backbone vs head
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr * 0.1},
        {"params": model.classifier.parameters(), "lr": lr},
    ], weight_decay=1e-4)

    # Cosine annealing + warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_acc = 0.0
    os.makedirs("runs/severity_classifier", exist_ok=True)

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        class_correct = [0] * 3
        class_total   = [0] * 3
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += len(labels)
                for p, l in zip(preds, labels):
                    class_correct[l] += (p == l).item()
                    class_total[l] += 1

        acc = correct / (total + 1e-6)
        per_class = [f"{SeverityDataset.CLASSES[i]}: {class_correct[i]/(class_total[i]+1e-6):.2f}"
                     for i in range(3)]
        logger.info(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | "
                   f"Val Acc: {acc:.3f} | {' | '.join(per_class)}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "runs/severity_classifier/best.pt")
            logger.success(f"  New best: {best_acc:.3f}")

    logger.success(f"Training complete. Best accuracy: {best_acc:.3f}")
    logger.info("Update config.yaml: classifier_weights: runs/severity_classifier/best.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["detector", "classifier"], default="detector")
    parser.add_argument("--data", default="data/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.task == "detector":
        yaml_path = create_dataset_yaml(args.data)
        train_detector(yaml_path, epochs=args.epochs, batch=args.batch, device=args.device)
    else:
        train_severity_classifier(args.data, epochs=args.epochs,
                                  batch=args.batch, device=args.device)
