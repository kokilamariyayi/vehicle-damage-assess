# 🚗 Vehicle Damage Assessment for Insurance Claims

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge)
![EfficientNet](https://img.shields.io/badge/EfficientNet--B4-timm-orange?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**AI system that automatically detects vehicle damage, classifies severity, and generates insurance claim reports with cost estimates.**

</div>

---

## 🎯 Problem Statement

Insurance companies process thousands of vehicle damage claims daily. Manual inspection is slow, inconsistent, and expensive. This system automates the entire process using computer vision and deep learning.

---

## ✨ Features

| Feature | Technology |
|---|---|
| 🔍 **Damage Localization** | YOLOv8 fine-tuned on CarDD dataset |
| 📊 **Severity Classification** | EfficientNet-B4 / ViT-B/16 |
| 🖼️ **Image Standardization** | CLAHE + denoising + smart crop |
| 💰 **Cost Estimation** | Part × Severity lookup tables + labor calc |
| 📄 **PDF Report Generation** | Professional reportlab PDF with annotated images |
| 🌐 **REST API** | FastAPI for integration with insurance systems |
| 🔥 **Severity Heatmap** | Visual heatmap showing damage intensity |

---

## 🧠 Architecture

```
Vehicle Photo
     │
     ▼
┌─────────────────┐
│  Preprocessor   │  ← CLAHE, EXIF fix, letterbox resize
└────────┬────────┘
         ▼
┌─────────────────┐
│ YOLOv8 Detector │  ← Damage type + location (8 classes)
└────────┬────────┘
         ▼
┌─────────────────┐
│EfficientNet-B4  │  ← Severity per damage crop (minor/moderate/severe)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Cost Estimator  │  ← Part × Damage × Severity lookup table
└────────┬────────┘
         ▼
┌─────────────────┐
│  PDF Reporter   │  ← Annotated image + cost breakdown + recommendation
└─────────────────┘
```

---

## 🔍 Damage Classes Detected

`dent` • `scratch` • `crack` • `shattered_glass` • `flat_tire` • `missing_part` • `deformation` • `rust`

## 📍 Vehicle Parts Localized

`bumper_front/rear` • `door_front/rear_left/right` • `hood` • `trunk` • `fender` • `windshield` • `headlight` • `taillight` • `roof`

---

## 📁 Project Structure

```
vehicle_damage/
├── main.py                    # CLI entry point
├── app.py                     # FastAPI web server
├── requirements.txt
├── configs/config.yaml        # All settings + cost tables
├── src/
│   ├── preprocessor.py        # Image standardization
│   ├── damage_detector.py     # YOLOv8 damage detection
│   ├── severity_classifier.py # EfficientNet-B4 severity
│   └── cost_estimator.py      # Repair cost calculation
├── reports/
│   └── report_generator.py    # PDF report generation
├── models/
│   └── train.py               # Training scripts
└── utils/
    └── visualizer.py          # Annotation + heatmap
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/vehicle-damage-assessment.git
cd vehicle-damage-assessment
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
# Assess single image
python main.py --image car_damage.jpg --report

# Batch process folder
python main.py --folder images/ --report

# Start API server
python main.py --serve

# CPU mode
python main.py --image car.jpg --device cpu
```

---

## 🌐 API Usage

```bash
# Start server
python main.py --serve

# Upload image
curl -X POST http://localhost:8000/assess \
  -F "file=@damaged_car.jpg" \
  -F "vehicle_make_model=Toyota Camry" \
  -F "vehicle_year=2019" \
  -F "generate_pdf=true"
```

Response:
```json
{
  "damages": [
    {"damage_type": "dent", "vehicle_part": "door_front_left",
     "severity": "moderate", "cost": "$500–$1200"}
  ],
  "estimate": {"grand_total": "$800–$2000"},
  "report_url": "/report/damage_report_CLM-ABC123.pdf"
}
```

---

## 🎓 Training on Real Datasets

| Dataset | Size | Link |
|---|---|---|
| **CarDD** | 4,000 images | [GitHub](https://github.com/CarDD-Dataset/CarDD) |
| **Vehicle Damage v2** | 3,500 images | [Roboflow](https://universe.roboflow.com/car-damage-kwmkc/vehicle-damage-v2) |
| **COCO Cars** | 5,000 images | [COCO](https://cocodataset.org) |

```bash
# Train detector
python models/train.py --task detector --data data/ --epochs 100

# Train severity classifier
python models/train.py --task classifier --data data/severity/ --epochs 50
```

---
## 📧 Contact

Feel free to connect for collaboration, internships, 
or project discussions.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/kokila-m-ai-ds/)
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:kokilakoki3376@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/kokilamariyayi)

---
<div align="center">⭐ Star this repo if it helped you!</div>
