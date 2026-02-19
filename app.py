"""
app.py
FastAPI web application for Vehicle Damage Assessment.

Endpoints:
  POST /assess          — Upload image → get damage report
  POST /assess/batch    — Upload multiple images
  GET  /report/{id}     — Download PDF report
  GET  /health          — Health check
"""

import os
import uuid
import shutil
import yaml
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from loguru import logger

# Load config
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

from src.preprocessor import ImagePreprocessor
from src.damage_detector import DamageDetector
from src.severity_classifier import SeverityClassifier
from src.cost_estimator import CostEstimator
from reports.report_generator import ReportGenerator
from utils.visualizer import DamageVisualizer
import cv2
import numpy as np

# Initialize components
preprocessor  = ImagePreprocessor(config["models"]["input_size"])
detector      = DamageDetector(config)
classifier    = SeverityClassifier(config)
estimator     = CostEstimator(config)
report_gen    = ReportGenerator(config)
visualizer    = DamageVisualizer()

app = FastAPI(
    title="Vehicle Damage Assessment API",
    description="AI-powered vehicle damage detection, severity classification, and cost estimation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/assess")
async def assess_damage(
    file: UploadFile = File(...),
    vehicle_make_model: str = Form(default="Unknown"),
    vehicle_year: int = Form(default=2020),
    region: str = Form(default="national"),
    generate_pdf: bool = Form(default=True),
):
    """
    Assess vehicle damage from an uploaded image.

    Returns:
    - Detected damage list with types and locations
    - Severity classification per damage
    - Cost estimate breakdown
    - PDF report download link (if requested)
    """
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    # Save upload
    file_id = uuid.uuid4().hex
    upload_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Preprocess
        original, processed = preprocessor.preprocess(upload_path)

        # Detect damages
        detections = detector.detect(processed)

        if not detections:
            return JSONResponse({
                "status": "success",
                "message": "No damage detected in this image.",
                "damages": [],
                "estimate": None,
            })

        # Classify severity for each detection
        severities = []
        confidences = []
        prob_lists = []
        for det in detections:
            if det.crop is not None:
                sev, conf, probs = classifier.classify(
                    det.crop, det.damage_type, det.area_pct
                )
            else:
                sev, conf, probs = "moderate", 0.6, [0.2, 0.6, 0.2]
            severities.append(sev)
            confidences.append(conf)
            prob_lists.append(probs)

        # Cost estimate
        vehicle_age = 2024 - vehicle_year
        estimate = estimator.estimate(detections, severities, vehicle_age, region)

        # Visualize
        annotated = visualizer.annotate(original, detections, severities, estimate)
        heatmap = visualizer.draw_severity_heatmap(original, detections, severities)

        # Save annotated image
        ann_path = os.path.join(UPLOAD_DIR, f"{file_id}_annotated.jpg")
        cv2.imwrite(ann_path, annotated)

        # Generate PDF
        report_path = None
        if generate_pdf:
            vehicle_info = {"make_model": vehicle_make_model, "year": vehicle_year}
            report_path = report_gen.generate(
                original, annotated, detections, severities, estimate, vehicle_info
            )

        # Build response
        response = {
            "status": "success",
            "claim_id": file_id,
            "vehicle": {"make_model": vehicle_make_model, "year": vehicle_year},
            "damages": [
                {
                    **det.to_dict(),
                    "severity": sev,
                    "severity_confidence": round(conf, 3),
                    "severity_probabilities": {
                        "minor": round(probs[0], 3),
                        "moderate": round(probs[1], 3),
                        "severe": round(probs[2], 3),
                    }
                }
                for det, sev, conf, probs in zip(detections, severities, confidences, prob_lists)
            ],
            "estimate": estimate.to_dict(),
            "annotated_image_url": f"/image/{file_id}_annotated.jpg",
        }

        if report_path:
            response["report_url"] = f"/report/{Path(report_path).name}"

        return JSONResponse(response)

    except Exception as e:
        logger.error(f"Assessment failed: {e}")
        raise HTTPException(500, f"Assessment failed: {str(e)}")
    finally:
        # Keep upload for report generation, clean later
        pass


@app.post("/assess/batch")
async def assess_batch(files: list[UploadFile] = File(...)):
    """Assess multiple images and return combined report."""
    results = []
    for file in files[:10]:  # Max 10 images
        result = await assess_damage(file)
        results.append(result)
    return {"batch_results": results, "total": len(results)}


@app.get("/report/{filename}")
def download_report(filename: str):
    """Download a generated PDF report."""
    path = os.path.join(config["report"]["output_dir"], filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Report not found.")
    return FileResponse(path, media_type="application/pdf",
                       filename=filename)


@app.get("/image/{filename}")
def get_image(filename: str):
    """Get annotated image."""
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Image not found.")
    return FileResponse(path, media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
