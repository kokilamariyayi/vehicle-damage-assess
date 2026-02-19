"""
main.py
Vehicle Damage Assessment System — CLI Entry Point

Usage:
    # Assess a single image
    python main.py --image path/to/car.jpg

    # Assess and save PDF report
    python main.py --image car.jpg --report

    # Run web API server
    python main.py --serve

    # Batch process a folder
    python main.py --folder path/to/images/
"""

import cv2
import yaml
import argparse
import os
import sys
from pathlib import Path
from loguru import logger


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def assess_image(image_path: str, config: dict, save_report: bool = True,
                  vehicle_info: dict = None, show: bool = True):
    """Run full assessment pipeline on a single image."""
    from src.preprocessor import ImagePreprocessor
    from src.damage_detector import DamageDetector
    from src.severity_classifier import SeverityClassifier
    from src.cost_estimator import CostEstimator
    from reports.report_generator import ReportGenerator
    from utils.visualizer import DamageVisualizer

    logger.info(f"Assessing: {image_path}")

    # Init
    preprocessor = ImagePreprocessor(config["models"]["input_size"])
    detector     = DamageDetector(config)
    classifier   = SeverityClassifier(config)
    estimator    = CostEstimator(config)
    visualizer   = DamageVisualizer()

    # Preprocess
    original, processed = preprocessor.preprocess(image_path)

    # Detect
    detections = detector.detect(processed)
    if not detections:
        logger.warning("No damage detected in this image.")
        if show:
            cv2.imshow("No Damage Detected", original)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    logger.info(f"Found {len(detections)} damage regions.")

    # Classify severity
    severities, confs = [], []
    for det in detections:
        crop = det.crop if det.crop is not None else processed
        sev, conf, _ = classifier.classify(crop, det.damage_type, det.area_pct)
        severities.append(sev)
        confs.append(conf)
        logger.info(f"  [{sev.upper()}] {det.damage_type} on {det.vehicle_part} "
                   f"(conf: {conf:.0%})")

    # Estimate cost
    age = 5
    if vehicle_info:
        age = 2024 - vehicle_info.get("year", 2019)
    estimate = estimator.estimate(detections, severities, age)

    # Print summary
    print("\n" + "="*55)
    print("  VEHICLE DAMAGE ASSESSMENT SUMMARY")
    print("="*55)
    print(f"  Overall Severity : {estimate.overall_severity.upper()}")
    print(f"  Damages Found    : {len(detections)}")
    print(f"  Estimated Cost   : ${estimate.grand_total_min:.0f} – ${estimate.grand_total_max:.0f}")
    print(f"  Recommendation   : {estimate.recommendation[:80]}...")
    print("="*55)
    for i, (det, sev) in enumerate(zip(detections, severities), 1):
        print(f"  {i}. [{sev.upper():8s}] {det.damage_type:20s} → {det.vehicle_part}")
    print("="*55 + "\n")

    # Visualize
    annotated = visualizer.annotate(original, detections, severities, estimate)
    heatmap   = visualizer.draw_severity_heatmap(original, detections, severities)

    if show:
        cv2.imshow("Damage Detection", annotated)
        cv2.imshow("Severity Heatmap", heatmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save outputs
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(image_path).stem
    cv2.imwrite(f"{out_dir}/{stem}_annotated.jpg", annotated)
    cv2.imwrite(f"{out_dir}/{stem}_heatmap.jpg", heatmap)
    logger.info(f"Saved annotated image and heatmap to {out_dir}/")

    # Generate PDF report
    if save_report:
        report_gen = ReportGenerator(config)
        report_path = report_gen.generate(
            original, annotated, detections, severities, estimate, vehicle_info
        )
        logger.success(f"PDF Report: {report_path}")

    return estimate


def batch_assess(folder: str, config: dict):
    """Process all images in a folder."""
    from src.preprocessor import ImagePreprocessor
    image_files = list(Path(folder).glob("*.jpg")) + \
                  list(Path(folder).glob("*.png")) + \
                  list(Path(folder).glob("*.jpeg"))

    logger.info(f"Found {len(image_files)} images in {folder}")

    results = []
    for img_path in image_files:
        try:
            estimate = assess_image(str(img_path), config,
                                    save_report=True, show=False)
            if estimate:
                results.append({
                    "image": img_path.name,
                    "severity": estimate.overall_severity,
                    "cost_min": estimate.grand_total_min,
                    "cost_max": estimate.grand_total_max,
                })
        except Exception as e:
            logger.error(f"Failed on {img_path.name}: {e}")

    # Summary CSV
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = "output/batch_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.success(f"Batch summary saved: {csv_path}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Damage Assessment AI")
    parser.add_argument("--image", help="Path to vehicle image")
    parser.add_argument("--folder", help="Folder of images for batch processing")
    parser.add_argument("--report", action="store_true", help="Generate PDF report")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--no-display", action="store_true", help="Don't show windows")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--vehicle", default="Unknown Vehicle")
    parser.add_argument("--year", type=int, default=2020)
    args = parser.parse_args()

    config = load_config(args.config)

    os.makedirs("logs", exist_ok=True)
    logger.add(config["logging"]["log_file"], rotation="10 MB")

    if args.device:
        config["inference"]["device"] = args.device

    if args.serve:
        import uvicorn
        logger.info("Starting API server at http://localhost:8000")
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

    elif args.folder:
        batch_assess(args.folder, config)

    elif args.image:
        vehicle_info = {"make_model": args.vehicle, "year": args.year}
        assess_image(
            args.image, config,
            save_report=args.report,
            vehicle_info=vehicle_info,
            show=not args.no_display,
        )

    else:
        parser.print_help()
