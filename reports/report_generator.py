"""
reports/report_generator.py
Generates structured PDF reports for vehicle damage assessments.

Report contains:
- Vehicle info header
- Original + annotated image
- Damage summary table
- Per-part severity breakdown
- Cost estimate table
- Recommendation
- Claim reference number
"""

import os
import cv2
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from loguru import logger

from src.damage_detector import DamageDetection
from src.cost_estimator import TotalEstimate


class ReportGenerator:
    """Generates professional PDF damage assessment reports."""

    SEVERITY_COLORS = {
        "minor":    (0.1, 0.7, 0.1),    # Green
        "moderate": (0.9, 0.5, 0.0),    # Orange
        "severe":   (0.8, 0.0, 0.0),    # Red
    }

    def __init__(self, config: dict):
        self.config = config
        self.company = config.get("report", {}).get("company_name", "AutoAssess AI")
        self.output_dir = config.get("report", {}).get("output_dir", "reports/output/")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(
        self,
        original_image: np.ndarray,
        annotated_image: np.ndarray,
        detections: List[DamageDetection],
        severities: List[str],
        estimate: TotalEstimate,
        vehicle_info: dict = None,
    ) -> str:
        """
        Generate a full PDF report.

        Returns:
            Path to generated PDF file
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table,
                TableStyle, Image as RLImage, HRFlowable
            )
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        except ImportError:
            logger.error("reportlab not installed. Run: pip install reportlab")
            return self._generate_text_report(detections, severities, estimate, vehicle_info)

        claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"damage_report_{claim_id}.pdf"
        filepath = os.path.join(self.output_dir, filename)

        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )

        styles = getSampleStyleSheet()
        story = []

        # ── Header ───────────────────────────────────────────────────────────
        title_style = ParagraphStyle(
            "Title", parent=styles["Title"],
            fontSize=20, textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=4,
        )
        subtitle_style = ParagraphStyle(
            "Sub", parent=styles["Normal"],
            fontSize=10, textColor=colors.grey,
            spaceAfter=2,
        )

        story.append(Paragraph(f"🚗 {self.company}", title_style))
        story.append(Paragraph("Vehicle Damage Assessment Report", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=2,
                                color=colors.HexColor("#1a1a2e")))
        story.append(Spacer(1, 10))

        # ── Claim Info ────────────────────────────────────────────────────────
        info_data = [
            ["Claim Reference", claim_id, "Assessment Date", timestamp],
            ["Overall Severity", estimate.overall_severity.upper(),
             "Total Damages Found", str(estimate.total_damaged_parts)],
        ]
        if vehicle_info:
            info_data.append(["Vehicle", vehicle_info.get("make_model", "N/A"),
                              "Year", str(vehicle_info.get("year", "N/A"))])

        info_table = Table(info_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
        sev_color = self._sev_color_rl(estimate.overall_severity)
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f8f9fa")),
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
            ("ROWBACKGROUNDS", (0,0), (-1,-1),
             [colors.HexColor("#f8f9fa"), colors.HexColor("#ffffff")]),
            ("TEXTCOLOR", (1,1), (1,1), sev_color),
            ("FONTNAME", (1,1), (1,1), "Helvetica-Bold"),
            ("PADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 15))

        # ── Images ────────────────────────────────────────────────────────────
        story.append(Paragraph("Damage Assessment Images", styles["Heading2"]))
        story.append(Spacer(1, 6))

        img_paths = []
        for img, label in [(original_image, "original"), (annotated_image, "annotated")]:
            tmp_path = os.path.join(self.output_dir, f"_tmp_{label}.jpg")
            cv2.imwrite(tmp_path, img)
            img_paths.append((tmp_path, label))

        img_row = []
        for path, label in img_paths:
            rl_img = RLImage(path, width=3.2*inch, height=2.4*inch)
            img_row.append([rl_img, Paragraph(
                f"<b>{'Original Photo' if label=='original' else 'Damage Analysis'}</b>",
                ParagraphStyle("c", parent=styles["Normal"],
                              fontSize=8, alignment=TA_CENTER)
            )])

        img_table = Table([[img_row[0][0], img_row[1][0]],
                           [img_row[0][1], img_row[1][1]]],
                          colWidths=[3.5*inch, 3.5*inch])
        img_table.setStyle(TableStyle([
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 15))

        # ── Damage Details ────────────────────────────────────────────────────
        story.append(Paragraph("Detected Damage Details", styles["Heading2"]))
        story.append(Spacer(1, 6))

        det_headers = ["#", "Vehicle Part", "Damage Type", "Severity",
                       "Coverage Area", "Confidence"]
        det_data = [det_headers]
        for i, (det, sev) in enumerate(zip(detections, severities), 1):
            det_data.append([
                str(i),
                det.vehicle_part.replace("_", " ").title(),
                det.damage_type.replace("_", " ").title(),
                sev.upper(),
                f"{det.area_pct:.1f}%",
                f"{det.confidence:.0%}",
            ])

        det_table = Table(det_data, colWidths=[0.4*inch, 1.5*inch, 1.3*inch,
                                                1.0*inch, 1.1*inch, 1.0*inch])
        det_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.HexColor("#ffffff"), colors.HexColor("#f8f9fa")]),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("PADDING", (0,0), (-1,-1), 5),
        ]))
        # Color severity cells
        for i, sev in enumerate(severities, 1):
            det_table.setStyle(TableStyle([
                ("TEXTCOLOR", (3,i), (3,i), self._sev_color_rl(sev)),
                ("FONTNAME", (3,i), (3,i), "Helvetica-Bold"),
            ]))

        story.append(det_table)
        story.append(Spacer(1, 15))

        # ── Cost Estimate ─────────────────────────────────────────────────────
        story.append(Paragraph("Repair Cost Estimate", styles["Heading2"]))
        story.append(Spacer(1, 6))

        cost_headers = ["Vehicle Part", "Damage", "Severity",
                        "Parts Cost", "Labor Cost", "Total Range"]
        cost_data = [cost_headers]
        for pe in estimate.part_estimates:
            cost_data.append([
                pe.vehicle_part.replace("_", " ").title(),
                pe.damage_type.replace("_", " ").title(),
                pe.severity.upper(),
                f"${pe.parts_cost_min:.0f}–${pe.parts_cost_max:.0f}",
                f"${pe.labor_cost:.0f}",
                f"${pe.total_min:.0f}–${pe.total_max:.0f}",
            ])

        # Totals row
        cost_data.append([
            "SUBTOTAL", "", "", "", "",
            f"${estimate.subtotal_min:.0f}–${estimate.subtotal_max:.0f}"
        ])
        cost_data.append([
            "TAX (8%)", "", "", "", "",
            f"${estimate.tax_amount:.0f}"
        ])
        cost_data.append([
            "GRAND TOTAL", "", "", "", "",
            f"${estimate.grand_total_min:.0f}–${estimate.grand_total_max:.0f}"
        ])

        cost_table = Table(cost_data, colWidths=[1.4*inch, 1.1*inch, 0.9*inch,
                                                  1.2*inch, 1.0*inch, 1.2*inch])
        cost_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.HexColor("#ffffff"), colors.HexColor("#f8f9fa")]),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
            ("ALIGN", (3,0), (-1,-1), "RIGHT"),
            ("FONTNAME", (0,-3), (-1,-1), "Helvetica-Bold"),
            ("BACKGROUND", (0,-1), (-1,-1), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0,-1), (-1,-1), colors.white),
            ("PADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(cost_table)
        story.append(Spacer(1, 15))

        # ── Recommendation ────────────────────────────────────────────────────
        story.append(Paragraph("Recommendation", styles["Heading2"]))
        rec_style = ParagraphStyle(
            "Rec", parent=styles["Normal"],
            fontSize=10, backColor=colors.HexColor("#fff3cd"),
            borderColor=colors.HexColor("#ffc107"),
            borderWidth=1, borderPadding=8, leading=16,
        )
        story.append(Paragraph(estimate.recommendation, rec_style))
        story.append(Spacer(1, 20))

        # ── Footer ────────────────────────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        footer_style = ParagraphStyle("Footer", parent=styles["Normal"],
                                      fontSize=7, textColor=colors.grey,
                                      alignment=TA_CENTER)
        story.append(Paragraph(
            f"Generated by {self.company} AI System | {timestamp} | Claim: {claim_id}<br/>"
            "This estimate is AI-generated. Actual repair costs may vary. "
            "Please obtain quotes from certified repair shops.",
            footer_style
        ))

        doc.build(story)

        # Cleanup temp images
        for path, _ in img_paths:
            try: os.remove(path)
            except: pass

        logger.success(f"Report generated: {filepath}")
        return filepath

    def _sev_color_rl(self, severity: str):
        from reportlab.lib import colors as rl_colors
        return {
            "minor": rl_colors.HexColor("#198754"),
            "moderate": rl_colors.HexColor("#fd7e14"),
            "severe": rl_colors.HexColor("#dc3545"),
        }.get(severity, rl_colors.black)

    def _generate_text_report(self, detections, severities, estimate, vehicle_info) -> str:
        """Fallback text report if reportlab unavailable."""
        path = os.path.join(self.output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(path, "w") as f:
            f.write("VEHICLE DAMAGE ASSESSMENT REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Overall Severity: {estimate.overall_severity.upper()}\n")
            f.write(f"Total Damages: {estimate.total_damaged_parts}\n\n")
            f.write("DAMAGE DETAILS:\n")
            for i, (det, sev) in enumerate(zip(detections, severities), 1):
                f.write(f"  {i}. {det.vehicle_part} — {det.damage_type} [{sev}]\n")
            f.write(f"\nESTIMATED COST: ${estimate.grand_total_min:.0f} – ${estimate.grand_total_max:.0f}\n")
            f.write(f"\nRECOMMENDATION:\n{estimate.recommendation}\n")
        return path
