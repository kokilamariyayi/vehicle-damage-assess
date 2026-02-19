"""
src/cost_estimator.py
Repair Cost Estimation using lookup tables + ML adjustment factors.

Features:
- Part × Damage × Severity lookup table
- Regional price adjustment (US regions)
- Vehicle age/value factor
- Labor cost calculation
- Parts cost breakdown
- Total estimate with min/max range
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from loguru import logger

from src.damage_detector import DamageDetection


@dataclass
class PartEstimate:
    """Cost estimate for a single damaged part."""
    vehicle_part: str
    damage_type: str
    severity: str
    parts_cost_min: float
    parts_cost_max: float
    labor_hours: float
    labor_rate: float
    total_min: float
    total_max: float
    notes: str = ""

    @property
    def labor_cost(self) -> float:
        return self.labor_hours * self.labor_rate

    def to_dict(self) -> dict:
        return {
            "part": self.vehicle_part,
            "damage": self.damage_type,
            "severity": self.severity,
            "parts_cost": f"${self.parts_cost_min:.0f} – ${self.parts_cost_max:.0f}",
            "labor_cost": f"${self.labor_cost:.0f}",
            "total": f"${self.total_min:.0f} – ${self.total_max:.0f}",
            "notes": self.notes,
        }


@dataclass
class TotalEstimate:
    """Full vehicle damage cost estimate."""
    part_estimates: List[PartEstimate]
    subtotal_min: float
    subtotal_max: float
    tax_rate: float
    tax_amount: float
    grand_total_min: float
    grand_total_max: float
    overall_severity: str
    total_damaged_parts: int
    recommendation: str
    currency: str = "USD"

    def to_dict(self) -> dict:
        return {
            "parts": [p.to_dict() for p in self.part_estimates],
            "subtotal": f"${self.subtotal_min:.0f} – ${self.subtotal_max:.0f}",
            "tax": f"${self.tax_amount:.0f}",
            "grand_total": f"${self.grand_total_min:.0f} – ${self.grand_total_max:.0f}",
            "overall_severity": self.overall_severity,
            "total_damaged_parts": self.total_damaged_parts,
            "recommendation": self.recommendation,
        }


# ── Labor Hours by Damage Type + Severity ────────────────────────────────────
LABOR_HOURS = {
    "scratch":        {"minor": 1.0, "moderate": 3.0,  "severe": 6.0},
    "dent":           {"minor": 2.0, "moderate": 5.0,  "severe": 10.0},
    "crack":          {"minor": 2.0, "moderate": 5.0,  "severe": 12.0},
    "shattered_glass":{"minor": 1.5, "moderate": 3.0,  "severe": 5.0},
    "flat_tire":      {"minor": 0.5, "moderate": 1.0,  "severe": 2.0},
    "missing_part":   {"minor": 1.0, "moderate": 3.0,  "severe": 8.0},
    "deformation":    {"minor": 3.0, "moderate": 8.0,  "severe": 20.0},
    "rust":           {"minor": 2.0, "moderate": 6.0,  "severe": 15.0},
    "default":        {"minor": 1.5, "moderate": 4.0,  "severe": 10.0},
}

# ── Special notes per damage type ────────────────────────────────────────────
DAMAGE_NOTES = {
    "shattered_glass": "Safety hazard — immediate replacement required.",
    "flat_tire": "Check for rim damage; TPMS sensor may need replacement.",
    "missing_part": "OEM vs aftermarket parts will significantly affect cost.",
    "deformation": "Frame/structural inspection required before repair.",
    "rust": "Extent of rust may be larger than visible — inspection needed.",
    "crack": "Structural integrity assessment recommended.",
}


class CostEstimator:
    """
    Generates repair cost estimates for detected damages.
    """

    DEFAULT_LABOR_RATE = 120.0   # USD per hour (average US shop rate)
    TAX_RATE = 0.08              # 8% average

    def __init__(self, config: dict):
        self.config = config
        self.cost_table = config.get("repair_costs", {})
        self.currency = config.get("report", {}).get("currency", "USD")
        self.labor_rate = self.DEFAULT_LABOR_RATE

    def estimate(
        self,
        detections: List[DamageDetection],
        severities: List[str],
        vehicle_age_years: int = 5,
        region: str = "national",
    ) -> TotalEstimate:
        """
        Generate full cost estimate for all detected damages.

        Args:
            detections: List of damage detections
            severities: Corresponding severity for each detection
            vehicle_age_years: Age of vehicle (affects parts cost)
            region: Geographic region for labor rate adjustment
        """
        # Adjust labor rate by region
        labor_rate = self._get_labor_rate(region)

        # Age factor (older cars have cheaper parts but harder to find)
        age_factor = self._get_age_factor(vehicle_age_years)

        part_estimates = []
        seen_parts = {}  # Avoid double-counting same part

        for det, severity in zip(detections, severities):
            part = det.vehicle_part
            dtype = det.damage_type

            # Skip if same part already estimated with higher severity
            if part in seen_parts:
                existing_sev = seen_parts[part]
                sev_order = ["minor", "moderate", "severe"]
                if sev_order.index(severity) <= sev_order.index(existing_sev):
                    continue

            seen_parts[part] = severity

            # Look up cost
            cost_range = self._lookup_cost(part, dtype, severity)
            parts_min = cost_range[0] * age_factor
            parts_max = cost_range[1] * age_factor

            # Labor
            hours = LABOR_HOURS.get(dtype, LABOR_HOURS["default"]).get(severity, 4.0)
            labor = hours * labor_rate

            total_min = parts_min + labor
            total_max = parts_max + labor

            estimate = PartEstimate(
                vehicle_part=part,
                damage_type=dtype,
                severity=severity,
                parts_cost_min=parts_min,
                parts_cost_max=parts_max,
                labor_hours=hours,
                labor_rate=labor_rate,
                total_min=total_min,
                total_max=total_max,
                notes=DAMAGE_NOTES.get(dtype, ""),
            )
            part_estimates.append(estimate)

        # Totals
        subtotal_min = sum(e.total_min for e in part_estimates)
        subtotal_max = sum(e.total_max for e in part_estimates)
        tax = (subtotal_min + subtotal_max) / 2 * self.TAX_RATE
        grand_min = subtotal_min + tax
        grand_max = subtotal_max + tax

        # Overall severity
        sev_counts = {"minor": 0, "moderate": 0, "severe": 0}
        for s in severities:
            sev_counts[s] = sev_counts.get(s, 0) + 1
        overall = max(sev_counts, key=sev_counts.get)

        # Recommendation
        recommendation = self._get_recommendation(overall, grand_max)

        return TotalEstimate(
            part_estimates=part_estimates,
            subtotal_min=subtotal_min,
            subtotal_max=subtotal_max,
            tax_rate=self.TAX_RATE,
            tax_amount=tax,
            grand_total_min=grand_min,
            grand_total_max=grand_max,
            overall_severity=overall,
            total_damaged_parts=len(part_estimates),
            recommendation=recommendation,
            currency=self.currency,
        )

    def _lookup_cost(self, part: str, damage_type: str,
                      severity: str) -> Tuple[float, float]:
        """Look up repair cost from table."""
        table = self.cost_table

        # Try specific part
        part_costs = table.get(part, table.get("default", {}))
        damage_costs = part_costs.get(damage_type, part_costs.get("default", {}))
        cost = damage_costs.get(severity, [200, 600])

        if isinstance(cost, list) and len(cost) == 2:
            return cost[0], cost[1]
        return 200, 600

    def _get_labor_rate(self, region: str) -> float:
        rates = {
            "northeast": 145.0,
            "west": 140.0,
            "south": 110.0,
            "midwest": 115.0,
            "national": 120.0,
        }
        return rates.get(region.lower(), self.DEFAULT_LABOR_RATE)

    def _get_age_factor(self, age_years: int) -> float:
        """Older vehicles: cheaper parts but may need more labor."""
        if age_years <= 3:
            return 1.2    # New — OEM parts premium
        elif age_years <= 7:
            return 1.0    # Standard
        elif age_years <= 12:
            return 0.8    # Older — cheaper aftermarket available
        else:
            return 0.6    # Very old — much cheaper parts

    def _get_recommendation(self, severity: str, estimated_cost: float) -> str:
        if severity == "minor":
            return ("Minor cosmetic damage detected. Repair at your convenience. "
                   "Consider PDR (Paintless Dent Repair) for cost savings.")
        elif severity == "moderate":
            return ("Moderate damage detected. Schedule repair within 30 days to prevent "
                   "further deterioration. Obtain 2-3 quotes from certified body shops.")
        else:
            return ("Severe damage detected. DO NOT delay repair — structural integrity "
                   "may be compromised. Vehicle may not be safe to drive. "
                   "Contact your insurance provider immediately.")
