"""
KPI Calculator — Z-Score + BSC + Cross-Role Ranking
=====================================================
Implements the 3-step fairness pipeline from [[employee-KPI-normalization-research]]:

  1. Within-role Z-Score:  z = (x - mu) / sigma
  2. BSC weighted sum:     S = sum(w_k * z_k) per role
  3. Cross-role ranking:   all employees ranked by S_i (same scale)

References:
  - Kaplan & Norton (1992), Balanced Scorecard
  - Fisher (1925), Z-Score formalization
  - Purwanto & Asbari (2021), procedural fairness > outcome fairness
"""

import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from kpi.config import ROLES, BSC_WEIGHTS


class KPICalculator:
    def __init__(self):
        self.roles = ROLES
        self.bsc_weights = BSC_WEIGHTS

    def normalize_within_role(self, employees_data):
        """
        Step 1: Z-Score normalization within each role.

        Args:
            employees_data: list of dicts, each with:
                {id, name, role, kpis: {kpi_name: raw_value, ...}}

        Returns:
            Same list with added "z_scores" field per employee.
        """
        # Group by role
        by_role = defaultdict(list)
        for emp in employees_data:
            by_role[emp["role"]].append(emp)

        for role, emps in by_role.items():
            role_config = self.roles.get(role)
            if not role_config:
                continue

            for kpi_name, kpi_config in role_config["kpis"].items():
                # Collect raw values for this KPI across all employees in this role
                values = []
                for emp in emps:
                    raw = emp.get("kpis", {}).get(kpi_name)
                    if raw is not None:
                        values.append(raw)

                if len(values) < 2:
                    continue  # Need at least 2 to compute std

                mean = np.mean(values)
                std = np.std(values, ddof=1)  # Sample std
                if std == 0:
                    std = 1.0  # Avoid division by zero

                for emp in emps:
                    raw = emp.get("kpis", {}).get(kpi_name)
                    if raw is None:
                        continue
                    if "z_scores" not in emp:
                        emp["z_scores"] = {}

                    # For "lower_better" KPIs, invert the Z-score
                    z = (raw - mean) / std
                    if kpi_config["direction"] == "lower_better":
                        z = -z
                    emp["z_scores"][kpi_name] = round(z, 4)

        return employees_data

    def compute_bsc_score(self, employees_data):
        """
        Step 2: Weighted BSC aggregation per employee.

        For each KPI: z_score * kpi_weight
        Aggregated by BSC dimension, then by BSC dimension weight.

        Args:
            employees_data: must already have "z_scores" from normalize_within_role()

        Returns:
            Same list with added "bsc_scores" and "total_score" fields.
        """
        for emp in employees_data:
            role = emp["role"]
            role_config = self.roles.get(role)
            if not role_config:
                continue

            emp["bsc_scores"] = {dim: 0.0 for dim in self.bsc_weights}
            emp["total_score"] = 0.0

            for kpi_name, kpi_config in role_config["kpis"].items():
                z = emp.get("z_scores", {}).get(kpi_name, 0.0)
                dim = kpi_config["bsc_dimension"]
                kpi_weight = kpi_config["weight"]

                # Weight the Z-score
                weighted_z = z * kpi_weight
                emp["bsc_scores"][dim] += weighted_z

            # Aggregate by BSC dimension weights
            for dim, dim_weight in self.bsc_weights.items():
                emp["total_score"] += emp["bsc_scores"][dim] * dim_weight

            emp["total_score"] = round(emp["total_score"], 4)

        return employees_data

    def cross_role_ranking(self, employees_data):
        """
        Step 3: Unified cross-role ranking.

        All employees sorted by total_score (same scale, comparable).
        Adds "rank" and "percentile" fields.

        Returns:
            List sorted by total_score descending.
        """
        ranked = sorted(employees_data, key=lambda e: e.get("total_score", 0), reverse=True)
        n = len(ranked)

        for i, emp in enumerate(ranked):
            emp["rank"] = i + 1
            emp["percentile"] = round((n - i) / n * 100, 1)

        return ranked

    def full_pipeline(self, employees_data):
        """
        Run the complete 3-step pipeline.

        Returns:
            Ranked list with z_scores, bsc_scores, total_score, rank, percentile.
        """
        data = self.normalize_within_role(employees_data)
        data = self.compute_bsc_score(data)
        return self.cross_role_ranking(data)

    def generate_report(self, ranked_data, month=None):
        """
        Generate a monthly KPI report ready for dashboard display.

        Returns:
            dict with summary, ranking, per-role breakdown
        """
        if month is None:
            month = datetime.now().strftime("%Y-%m")

        report = {
            "month": month,
            "generated_at": datetime.now().isoformat(),
            "total_employees": len(ranked_data),
            "ranking": [],
            "by_role": {},
            "top_performer": None,
            "most_improved": None,  # requires previous month data
        }

        for emp in ranked_data:
            role = emp["role"]
            if role not in report["by_role"]:
                report["by_role"][role] = {
                    "role_name": self.roles.get(role, {}).get("name_cn", role),
                    "employees": [],
                    "avg_total_score": 0.0,
                }

            entry = {
                "id": emp["id"],
                "name": emp["name"],
                "role": emp["role"],
                "role_name": self.roles.get(role, {}).get("name_cn", role),
                "total_score": emp["total_score"],
                "rank": emp["rank"],
                "percentile": emp["percentile"],
                "bsc_breakdown": emp.get("bsc_scores", {}),
                "kpi_details": {},
            }

            # Add individual KPI details
            role_config = self.roles.get(role, {})
            for kpi_name, kpi_config in role_config.get("kpis", {}).items():
                raw = emp.get("kpis", {}).get(kpi_name)
                z = emp.get("z_scores", {}).get(kpi_name, 0.0)
                if raw is not None:
                    entry["kpi_details"][kpi_name] = {
                        "name_cn": kpi_config["name_cn"],
                        "raw_value": raw,
                        "z_score": z,
                        "unit": kpi_config["unit"],
                        "dimension": kpi_config["bsc_dimension"],
                    }

            report["ranking"].append(entry)
            report["by_role"][role]["employees"].append(entry)

        # Compute per-role averages
        for role, data in report["by_role"].items():
            scores = [e["total_score"] for e in data["employees"]]
            data["avg_total_score"] = round(np.mean(scores), 4) if scores else 0.0
            # Sort within role
            data["employees"].sort(key=lambda e: e["total_score"], reverse=True)

        # Top performer
        if ranked_data:
            top = ranked_data[0]
            report["top_performer"] = {
                "name": top["name"],
                "role": self.roles.get(top["role"], {}).get("name_cn", top["role"]),
                "score": top["total_score"],
            }

        return report

    def dashboard_format(self, report):
        """
        Convert report to dashboard-ready format matching [[dashboard-designs]].

        Returns:
            dict with panels: attendance, shift, kpi_summary
        """
        return {
            "kpi_summary": {
                "month": report["month"],
                "total_employees": report["total_employees"],
                "top_performer": report["top_performer"],
                "ranking": [
                    {
                        "rank": e["rank"],
                        "name": e["name"],
                        "role": e["role_name"],
                        "score": e["total_score"],
                        "percentile": e["percentile"],
                        "highlights": self._get_highlights(e),
                    }
                    for e in report["ranking"]
                ],
            },
            "role_breakdown": {
                role: {
                    "name": data["role_name"],
                    "avg_score": data["avg_total_score"],
                    "top": data["employees"][0]["name"] if data["employees"] else "-",
                }
                for role, data in report["by_role"].items()
            },
        }

    def _get_highlights(self, entry, top_n=2):
        """Extract top-2 highest and lowest Z-scores for display."""
        details = entry.get("kpi_details", {})
        if not details:
            return {"strengths": [], "weaknesses": []}

        sorted_kpis = sorted(details.items(), key=lambda x: x[1]["z_score"], reverse=True)
        strengths = [{"kpi": kpi_config["name_cn"], "z": kpi_config["z_score"]}
                     for _, kpi_config in sorted_kpis[:top_n]]
        weaknesses = [{"kpi": kpi_config["name_cn"], "z": kpi_config["z_score"]}
                      for _, kpi_config in sorted_kpis[-top_n:]]
        return {"strengths": strengths, "weaknesses": weaknesses}
