"""
KPI Calculator - Robust Z + BSC Weighted Aggregation
======================================================
Two-phase fairness pipeline:

  1. Within-role Robust Z:  z = (x - median) / MAD
     (Iglewicz & Hoaglin 1993)
  2. BSC weighted aggregation: group Z-scores by BSC dimension,
     apply per-KPI weights, then cross-dimension BSC weights
     (Kaplan & Norton 1992)

Dual-role support: employees with role2 are evaluated in both roles,
                    the higher BSC score is kept for final ranking.
"""

import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from kpi.config import ROLES, BSC_WEIGHTS


class KPICalculator:
    def __init__(self):
        self.roles = ROLES

    # ==================================================================
    # PHASE 1: Robust Z-Score within each role
    # ==================================================================

    def normalize_within_role(self, employees_data):
        by_role = defaultdict(list)
        for emp in employees_data:
            by_role[emp["role"]].append(emp)

        for role, emps in by_role.items():
            role_config = self.roles.get(role)
            if not role_config:
                continue

            for kpi_name, kpi_config in role_config["kpis"].items():
                values = []
                for emp in emps:
                    raw = emp.get("kpis", {}).get(kpi_name)
                    if raw is not None:
                        values.append(raw)

                if len(values) < 2:
                    for emp in emps:
                        if "z_scores" not in emp:
                            emp["z_scores"] = {}
                        emp["z_scores"][kpi_name] = 0.0
                    continue

                median = np.median(values)
                mad = np.median(np.abs(np.array(values) - median))
                if mad < 1e-10:
                    # Fallback to std when all/most values identical (small sample edge case)
                    std_val = np.std(values)
                    mad = max(std_val, abs(median) * 0.01, 1e-6)

                for emp in emps:
                    raw = emp.get("kpis", {}).get(kpi_name)
                    if raw is None:
                        continue
                    if "z_scores" not in emp:
                        emp["z_scores"] = {}
                    z = (raw - median) / mad
                    if kpi_config["direction"] == "lower_better":
                        z = -z
                    emp["z_scores"][kpi_name] = round(float(z), 4)

        # Cross-role normalization for shared KPIs (punctuality)
        cross_kpis = {}
        for role, emps in by_role.items():
            role_config = self.roles.get(role)
            if not role_config:
                continue
            for kpi_name, kpi_config in role_config["kpis"].items():
                if kpi_config.get("cross_role"):
                    if kpi_name not in cross_kpis:
                        cross_kpis[kpi_name] = []
                    for emp in emps:
                        raw = emp.get("kpis", {}).get(kpi_name)
                        if raw is not None:
                            cross_kpis[kpi_name].append((emp, raw))

        for kpi_name, pairs in cross_kpis.items():
            values = [v for _, v in pairs]
            if len(values) < 2:
                continue
            median = np.median(values)
            mad = np.median(np.abs(np.array(values) - median))
            if mad == 0:
                mad = 1.0
            for emp, raw in pairs:
                if "z_scores" not in emp:
                    emp["z_scores"] = {}
                z = (raw - median) / mad
                direction = "higher_better"
                for role_cfg in self.roles.values():
                    kc = role_cfg.get("kpis", {}).get(kpi_name)
                    if kc:
                        direction = kc.get("direction", "higher_better")
                        break
                if direction == "lower_better":
                    z = -z
                emp["z_scores"][kpi_name] = round(float(z), 4)

        return employees_data

    # ==================================================================
    # PHASE 2: BSC Weighted Aggregation
    # ==================================================================

    def compute_bsc_aggregation(self, employees_data):
        if not employees_data:
            return employees_data

        for emp in employees_data:
            role = emp.get("role")
            role_config = self.roles.get(role)
            if not role_config:
                emp["total_score"] = 0.0
                emp["bsc_breakdown"] = {}
                continue

            dim_scores = {}
            dim_weight_sum = {}
            for dim_name in BSC_WEIGHTS:
                dim_scores[dim_name] = 0.0
                dim_weight_sum[dim_name] = 0.0

            for kpi_name, kpi_config in role_config["kpis"].items():
                dim = kpi_config.get("bsc_dimension", "Internal Process")
                z = emp.get("z_scores", {}).get(kpi_name, 0.0)
                w = kpi_config.get("weight", 0.0)
                dim_scores[dim] += z * w
                dim_weight_sum[dim] += w

            total_score = 0.0
            bsc_breakdown = {}
            for dim_name, dim_weight in BSC_WEIGHTS.items():
                if dim_weight_sum[dim_name] > 0:
                    dim_avg = dim_scores[dim_name] / dim_weight_sum[dim_name]
                else:
                    dim_avg = 0.0
                weighted = dim_avg * dim_weight
                total_score += weighted
                bsc_breakdown[dim_name] = round(weighted, 4)

            emp["total_score"] = round(total_score, 4)
            emp["bsc_breakdown"] = bsc_breakdown

        return employees_data

    # ==================================================================
    # PHASE 2.5: Dual-role resolution
    # ==================================================================

    def _resolve_dual_roles(self, employees_data):
        """For dual-role entries, keep the one with higher total_score."""
        groups = defaultdict(list)
        for emp in employees_data:
            if emp.get("is_dual_role"):
                primary = emp.get("primary_role")
                key = (emp["id"], primary)
                groups[key].append(emp)
            else:
                groups[(emp["id"], emp["role"])].append(emp)

        resolved = []
        for key, entries in groups.items():
            if len(entries) == 1:
                resolved.append(entries[0])
            else:
                best = max(entries, key=lambda e: e.get("total_score", -999))
                best["dual_role_evaluated_as"] = best["role"]
                best["role"] = key[1]  # primary role for display
                resolved.append(best)
        return resolved

    # ==================================================================
    # PHASE 3: Cross-role ranking
    # ==================================================================

    def cross_role_ranking(self, employees_data):
        employees_data.sort(key=lambda e: e["total_score"], reverse=True)
        n = len(employees_data)
        for i, emp in enumerate(employees_data):
            emp["rank"] = i + 1
            emp["percentile"] = round(100.0 - (i / max(n - 1, 1)) * 100.0, 1) if n > 1 else 100.0
        return employees_data

    # ==================================================================
    # Full pipeline
    # ==================================================================

    def full_pipeline(self, employees_data):
        data = self.normalize_within_role(employees_data)
        data = self.compute_bsc_aggregation(data)
        data = self._resolve_dual_roles(data)
        return self.cross_role_ranking(data)

    # ==================================================================
    # Report generation
    # ==================================================================

    def generate_report(self, ranked_data, month=None):
        if month is None:
            month = datetime.now().strftime("%Y-%m")

        report = {
            "month": month,
            "generated_at": datetime.now().isoformat(),
            "total_employees": len(ranked_data),
            "ranking": [],
            "by_role": {},
            "top_performer": None,
        }

        for emp in ranked_data:
            role = emp.get("dual_role_evaluated_as") or emp["role"]
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
                "primary_role": emp.get("primary_role", emp["role"]),
                "evaluated_as": emp.get("dual_role_evaluated_as"),
                "total_score": emp["total_score"],
                "rank": emp["rank"],
                "percentile": emp["percentile"],
                "bsc_breakdown": emp.get("bsc_breakdown", {}),
                "kpi_details": {},
            }

            eval_role = emp.get("dual_role_evaluated_as") or emp["role"]
            role_config = self.roles.get(eval_role, {})
            for kpi_name, kpi_config in role_config.get("kpis", {}).items():
                raw = emp.get("kpis", {}).get(kpi_name)
                z = emp.get("z_scores", {}).get(kpi_name, 0.0)
                if raw is not None:
                    entry["kpi_details"][kpi_name] = {
                        "name_cn": kpi_config["name_cn"],
                        "raw_value": raw,
                        "z_score": z,
                        "unit": kpi_config["unit"],
                    }

            report["ranking"].append(entry)
            report["by_role"][role]["employees"].append(entry)

        for role, data in report["by_role"].items():
            scores = [e["total_score"] for e in data["employees"]]
            data["avg_total_score"] = round(np.mean(scores), 4) if scores else 0.0
            data["employees"].sort(key=lambda e: e["total_score"], reverse=True)

        if ranked_data:
            top = ranked_data[0]
            report["top_performer"] = {
                "name": top["name"],
                "role": self.roles.get(top["role"], {}).get("name_cn", top["role"]),
                "score": top["total_score"],
            }

        return report

    def dashboard_format(self, report):
        return {
            "kpi_summary": {
                "month": report["month"],
                "total_employees": report["total_employees"],
                "top_performer": report["top_performer"],
                "ranking": [
                    {
                        "rank": e["rank"],
                        "name": e["name"],
                        "role": e["role_name"] if isinstance(e.get("role_name"), str) else e["role"],
                        "evaluated_as": e.get("evaluated_as"),
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
        details = entry.get("kpi_details", {})
        if not details:
            return {"strengths": [], "weaknesses": []}
        sorted_kpis = sorted(details.items(), key=lambda x: x[1]["z_score"], reverse=True)
        strengths = [{"kpi": v["name_cn"], "z": v["z_score"]}
                     for _, v in sorted_kpis[:top_n]]
        weaknesses = [{"kpi": v["name_cn"], "z": v["z_score"]}
                      for _, v in sorted_kpis[-top_n:]]
        return {"strengths": strengths, "weaknesses": weaknesses}
