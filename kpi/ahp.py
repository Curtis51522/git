"""
AHP (Analytic Hierarchy Process) Weight Calculator
====================================================
Saaty's pairwise comparison method for scientific KPI weight determination.

Based on: [[kpi-research]] — Yu (2025), Shi & Wang (2020)

Scale (Saaty 1-9):
  1 = equally important
  3 = moderately more important
  5 = strongly more important
  7 = very strongly more important
  9 = extremely more important

Consistency check: CR < 0.10 required for acceptable weights.
"""

import numpy as np


class AHPCalculator:
    """AHP pairwise comparison → weights with consistency check."""

    def __init__(self):
        self.random_index = {
            1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
        }

    def compute_weights(self, pairwise_matrix):
        """
        Compute AHP weights from pairwise comparison matrix.

        Args:
            pairwise_matrix: n x n list of lists, where M[i][j] = importance of i over j
                            (Saaty 1-9 scale). Diagonal = 1. M[j][i] = 1/M[i][j].

        Returns:
            dict: {weights: list, eigenvalue: float, CI: float, CR: float, consistent: bool}
        """
        n = len(pairwise_matrix)
        M = np.array(pairwise_matrix, dtype=float)

        # Normalize columns
        col_sums = M.sum(axis=0)
        norm = M / col_sums

        # Weights = row averages
        weights = norm.mean(axis=1)

        # Consistency check
        # lambda_max = average of (A*w / w)
        Aw = M.dot(weights)
        lambda_max = (Aw / weights).mean()

        CI = (lambda_max - n) / (n - 1) if n > 1 else 0
        RI = self.random_index.get(n, 1.49)
        CR = CI / RI if RI > 0 else 0

        return {
            "weights": [round(w, 4) for w in weights],
            "eigenvalue": round(lambda_max, 4),
            "CI": round(CI, 4),
            "CR": round(CR, 4),
            "consistent": CR < 0.10,
        }

    def bsc_dimension_weights(self, custom_comparisons=None):
        """
        Default BSC dimension pairwise comparison.
        Customizable via custom_comparisons dict.
        """
        if custom_comparisons is None:
            # Default: Internal > Financial = Customer > Learning
            # Matrix: [Financial, Customer, Internal, Learning]
            M = [
                [1,   1,   1/2, 2  ],   # Financial
                [1,   1,   1/2, 2  ],   # Customer
                [2,   2,   1,   3  ],   # Internal Process
                [1/2, 1/2, 1/3, 1  ],   # Learning & Growth
            ]
        else:
            M = custom_comparisons

        result = self.compute_weights(M)
        dims = ["Financial", "Customer", "Internal Process", "Learning & Growth"]
        result["dimension_weights"] = {
            dims[i]: result["weights"][i] for i in range(len(dims))
        }
        return result

    def role_kpi_weights(self, role_config):
        """
        Compute AHP weights for KPIs within a role.
        
        Args:
            role_config: dict from kpi.config.ROLES[role]

        Returns:
            dict: {kpi_name: weight} with consistency check
        """
        kpi_names = list(role_config["kpis"].keys())
        n = len(kpi_names)

        if n <= 2:
            # Simple equal-weight fallback for 1-2 KPIs
            w = 1.0 / n
            return {
                "kpi_weights": {k: round(w, 4) for k in kpi_names},
                "CR": 0.0,
                "consistent": True,
            }

        # Build pairwise matrix using BSC dimension priority as heuristic
        # Internal Process > Financial > Customer > Learning (default bakery priority)
        dim_priority = {"Internal Process": 4, "Financial": 3, "Customer": 2, "Learning & Growth": 1}
        
        M = []
        for i, ki in enumerate(kpi_names):
            row = []
            dim_i = role_config["kpis"][ki]["bsc_dimension"]
            for j, kj in enumerate(kpi_names):
                if i == j:
                    row.append(1.0)
                else:
                    dim_j = role_config["kpis"][kj]["bsc_dimension"]
                    diff = dim_priority.get(dim_i, 2) - dim_priority.get(dim_j, 2)
                    if diff > 1:
                        row.append(5.0)
                    elif diff == 1:
                        row.append(3.0)
                    elif diff == 0:
                        row.append(1.0)
                    elif diff == -1:
                        row.append(1/3.0)
                    else:
                        row.append(1/5.0)
            M.append(row)

        result = self.compute_weights(M)
        result["kpi_weights"] = {
            kpi_names[i]: result["weights"][i] for i in range(n)
        }
        return result

    def recalibrate(self, new_comparisons, names):
        """
        Recalibrate weights from manager-provided pairwise comparisons.
        
        Args:
            new_comparisons: flat list of upper-triangle comparisons
                             e.g., [3, 5, 1/3] for 3 items means:
                             M[0][1]=3, M[0][2]=5, M[1][2]=1/3
            names: list of item names

        Returns:
            dict with weights and consistency
        """
        n = len(names)
        M = np.ones((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                M[i][j] = new_comparisons[idx]
                M[j][i] = 1.0 / new_comparisons[idx]
                idx += 1

        result = self.compute_weights(M.tolist())
        result["item_weights"] = {
            names[i]: result["weights"][i] for i in range(n)
        }
        return result
