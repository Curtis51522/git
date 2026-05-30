"""
Fusion Module -- deterministic cross-module business logic (zero LLM tokens).

Each intent has its own compute method that takes raw data from S1/S2/S3
and produces the decision-relevant numbers.
"""

from collections import defaultdict


class FusionModule:
    # ------------------------------------------------------------------
    # stock_query: restock = max(forecast - inventory, 0), capped by capacity
    # ------------------------------------------------------------------
    def compute_restock(self, forecast: float, inventory: int, capacity: int,
                       user_target: int = None, query_type: str = "forecast_query",
                       forecast_low: float = None, forecast_high: float = None,
                       carryover: int = 0) -> dict:
        """Compute restock recommendation using S2 forecast interval.
        
        Strategy: stock to upper bound (conservative), subtract current inventory.
        carryover: units carried over from previous day's production.
        """
        low = forecast_low if forecast_low is not None else forecast
        high = forecast_high if forecast_high is not None else forecast
        
        # Conservative: use upper bound for stocking
        # Note: router already sets inventory=carryover for follow-up queries
        needed = max(int(high - inventory + 0.5), 0)
        feasible = min(needed, capacity)

        result = {
            "forecast": forecast,
            "forecast_low": low,
            "forecast_high": high,
            "inventory": inventory,
            "capacity": capacity,
            "recommended_restock": feasible,
            "status": "ok" if feasible >= needed else "capacity_limited",
        }
        if carryover > 0:
            result["carryover"] = carryover

        # ---- User target comparison ----
        if user_target is not None:
            result["user_target"] = user_target
            result["query_type"] = query_type

            # Check capacity
            if user_target > capacity:
                result["user_target_feasible"] = False
                result["user_target_blocker"] = "capacity"
                result["user_target_gap"] = user_target - capacity
                result["status"] = "capacity_limited"
            # Check forecast alignment
            elif user_target > forecast + inventory:
                surplus = user_target - forecast - inventory
                result["user_target_feasible"] = True
                result["user_target_warning"] = (
                    f"Target {user_target} exceeds forecast ({forecast}) + inventory ({inventory}) "
                    f"by {surplus}. Risk of overproduction."
                )
                result["surplus_risk"] = surplus
                if surplus > forecast * 0.3:
                    result["user_target_risk"] = "high"
                elif surplus > forecast * 0.1:
                    result["user_target_risk"] = "medium"
                else:
                    result["user_target_risk"] = "low"
            elif user_target < needed:
                result["user_target_feasible"] = True
                result["user_target_warning"] = (
                    f"Target {user_target} is below recommended {needed}. "
                    f"May not meet forecasted demand."
                )
                result["shortfall"] = needed - user_target
            else:
                result["user_target_feasible"] = True
                result["user_target_warning"] = None

            # Override recommended_restock to show user's target vs system recommendation
            result["system_recommendation"] = feasible
            result["recommended_restock"] = user_target  # show user what they asked

        return result


    # ------------------------------------------------------------------
    # profit_analysis: total revenue, cost, and profit from transactions
    # ------------------------------------------------------------------
    def compute_profit(self, transactions: list, products: dict = None) -> dict:
        """Calculate profit breakdown from inventory_transactions.
        
        transactions: list of outflow records from inventory_transactions
        products: dict of {product_name: {selling_price, cost_price}} from products table
        """
        from collections import defaultdict
        if not transactions:
            return {"total_revenue": 0, "total_cost": 0, "gross_profit": 0,
                    "margin_pct": 0, "by_product": [], "transaction_count": 0}
        
        products = products or {}
        by_product = defaultdict(lambda: {"revenue": 0.0, "cost": 0.0, "quantity": 0, "transactions": 0})
        total_revenue = 0.0
        total_cost = 0.0
        
        for txn in transactions:
            pn = txn.get("product_name", "")
            qty = txn.get("quantity", 0)
            price = float(txn.get("unit_price", 0) or 0)
            revenue = price * qty
            # Cost: prefer from products table, fallback to price * 0.3
            prod_info = products.get(pn, {})
            cost_price = float(prod_info.get("cost_price", 0) or 0)
            if cost_price <= 0:
                cost_price = price * 0.3  # fallback: 30% cost ratio
            cost = cost_price * qty
            
            by_product[pn]["revenue"] += revenue
            by_product[pn]["cost"] += cost
            by_product[pn]["quantity"] += qty
            by_product[pn]["transactions"] += 1
            total_revenue += revenue
            total_cost += cost
        
        gross_profit = total_revenue - total_cost
        margin_pct = round((gross_profit / total_revenue * 100), 1) if total_revenue > 0 else 0.0
        
        # Sort products by profit descending
        product_list = []
        for pn, data in sorted(by_product.items(), key=lambda x: x[1]["revenue"] - x[1]["cost"], reverse=True):
            prod_profit = data["revenue"] - data["cost"]
            prod_margin = round((prod_profit / data["revenue"] * 100), 1) if data["revenue"] > 0 else 0.0
            product_list.append({
                "product_name": pn,
                "revenue": round(data["revenue"], 2),
                "cost": round(data["cost"], 2),
                "profit": round(prod_profit, 2),
                "margin_pct": prod_margin,
                "quantity_sold": data["quantity"],
                "transactions": data["transactions"],
            })
        
        return {
            "status": "ok",
            "total_revenue": round(total_revenue, 2),
            "total_cost": round(total_cost, 2),
            "gross_profit": round(gross_profit, 2),
            "margin_pct": margin_pct,
            "by_product": product_list,
            "transaction_count": len(transactions),
        }

    # ------------------------------------------------------------------
    # waste_analysis: deviation between forecast and actual
    # ------------------------------------------------------------------
    def compute_waste(self, predictions: list, actuals: list) -> dict:
        deviations = []
        for p, a in zip(predictions, actuals):
            dev = (a - p) / max(p, 1) * 100 if p > 0 else 0
            deviations.append(round(dev, 1))
        avg_dev = round(sum(deviations) / len(deviations), 1) if deviations else 0
        return {
            "deviations": deviations,
            "avg_deviation": avg_dev,
            "waste_flag": avg_dev > 30,
        }

    # ------------------------------------------------------------------
    # promo_eval: net profit from promotion
    # ------------------------------------------------------------------
    def compute_promo_roi(self, incremental_revenue: float, discount_cost: float) -> dict:
        net = incremental_revenue - discount_cost
        roi = (net / discount_cost * 100) if discount_cost > 0 else 0
        return {
            "net_profit": round(net, 2),
            "roi_percent": round(roi, 1),
            "recommendation": "continue" if net > 0 else "reconsider",
        }

    # ------------------------------------------------------------------
    # schedule_audit: detect under/over staffing per time slot
    # ------------------------------------------------------------------
    def compute_schedule_audit(
        self,
        schedule: list,           # [{"date":..., "time_slot":..., "headcount":...}, ...]
        transactions: list,       # [{"transaction_time":..., ...}, ...]
    ) -> dict:
        """Cross-reference S3 schedule headcount vs S1 transaction volume.

        Returns anomalies where headcount is too low for the transaction load.
        """
        if not schedule:
            return {"anomalies": [], "message": "No schedule data found for the requested date."}
        if not transactions:
            # Show schedule summary even without transaction cross-reference
            from collections import Counter
            dates = sorted(set(s.get("date", "") for s in schedule))
            roles = Counter(s.get("role", "") for s in schedule)
            emp_count = len(set(s.get("employee_name", "") for s in schedule))
            return {
                "anomalies": [],
                "message": f"Schedule covers {len(dates)} days ({len(schedule)} shifts, {emp_count} employees). No transaction data available for audit.",
                "schedule_summary": {
                    "days": dates,
                    "total_shifts": len(schedule),
                    "employees": emp_count,
                    "roles": dict(roles),
                },
                "schedule": schedule,
            }

    # ------------------------------------------------------------------
    # cross_source_audit: full-store health check across R6-R8
    # ------------------------------------------------------------------
    def compute_cross_audit(
        self,
        forecast: float,
        inventory: int,
        capacity: int,
        schedule: list,
        transactions: list,
    ) -> dict:
        """Run R6, R7, R8 checks across all available data and return a health report."""
        issues = []
        all_clear = []

        # R6: Forecast vs actual
        if forecast > 0:
            needed = max(int(forecast - inventory + 0.5), 0)
            if needed > capacity:
                issues.append({
                    "rule": "R6",
                    "severity": "high",
                    "message": f"Forecast demand ({needed} units) exceeds capacity ({capacity}). Risk of stockout."
                })
            elif needed > capacity * 0.9:
                issues.append({
                    "rule": "R6",
                    "severity": "medium",
                    "message": f"Forecast demand ({needed}) near capacity limit ({capacity}). Monitor closely."
                })
            else:
                all_clear.append("R6: Forecast-to-capacity ratio OK")

        # R7: Schedule vs transactions
        if schedule and transactions:
            audit_result = self.compute_schedule_audit(schedule, transactions) or {}
            anomalies = audit_result.get("anomalies", [])
            if anomalies:
                for a in anomalies[:5]:  # top 5
                    issues.append({
                        "rule": "R7",
                        "severity": a.get("severity", "medium"),
                        "message": f"{a['date']} {a['time_slot']}: {a['headcount']} staff for {a['transactions']} transactions"
                    })
            else:
                all_clear.append("R7: Schedule matches transaction demand")
        elif schedule:
            all_clear.append("R7: Schedule exists (no transaction data to cross-check)")

        # R8: Restock vs capacity (already covered in R6)

        # Summary
        if not issues:
            return {
                "status": "healthy",
                "issues": [],
                "all_clear": all_clear,
                "summary": f"All systems clear. {len(all_clear)} checks passed.",
                "issue_count": 0,
            }

        high = [i for i in issues if i["severity"] == "high"]
        medium = [i for i in issues if i["severity"] == "medium"]

        return {
            "status": "attention_needed" if high else "monitor",
            "issues": issues,
            "all_clear": all_clear,
            "high_count": len(high),
            "medium_count": len(medium),
            "summary": f"Found {len(issues)} issue(s): {len(high)} high, {len(medium)} medium.",
            "issue_count": len(issues),
        }

        # Count transactions per date + hour
        from datetime import datetime
        txn_counts = defaultdict(int)
        for txn in transactions:
            ts = txn.get("transaction_time", "")
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    key = dt.strftime("%Y-%m-%d") + "|" + f"{dt.hour:02d}:00"
                    txn_counts[key] += 1
            except Exception as e:
                            logger.warning("Failed to count transaction: %s", e)

        # Map time slots to hours
        slot_hours = {
            "07:00-10:00": [7, 8, 9],
            "10:00-14:00": [10, 11, 12, 13],
            "14:00-17:00": [14, 15, 16],
            "17:00-20:00": [17, 18, 19],
        }

        # Detect anomalies
        anomalies = []
        for s in schedule:
            date = s.get("schedule_date", s.get("date", ""))
            slot = s.get("time_slot", "")
            hc = s.get("headcount", s.get("staff_count", 0))

            hours = slot_hours.get(slot, [])
            total_txn = 0
            for h in hours:
                total_txn += txn_counts.get(f"{date}|{h:02d}:00", 0)

            # Heuristic: each staff can handle ~8 transactions per hour
            capacity = hc * 8 * len(hours)
            if total_txn > capacity and hc > 0:
                anomalies.append({
                    "date": date,
                    "time_slot": slot,
                    "headcount": hc,
                    "transactions": total_txn,
                    "capacity": capacity,
                    "severity": "high" if total_txn > capacity * 1.5 else "medium",
                })

        return {
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "message": (
                f"Found {len(anomalies)} scheduling anomalies"
                if anomalies else "No scheduling anomalies detected"
            ),
        }
