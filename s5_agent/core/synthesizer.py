# s5_agent/core/synthesizer.py

import logging, os, json, httpx

from typing import Dict, Any, List



logger = logging.getLogger("s5.synthesizer")



class Synthesizer:

    def __init__(self):

        self.llm_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") + "/chat/completions"

        self.llm_key = os.getenv("DEEPSEEK_API_KEY", "")



    async def synthesize(self, dag_result: Dict, deliberation: Dict = None,

                          memory = None, lang: str = "en") -> Dict[str, Any]:

        results = dag_result.get("results", {})

        query = dag_result.get("query", "")

        intent = dag_result.get("intent", "")



        analysis = await self._llm_analyze(results, intent, deliberation, lang=lang)

        baseline = self._build_baseline(results)

        trend = self._build_trend(results, memory)

        significance = self._check_significance(results, lang=lang)

        evidence = self._build_evidence(results)



        return {

            "summary": analysis.get("summary", ""),

            "baseline": baseline,

            "trend": trend,

            "attribution": analysis.get("attribution", []),

            "significance": significance,

            "recommendations": analysis.get("recommendations", []),

            "evidence": evidence,

            "deliberation": deliberation,

            "query": query,

            "intent": intent,

        }



    async def _llm_analyze(self, results: Dict, intent: str, deliberation: Dict = None, lang: str = "en") -> Dict[str, Any]:

        agent_summaries = []

        agent_recs = []

        for name, r in results.items():

            r = r if isinstance(r, dict) else r.__dict__

            opinion = r.get("opinion", "")

            if opinion:

                agent_summaries.append(f"[{name}]: {opinion}")

            for rec in r.get("recommendations", []):

                agent_recs.append(f"[{name}] {rec.get('action', '')} (urgency={rec.get('urgency', 'medium')}, rationale={rec.get('rationale', '')})")



        if not self.llm_key:

            return {"summary": " | ".join(agent_summaries)[:200], "attribution": [], "recommendations": []}



        lang_instr = "in Chinese (Simplified Chinese)" if lang == "zh" else "in English"

        if lang == "bm":

            lang_instr = "in Bahasa Malaysia"



        # Theme definitions for the prompt - varies by intent
        if intent == "production_advice":
            theme_prompt = "You are a sharp bakery production planner. Agents below have analyzed the 7-day demand forecast, production plan, material needs, and forecast reliability. Your output is a structured briefing " + lang_instr + ".\n\nWrite a single flowing narrative that tells the full story of the upcoming 7-day production window. No section headers, no bullet points, no labels. Start with the most important finding.\n\nWeave these themes together seamlessly:\n- 7-day demand outlook: total projected revenue, trend direction, peak/valley days, top products (from ForecastOverviewAgent).\n- Uncertainty hotspots: which products have widest demand ranges, where the risk lies (from ForecastUncertaintyAgent).\n- Production plan: total bake, revenue, profit, buffer used. Is the plan aligned with demand? (from ProductionPlanAgent).\n- Material readiness: ALWAYS list the top 5 materials by weekly consumption with exact need/stock/gap numbers. Then note any critical/low stock alerts and total to order. Use the Top consumption line verbatim from MaterialProcurementAgent.\n- Forecast reliability: how well have past forecasts matched actual sales? If no data yet, say so and explain what it means for planning confidence (from ForecastAccuracyAgent).\n- Cross-checks: plan feasibility (PlanFeasibilityAgent), demand concentration risks (DemandRiskAgent), over/under-baking assessment (EfficiencyAgent), worst-case wastage scenarios (WastageRiskAgent).\n\nThe reader is the bakery owner planning the next week's production and procurement. Give them actionable clarity.\n\nRULES:\n- Numbers only from agents. Currency: RMB yuan. NEVER approximate. Use exact agent values.\n- If the production plan bakes less than forecast demand, check whether starting stock (day1_stock) covers the gap before calling it lost revenue. Total available = bake + starting stock. Only the portion exceeding total available is truly unserved demand.\n- Distinguish clearly between the production plan buffer setting and the actual supply coverage ratio (bake+stock divided by forecast). Never conflate the two.\n- The production plan allocates differently across products based on uncertainty; do not say \"overbakes\" or \"underbakes\" \u2014 say \"bakes X vs Y forecast\" instead.\n- FORBIDDEN words (will be filtered): 30-day, rolling average, predictor, model, self-reinforcing, concentration-lag, momentum-driven, WAPE, interval coverage, prediction interval, Q10, Q50. Use plain business words instead.\n- Let the L2 agents do the heavy analytical lifting.\n- If a section has insufficient data, say so honestly in one sentence.\n- Use agent recommendations below as your primary source \u2014 deduplicate, combine overlapping ones, and rank by urgency. Do NOT invent new products, quantities, or time frames that are not in the agent recommendations. You may only rephrase and merge them.\n- Recommendations: Each MUST include a time anchor (For tomorrow/This week/Over the next 30 days). Each MUST name specific products or numbers from agents, never generic. Each SHOULD include expected impact. Recommendations go in the separate JSON array, NOT in the summary text.\n- Output MUST be valid JSON only, no markdown, no explanatory text outside the JSON.\n\nOUTPUT valid JSON:\n{\"summary\": \"...\",\n \"attribution\": [{\"factor\": \"...\", \"evidence\": \"agent + number\"}],\n \"recommendations\": [{\"action\": \"...\", \"urgency\": \"high|medium|low\", \"time_horizon\": \"today|tomorrow|this_week|next_7_days|next_30_days|ongoing\", \"rationale\": \"...\", \"expected_impact\": \"...\"}]}\n"
        elif intent == "wastage_root_cause":
            theme_prompt = "You are a bakery kitchen manager reviewing today's wastage check with 14-day trend context. Agents below have analyzed material consumption, wastage trends, stock levels, and production yield. Write a plain-language briefing " + lang_instr + ".\n\nSingle flowing narrative, no section headers, no bullets. Start with the most important finding — any wastage rising? Any patterns breaking?\n\nWeave these together:\n- Rising wastage: which materials show increasing waste vs their 7-day average? Name them, give today's rate vs average, yuan cost. Include the agent's likely root cause suggestion.\n- Stable wastage: mention that other materials are within normal thresholds to give a complete picture.\n- Total waste cost: the yuan figure across all materials today. If it's negligible, say so and confirm inventory is clean.\n- Production context: what did we bake today, how many units, any yield issues?\n- Stock check: any materials running low as a result?\n- Overall verdict: is wastage under control, or do we have an emerging problem?\n\nRULES:\n- Numbers from agents only. RMB yuan. No approximation.\n- If no wastage check submitted today, say so clearly and when the last check was.\n- Always include yuan cost of waste.\n- FORBIDDEN words: 30-day, rolling average, predictor, model, self-reinforcing, concentration-lag, momentum-driven. Use plain kitchen words.\n- Recommendations from agent recommendations below. Deduplicate.\n- Valid JSON only.\n\nOUTPUT: {\"summary\": \"...\", \"recommendations\": [{\"action\": \"...\", \"urgency\": \"high|medium|low\", \"time_horizon\": \"today|tomorrow|this_week|ongoing\", \"rationale\": \"...\", \"expected_impact\": \"...\"}]}\n"
        else:
            theme_prompt = "You are a sharp bakery analyst. Agents below have analyzed individual business cards, then cross-referenced each other. Your output is a structured briefing " + lang_instr + ".\n\nWrite a single flowing narrative that tells the full story of today's performance \u2014 no section headers, no bullet points, no labels. Start with the most important finding and let each sentence naturally lead to the next.\n\nWeave these themes together seamlessly:\n- Revenue, orders, ATV, and 7-day trend direction (from TrendAgent, DemandAgent, ProfitAgent). Always compare orders vs revenue movement \u2014 if they diverge, explain the ATV shift.\n- Product concentration, category split, top-seller performance, and what drives demand (from ProductMixAgent, FeatureSensitivityAgent). Use business language, never mention model names (XGBoost), feature importance percentages, or technical parameters in the summary. Translate model insights into plain cause-and-effect statements the store owner can act on.\n- Operations: staffing, attendance, wastage, production yield (from StaffingAgent, AttendanceAgent, WastageAgent, YieldAgent) \u2014 weave in where relevant, not as a standalone section.\n- External context: weather, holidays, promos, discounts, competitor activity (from ExternalFactorsAgent, PricingAgent).\n- Cross-card conclusions: MetricConflictAgent divergences, CausalChainAgent root cause chain, CrossRiskAgent self-reinforcing loops. Do NOT expose agent-level disagreements. Synthesize into one reconciled insight.\n\nThe reader should feel like they're reading a colleague's update, not filling out a template.\n\nRULES:\n- Numbers only from agents. Currency: RMB yuan. NEVER approximate: if an agent says 2490, do NOT write ~2500 or ~3158. Use the exact agent number. If you must compute a derived number, show both the agent values and the result.\n- Let the L2 agents do the heavy analytical lifting.\n- Write as a sharp colleague briefing the store owner.\n- If a section has insufficient data, say so honestly in one sentence.\n- Use agent recommendations below as your primary source \u2014 deduplicate, combine overlapping ones, and rank by urgency. Do NOT invent new products, quantities, or time frames that are not in the agent recommendations. You may only rephrase and merge them.\n- Recommendations: Each MUST include a time anchor (For tomorrow/This week/Over the next 30 days). Each MUST name specific products or numbers from agents, never generic. Each SHOULD include expected impact when computable. Recommendations go in the separate JSON array, NOT in the summary text.\n\nOUTPUT valid JSON:\n{\"summary\": \"Today's revenue was \\u00a5... from ... orders. ...\",\n \"attribution\": [{\"factor\": \"...\", \"evidence\": \"agent + number\"}],\n \"recommendations\": [{\"action\": \"...\", \"urgency\": \"high|medium|low\", \"time_horizon\": \"today|tomorrow|this_week|next_7_days|next_30_days|ongoing\", \"rationale\": \"...\", \"expected_impact\": \"...\"}]}\n"

        recs_text = '\n\nAgent recommendations (PRIMARY SOURCE \u2014 deduplicate and combine these):\n\n' + chr(10).join(agent_recs) if agent_recs else ''
        theme_prompt += '\n\n\n\nIntent: ' + intent + '\n\nAgent findings:\n\n' + chr(10).join(agent_summaries) + recs_text



        try:

            async with httpx.AsyncClient(timeout=180) as client:

                r = await client.post(self.llm_url, json={

                    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), "temperature": 0.2,

                    "messages": [{"role": "user", "content": theme_prompt}]

                }, headers={"Authorization": f"Bearer {self.llm_key}"})

                if r.status_code == 200:

                    raw = r.json()["choices"][0]["message"]["content"].strip()
                    
                    if raw.startswith("```"):

                        first_nl = raw.find("\n")

                        if first_nl != -1:

                            raw = raw[first_nl + 1:]

                        else:

                            raw = raw[3:]

                    raw = raw.strip()

                    if raw.endswith("```"):

                        raw = raw[:-3].strip()

                    parsed = json.loads(raw)

                    return {

                        "summary": self._sanitize_summary(parsed.get("summary", "")),

                        "attribution": parsed.get("attribution", []),

                        "recommendations": self._sanitize_recommendations(parsed.get("recommendations", [])),

                    }

        except Exception as e:

            logger.warning("LLM analysis failed: %s", e)
            
        # Fallback

        fallback_summary = "; ".join(agent_summaries) if agent_summaries else "Analysis complete"

        fallback_attribution = []

        for name, r in results.items():

            r = r if isinstance(r, dict) else r.__dict__

            opinion = r.get("opinion", "")

            if opinion:

                fallback_attribution.append({"factor": name, "evidence": opinion[:200]})

        fallback_recs = []

        for rec_item in agent_recs:

            fallback_recs.append({"action": rec_item, "urgency": "medium", "rationale": ""})

        return {

            "summary": self._sanitize_summary(fallback_summary),

            "attribution": fallback_attribution,

            "recommendations": self._sanitize_recommendations(fallback_recs[:6]),

        }



    def _build_baseline(self, results: Dict) -> Dict:

        baseline = {"actual": {}, "expected": {}, "anomaly": {}}

        for name, r in results.items():

            r = r if isinstance(r, dict) else r.__dict__

            attr = r.get("attribution", {})

            metric = attr.get("metric", "")

            deviation = attr.get("deviation")

            if metric and deviation is not None:

                baseline["actual"][metric] = deviation

                baseline["anomaly"][metric] = deviation

        return baseline



    def _build_trend(self, results: Dict, memory) -> Dict:

        if not memory:

            return {"direction": "unknown", "weeks": []}

        history = memory.data.get("query_history", [])

        recent = [h for h in history[-20:] if h.get("intent") in ("profit_root_cause","full_diagnosis")]

        anomaly_count = sum(1 for h in recent if h.get("outcome",{}).get("significance",{}).get("significant", False))

        if len(recent) >= 3:

            mid = len(recent) // 2

            first_half = sum(1 for h in recent[:mid] if h.get("outcome",{}).get("significance",{}).get("significant", False))

            second_half = sum(1 for h in recent[mid:] if h.get("outcome",{}).get("significance",{}).get("significant", False))

            if second_half > first_half:

                direction = "worsening"

            elif second_half < first_half:

                direction = "improving"

            else:

                direction = "stable"

        else:

            direction = "insufficient_data"

        return {"direction": direction, "recent_anomalies": anomaly_count, "periods": len(recent)}



    def _check_significance(self, results: Dict, lang: str = "en") -> Dict:

        significant = False

        factors = []

        for name, r in results.items():

            r = r if isinstance(r, dict) else r.__dict__

            deviation = r.get("attribution", {}).get("deviation")

            if deviation and abs(deviation) > 15:

                significant = True

                factors.append(name)

        msgs = {

            "en": {"sig": "Action recommended", "norm": "Normal fluctuation, no action needed"},

            "zh": {"sig": "??????", "norm": "?????????"},

            "bm": {"sig": "Tindakan disyorkan", "norm": "Turun naik normal, tiada tindakan diperlukan"},

        }

        lm = msgs.get(lang, msgs["en"])

        return {"significant": significant, "flagged_agents": factors,

                "message": lm["sig"] if significant else lm["norm"]}



    def _build_evidence(self, results: Dict) -> Dict:

        evidence = {}

        for name, r in results.items():

            r = r if isinstance(r, dict) else r.__dict__

            evidence[name] = {

                "opinion": r.get("opinion", ""),

                "confidence": r.get("confidence", 0),

                "data_sources": r.get("data", {}),

                "elapsed_ms": r.get("elapsed_ms", 0),

            }

        return evidence



    def _sanitize_summary(self, text):
        """Replace forbidden technical terms with plain business language."""
        # Normalize Unicode: replace fancy dashes/hyphens with ASCII hyphen
        for ch in ['‐', '‑', '‒', '–', '—', '−']:
            text = text.replace(ch, '-')
        reps = [
            ("30-day rolling average", "recent sales pattern"),
            ("30-day rolling sales average", "recent sales pattern"),
            ("30-day average", "recent trend"),
            ("30-day rolling", "recent"),
            ("rolling average", "recent trend"),
            ("30-day", "recent"),
            ("self-reinforce into", "feed on itself into"),
            ("self-reinforce", "feed on itself"),
            ("self-reinforcing loop", "a cycle where fewer sales lead to even fewer sales"),
            ("self-reinforcing decline loop", "over-dependence risk"),
            ("self-reinforcing decline", "a downward spiral"),
            ("self-reinforcing risk", "over-dependence risk"),
            ("self-reinforcing cycle", "a cycle where less selling leads to even less"),
            ("self-reinforcing", "a repeating pattern"),
            ("self-perpetuating", "a repeating pattern"),
            ("concentration-lag loop", "over-dependence on a few products"),
            ("concentration-lag cycle", "over-dependence on a few products"),
            ("concentration loop", "over-dependence on a few products"),
            ("momentum-driven", "driven by habit"),
            ("lag-driven", "history-driven"),
            ("predictor", "driver"),
            ("demand model", "sales patterns"),
            ("model projections", "demand outlook"),
            ("model predicts", "the pattern suggests"),
            ("the model", "the pattern"),
            (" model", " pattern"),
            ("prediction interval", "demand range"),
            ("prediction intervals", "demand ranges"),
            ("interval coverage", "forecast accuracy"),
            ("WAPE", "forecast error"),
            ("wape", "forecast error"),
            ("Q50 scenario", "higher-demand scenario"),
            ("Q10 scenario", "lower-demand scenario"),
            ("Q50 trajectory", "higher-demand trajectory"),
            ("Q10 trajectory", "lower-demand trajectory"),
            ("Q50", "higher-demand"),
            ("Q10", "lower-demand"),
        ]
        for old_t, new_t in reps:
            text = text.replace(old_t, new_t)
        import re
        text = re.sub(r'\d+\.?\d*% of (the )?(prediction|demand|outcome|variance)', 'the single biggest factor', text)
        text = re.sub(r'accounts? for \d+\.?\d*% of', 'is the main driver of', text)
        text = re.sub(r'(carry|carries|hold|holds) \d+\.?\d*%[^.]*?(influence|weight)', 'are the strongest factors', text)
        text = re.sub(r'(weights|weighted) at \d+\.?\d*%', 'is the main driver', text)
        text = re.sub(r'exerting \d+\.?\d*% of the pull', 'being the main driver', text)
        text = re.sub(r'a secondary \d+\.?\d*%', 'another factor', text)
        # Clean up grammar artifacts from replacements
        text = re.sub(r'\ba a\b', 'a', text)
        text = text.replace('pattern of sales', 'pattern')
        text = text.replace('pattern sales', 'patterns')
        text = text.replace('driven by driven by', 'driven by')
        text = text.replace('driven by habit by', 'driven by')
        text = re.sub(r'\ba (\w+), a (\w)', r'a \1, \2', text)
        text = text.replace('a over-', 'an over-')
        return text

    def _sanitize_recommendations(self, recs):
        """Apply sanitizer to all text fields in recommendations."""
        clean = []
        for r in recs:
            if isinstance(r, dict):
                clean.append({
                    k: self._sanitize_summary(v) if isinstance(v, str) and k in ('action', 'reason', 'impact', 'rationale', 'expected_impact') else v
                    for k, v in r.items()
                })
            else:
                clean.append(r)
        return clean
