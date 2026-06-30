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

        # Theme definitions for the prompt
        theme_prompt = """You are a sharp bakery analyst. Agents below have analyzed individual business cards, then cross-referenced each other. Your output is a structured briefing """ + lang_instr + """$.

STRUCTURE YOUR SUMMARY BY BUSINESS THEMES (not by agent). Group findings into these thematic sections, each a flowing paragraph that synthesizes multiple agents:

Revenue & Demand -- Revenue, orders, average ticket, profit margin, and 7-day trend direction. Combine TrendAgent, DemandAgent, and ProfitAgent.

Product Mix & Risk -- Product concentration, category split (bread vs beverages), top-seller performance, and model sensitivity. Combine ProductMixAgent and FeatureSensitivityAgent.

Operations -- Staffing, attendance, wastage, and production. Combine StaffingAgent, AttendanceAgent, WastageAgent, and YieldAgent.

External Factors -- Weather, holidays, promotions, discounts, competitor activity. Combine ExternalFactorsAgent and PricingAgent.

Cross-Card Insights -- This is the most important section. MetricConflictAgent, CausalChainAgent, and CrossRiskAgent have already computed cross-references. Summarize their findings: conflicts, root cause chain, and cross-card risks.

RULES:
- Numbers only from agents. Currency: RMB yuan.
- Let the L2 agents do the heavy analytical lifting.
- Write as a sharp colleague briefing the store owner.
- Each section 2-4 sentences, flowing naturally.
- If a section has insufficient data, say so honestly in one sentence.
- Recommendations go in the separate JSON array, NOT in the summary text.

OUTPUT valid JSON:
{"summary": "Revenue & Demand\n...\n\nProduct Mix & Risk\n...\n\nOperations\n...\n\nExternal Factors\n...\n\nCross-Card Insights\n...",
 "attribution": [{"factor": "...", "evidence": "agent + number"}],
 "recommendations": [{"action": "...", "urgency": "high|medium|low", "rationale": "agent + number"}]}

Intent: """ + intent + """
Agent findings:
""" + chr(10).join(agent_summaries)

        try:
            async with httpx.AsyncClient(timeout=90) as client:
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
                        "summary": parsed.get("summary", ""),
                        "attribution": parsed.get("attribution", []),
                        "recommendations": parsed.get("recommendations", []),
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
            "summary": fallback_summary,
            "attribution": fallback_attribution,
            "recommendations": fallback_recs[:6],
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
