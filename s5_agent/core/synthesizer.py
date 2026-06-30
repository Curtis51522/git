# s5_agent/core/synthesizer.py
import logging, os, json, httpx
from typing import Dict, Any, List

logger = logging.getLogger("s5.synthesizer")

class Synthesizer:
    def __init__(self):
        self.llm_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") + "/chat/completions"
        self.llm_key = os.getenv("DEEPSEEK_API_KEY", "")

    async def synthesize(self, dag_result: Dict, deliberation: Dict = None,
                          memory = None) -> Dict[str, Any]:
        results = dag_result.get("results", {})
        query = dag_result.get("query", "")
        intent = dag_result.get("intent", "")

        summary = await self._generate_summary(results, intent)
        baseline = self._build_baseline(results)
        trend = self._build_trend(results, memory)
        attribution = self._build_attribution_table(results, deliberation)
        significance = self._check_significance(results)
        recommendations = self._build_recommendations(results)
        evidence = self._build_evidence(results)

        return {
            "summary": summary,
            "baseline": baseline,
            "trend": trend,
            "attribution": attribution,
            "significance": significance,
            "recommendations": recommendations,
            "evidence": evidence,
            "deliberation": deliberation,
            "query": query,
            "intent": intent,
        }

    async def _generate_summary(self, results: Dict, intent: str) -> str:
        agent_summaries = []
        for name, r in results.items():
            r = r if isinstance(r, dict) else r.__dict__
            opinion = r.get("opinion", "")
            if opinion:
                agent_summaries.append(f"[{name}]: {opinion}")

        if not self.llm_key:
            return " | ".join(agent_summaries)[:200]

        prompt = f"""You are a bakery operations analyst. Summarize the following agent findings into ONE sentence (<30 words in English) that captures the root cause chain.
Intent: {intent}
Agent findings:
{chr(10).join(agent_summaries)}
Return JSON: {{"summary": "one sentence here"}}"""
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(self.llm_url, json={
                    "model": "deepseek-chat", "temperature": 0.3,
                    "messages": [{"role": "user", "content": prompt}]
                }, headers={"Authorization": f"Bearer {self.llm_key}"})
                if r.status_code == 200:
                    return json.loads(r.json()["choices"][0]["message"]["content"]).get("summary", agent_summaries[0] if agent_summaries else "No findings")
        except Exception as e:
            logger.warning("LLM summary failed: %s", e)
        return agent_summaries[0] if agent_summaries else "Analysis complete"

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

    def _build_attribution_table(self, results: Dict, deliberation: Dict) -> List[Dict]:
        table = []
        profit_agent = results.get("ProfitAgent", {})
        profit_agent = profit_agent if isinstance(profit_agent, dict) else profit_agent.__dict__
        contributions = profit_agent.get("attribution", {}).get("contributions", {})

        for name, r in results.items():
            r = r if isinstance(r, dict) else r.__dict__
            attr = r.get("attribution", {})
            root_cause = attr.get("root_cause", "")
            if root_cause:
                table.append({
                    "factor": root_cause,
                    "contribution_pct": contributions.get(name, 0) or attr.get("contribution_pct", 0),
                    "confidence": r.get("confidence", 0),
                    "source_agent": name,
                })

        table.sort(key=lambda x: -x["contribution_pct"])
        if deliberation and deliberation.get("resolved") is False:
            minority = deliberation.get("minority_opinion", {})
            if minority:
                table.append({
                    "factor": f"[Alternative] {minority.get('attribution',{}).get('root_cause','')}",
                    "contribution_pct": 0, "confidence": deliberation.get("confidence", 0),
                    "source_agent": "deliberation_minority",
                })
        return table

    def _check_significance(self, results: Dict) -> Dict:
        significant = False
        factors = []
        for name, r in results.items():
            r = r if isinstance(r, dict) else r.__dict__
            deviation = r.get("attribution", {}).get("deviation")
            if deviation and abs(deviation) > 15:
                significant = True
                factors.append(name)
        return {"significant": significant, "flagged_agents": factors,
                "message": "Action recommended" if significant else "Normal fluctuation, no action needed"}

    def _build_recommendations(self, results: Dict) -> List[Dict]:
        recs = []
        for name, r in results.items():
            r = r if isinstance(r, dict) else r.__dict__
            for rec in r.get("recommendations", []):
                recs.append({**rec, "source_agent": name})
        recs.sort(key=lambda x: -(x.get("projected_gain", 0) * {"high": 3, "medium": 2, "low": 1}.get(x.get("ease","medium"), 2)))
        return recs[:5]

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
