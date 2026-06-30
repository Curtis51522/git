# s5_agent/core/deliberator.py
import logging, os, json, httpx
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger("s5.deliberator")

class Deliberator:
    def __init__(self, memory=None):
        self.memory = memory
        self.max_rounds = 5
        self.convergence_threshold = 0.05
        self.confidence_gap_threshold = 0.15
        self.llm_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") + "/chat/completions"
        self.llm_key = os.getenv("DEEPSEEK_API_KEY", "")

    def detect_conflict(self, results: Dict[str, Any]) -> List[Tuple[str, str, Dict, Dict]]:
        """Detect conflicting attributions among agents. Returns [(agent_a, agent_b, opinion_a, opinion_b)]."""
        conflicts = []
        agent_names = list(results.keys())
        for i in range(len(agent_names)):
            for j in range(i+1, len(agent_names)):
                a, b = agent_names[i], agent_names[j]
                ra = results[a] if isinstance(results[a], dict) else results[a].__dict__
                rb = results[b] if isinstance(results[b], dict) else results[b].__dict__
                attr_a = ra.get("attribution", {})
                attr_b = rb.get("attribution", {})
                if attr_a and attr_b:
                    metric_a = attr_a.get("metric", "")
                    metric_b = attr_b.get("metric", "")
                    root_a = attr_a.get("root_cause", "")
                    root_b = attr_b.get("root_cause", "")
                    if metric_a == metric_b and root_a and root_b and root_a != root_b:
                        conflicts.append((a, b, ra, rb))
        return conflicts

    async def _llm_judge(self, prompt: str) -> str:
        if not self.llm_key:
            return json.dumps({"classification": "conflict", "reason": "no LLM available, defaulting to conflict"})
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(self.llm_url, json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                }, headers={"Authorization": f"Bearer {self.llm_key}"})
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                return json.dumps({"classification": "conflict", "reason": f"LLM error: {r.status_code}"})
        except Exception as e:
            return json.dumps({"classification": "conflict", "reason": str(e)})

    async def classify_agreement(self, agent_a: str, agent_b: str,
                                  opinion_a: Dict, opinion_b: Dict) -> str:
        prompt = f"""Classify the relationship between these two agent opinions:
Agent {agent_a}: root_cause={opinion_a.get('attribution',{}).get('root_cause','')}, metric={opinion_a.get('attribution',{}).get('metric','')}
Agent {agent_b}: root_cause={opinion_b.get('attribution',{}).get('root_cause','')}, metric={opinion_b.get('attribution',{}).get('metric','')}
Return JSON: {{"classification": "agree|complementary|conflict", "reason": "brief explanation"}}"""
        result = await self._llm_judge(prompt)
        try:
            return json.loads(result).get("classification", "conflict")
        except json.JSONDecodeError:
            return "conflict"

    async def deliberate(self, agent_a: str, agent_b: str,
                          opinion_a: Dict, opinion_b: Dict,
                          conflict_type: str = "") -> Dict[str, Any]:
        if self.memory:
            precedent = self.memory.get_conflict_precedent(agent_a, agent_b, conflict_type)
        else:
            precedent = None

        round_log = []
        conf_a = opinion_a.get("confidence", 0.5)
        conf_b = opinion_b.get("confidence", 0.5)

        for rnd in range(1, self.max_rounds + 1):
            prev_a, prev_b = conf_a, conf_b

            prompt = self._build_deliberation_prompt(agent_a, agent_b, opinion_a, opinion_b,
                                                      round_log, rnd, precedent)
            response = await self._llm_judge(prompt)
            try:
                ruling = json.loads(response)
            except json.JSONDecodeError:
                ruling = {"phase": "ongoing", "confidence_a": conf_a, "confidence_b": conf_b}

            round_log.append({"round": rnd, "ruling": ruling})
            conf_a = ruling.get("confidence_a", conf_a)
            conf_b = ruling.get("confidence_b", conf_b)

            if ruling.get("phase") == "ruling":
                if self.memory:
                    self.memory.record_conflict_ruling(
                        agent_a, agent_b, conflict_type,
                        ruling.get("winner", ""), ruling.get("rationale", ""),
                        ruling.get("confidence", 0.5)
                    )
                return {
                    "resolved": True, "winner": ruling.get("winner"),
                    "rationale": ruling.get("rationale", ""),
                    "confidence": ruling.get("confidence", 0.5),
                    "minority_opinion": opinion_b if ruling.get("winner") == agent_a else opinion_a,
                    "rounds": rnd, "log": round_log,
                }

            delta_a = abs(conf_a - prev_a)
            delta_b = abs(conf_b - prev_b)
            if max(delta_a, delta_b) < self.convergence_threshold:
                winner = agent_a if conf_a >= conf_b else agent_b
                rationale = f"Stagnation at round {rnd}: Δconfidence < {self.convergence_threshold}"
                if self.memory:
                    self.memory.record_conflict_ruling(agent_a, agent_b, conflict_type, winner, rationale, max(conf_a, conf_b))
                return {
                    "resolved": True, "winner": winner, "rationale": rationale,
                    "confidence": max(conf_a, conf_b),
                    "minority_opinion": opinion_b if winner == agent_a else opinion_a,
                    "rounds": rnd, "log": round_log,
                }

        winner = agent_a if conf_a >= conf_b else agent_b
        return {
            "resolved": False, "winner": winner,
            "rationale": f"Max rounds ({self.max_rounds}) reached without clear convergence",
            "confidence": max(conf_a, conf_b),
            "minority_opinion": opinion_b if winner == agent_a else opinion_a,
            "rounds": self.max_rounds, "log": round_log,
        }

    def _build_deliberation_prompt(self, agent_a, agent_b, opinion_a, opinion_b,
                                    round_log, rnd, precedent) -> str:
        precedent_text = ""
        if precedent:
            precedent_text = f"Precedent: {json.dumps(precedent, ensure_ascii=False)}"
        return f"""You are a judge resolving a disagreement between two AI agents analyzing a bakery operation.

Agent {agent_a} says: {json.dumps(opinion_a.get('attribution',{}), ensure_ascii=False)}
Evidence: {json.dumps(opinion_a.get('evidence',{}), ensure_ascii=False)[:500]}

Agent {agent_b} says: {json.dumps(opinion_b.get('attribution',{}), ensure_ascii=False)}
Evidence: {json.dumps(opinion_b.get('evidence',{}), ensure_ascii=False)[:500]}

{precedent_text}
Deliberation history: {json.dumps(round_log, ensure_ascii=False)}
Current round: {rnd}/{self.max_rounds}

Return JSON:
If you can rule now: {{"phase":"ruling","winner":"agent_name","rationale":"why","confidence":0.0-1.0,"confidence_a":float,"confidence_b":float}}
If you need more cross-examination: {{"phase":"ongoing","question_to_a":"question","question_to_b":"question","confidence_a":float,"confidence_b":float}}"""
