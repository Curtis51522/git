# s5_agent/core/memory.py
import json, os, time, logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("s5.memory")

class StructuredMemory:
    def __init__(self, storage_path: str = None):
        if storage_path is None:
            storage_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "s5_memory.json")
        self.storage_path = storage_path
        self.data = self._load()

    def _load(self) -> Dict:
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("Failed to load memory: %s", e)
        return {"entities": {}, "patterns": {}, "query_history": [], "version": 1}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error("Failed to save memory: %s", e)

    def get_entity(self, entity_type: str, entity_id: str) -> Dict:
        key = f"{entity_type}:{entity_id}"
        return self.data["entities"].get(key, {})

    def update_entity(self, entity_type: str, entity_id: str, updates: Dict):
        key = f"{entity_type}:{entity_id}"
        if key not in self.data["entities"]:
            self.data["entities"][key] = {}
        self.data["entities"][key].update(updates)
        self.data["entities"][key]["_last_updated"] = datetime.now().isoformat()
        self._save()

    def get_pattern(self, pattern_name: str) -> Dict:
        return self.data["patterns"].get(pattern_name, {})

    def update_pattern(self, pattern_name: str, updates: Dict):
        if pattern_name not in self.data["patterns"]:
            self.data["patterns"][pattern_name] = {}
        self.data["patterns"][pattern_name].update(updates)
        self.data["patterns"][pattern_name]["_last_updated"] = datetime.now().isoformat()
        self._save()

    def add_query(self, query: str, intent: str, summary: str, outcome: Dict):
        self.data["query_history"].append({
            "query": query, "intent": intent, "summary": summary,
            "outcome": outcome, "timestamp": datetime.now().isoformat()
        })
        if len(self.data["query_history"]) > 200:
            self.data["query_history"] = self.data["query_history"][-200:]
        self._save()

    def find_similar_queries(self, query: str, limit: int = 3) -> List[Dict]:
        history = self.data.get("query_history", [])
        if not history:
            return []
        recent = history[-50:]
        return [q for q in reversed(recent)][:limit]

    def get_conflict_precedent(self, agent_a: str, agent_b: str, conflict_type: str) -> Optional[Dict]:
        pattern_key = f"conflict:{agent_a}:{agent_b}:{conflict_type}"
        return self.data["patterns"].get(pattern_key)

    def record_conflict_ruling(self, agent_a: str, agent_b: str, conflict_type: str,
                               winner: str, rationale: str, confidence: float):
        pattern_key = f"conflict:{agent_a}:{agent_b}:{conflict_type}"
        history = self.data["patterns"].get(pattern_key, {}).get("rulings", [])
        history.append({
            "winner": winner, "rationale": rationale, "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        if len(history) > 20:
            history = history[-20:]
        self.update_pattern(pattern_key, {"rulings": history, "last_winner": winner})

    def prune(self, max_age_days: int = 90):
        cutoff = datetime.now() - timedelta(days=max_age_days)
        for key in list(self.data["entities"].keys()):
            lu = self.data["entities"][key].get("_last_updated", "")
            if lu:
                try:
                    if datetime.fromisoformat(lu) < cutoff:
                        del self.data["entities"][key]
                except ValueError:
                    pass
        self._save()

    def get_context_for_agent(self, agent_name: str, params: Dict) -> str:
        parts = []
        for entity_type, extract_key in [("material", "material_name"), ("product", "product_name")]:
            if extract_key in params:
                entity = self.get_entity(entity_type, params[extract_key])
                if entity:
                    anomalies = entity.get("past_anomalies", [])
                    if anomalies:
                        parts.append(f"Past {entity_type} anomalies: {anomalies[-3:]}")
        return "\n".join(parts) if parts else ""
