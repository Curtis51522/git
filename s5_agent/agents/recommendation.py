import re, logging
from s5_agent.core.base import BaseAgent, AgentOpinion
logger = logging.getLogger("s5.agent.recommendation")

class RecommendationAgent(BaseAgent):
    """Synthesizes upstream agent findings into actionable Top-3 bundle priorities.
    
    Reads: ProductMixAgent (concentration), WastageAgent (Day-1 stock), 
           RevenueTrendAgent (momentum), ProfitAgent (margins), and module context.
    Outputs: Priority product+coffee pairs with boost multipliers for /s4/combo.
    """
    
    def analyze(self, raw, params, context="", history="", key_metrics=None):
        if not context:
            return AgentOpinion(agent=self.name,
                opinion="No upstream context for recommendation synthesis.",
                confidence=0.3)
        
        priorities = []
        
        # Parse upstream signals
        signals = self._parse_signals(context)
        
        # Rule 1: Day-1 stock pressure -> push clearance combos
        day1_items = signals.get("day1_products", [])
        if day1_items:
            for item in day1_items[:3]:
                coffee = self._best_coffee_for(item, signals)
                priorities.append({
                    "product": item,
                    "coffee": coffee,
                    "reason": f"Day-1 clearance: {item} needs to move before expiry",
                    "boost": 2.5,
                    "strategy": "clearance"
                })
        
        # Rule 2: Rising momentum products -> ride the wave
        rising = signals.get("rising_products", [])
        for item in rising[:2]:
            if item not in [p["product"] for p in priorities]:
                coffee = self._best_coffee_for(item, signals)
                priorities.append({
                    "product": item,
                    "coffee": coffee,
                    "reason": f"Momentum: {item} volume rising, amplify with bundle",
                    "boost": 2.0,
                    "strategy": "amplify"
                })
        
        # Rule 3: High-margin pairings -> profit maximization
        high_margin = signals.get("high_margin_products", [])
        for item in high_margin[:2]:
            if item not in [p["product"] for p in priorities]:
                coffee = signals.get("high_margin_coffee", "cold_brew")
                priorities.append({
                    "product": item,
                    "coffee": coffee,
                    "reason": f"Margin play: {item} has strong profit margin, pair with {coffee}",
                    "boost": 1.8,
                    "strategy": "margin"
                })
        
        # Rule 4: Concentration risk -> diversify
        conc_risk = signals.get("concentration_risk_products", [])
        for item in conc_risk[:2]:
            if item not in [p["product"] for p in priorities]:
                coffee = self._best_coffee_for(item, signals)
                priorities.append({
                    "product": item,
                    "coffee": coffee,
                    "reason": f"Diversification: reduce reliance on {signals.get('top_seller', 'hero SKU')}",
                    "boost": 1.5,
                    "strategy": "diversify"
                })
        
        # Limit to 3
        priorities = priorities[:3]
        
        if not priorities:
            return AgentOpinion(agent=self.name,
                opinion="No priority recommendations generated - insufficient upstream signals.",
                confidence=0.4)
        
        # Build opinion text
        lines = [f"Priority bundle recommendations ({len(priorities)}):"]
        for i, p in enumerate(priorities):
            lines.append(f"  #{i+1}: {p['product']} + {p['coffee']} (boost={p['boost']}) - {p['reason']}")
        
        return AgentOpinion(
            agent=self.name,
            opinion="\n".join(lines),
            confidence=0.75,
            metadata={"priority_recommendations": priorities}
        )
    
    def _parse_signals(self, context: str) -> dict:
        """Extract actionable signals from upstream agent opinions."""
        signals = {
            "day1_products": [],
            "rising_products": [],
            "high_margin_products": [],
            "concentration_risk_products": [],
            "top_seller": "macaron",
            "high_margin_coffee": "cold_brew",
        }
        
        # Parse Day-1 stock from WastageAgent
        day1_match = re.findall(r"(\w+).*?Day-1.*?(\d+)\s*units", context, re.IGNORECASE)
        for match in day1_match:
            pn = match[0].lower().replace(" ", "_")
            signals["day1_products"].append(pn)
        
        # Parse rising products from upstream module context
        rising_match = re.findall(r"(\w+).*?\+(\d+)%.*?quantit", context, re.IGNORECASE)
        for match in rising_match:
            pct = int(match[1])
            if pct > 20:
                pn = match[0].lower().replace(" ", "_")
                signals["rising_products"].append(pn)
        
        # Parse concentration from ProductMixAgent
        conc_match = re.search(r"top\s*3.*?(\d+)%", context, re.IGNORECASE)
        if conc_match and int(conc_match.group(1)) > 70:
            # Extract product names after concentration mention
            after_conc = context[conc_match.end():conc_match.end()+300]
            products = re.findall(r"\b(macaron|melon_bread|croissant_chocolate|croissant|donut|baguette|sourdough|brioche|chiffon)\b", after_conc, re.IGNORECASE)
            signals["concentration_risk_products"] = [p.lower() for p in products[:3]]
        
        # Parse margins from ProfitAgent
        margin_match = re.search(r"margin.*?(\d+\.?\d*)\s*%", context, re.IGNORECASE)
        if margin_match:
            # Products with "profit" or "margin" nearby
            profit_section = context[max(0, context.find("profit")-200):context.find("profit")+500]
            margin_products = re.findall(r"\b(\w+_\w+|macaron|cold_brew|mocha|latte|cappuccino)\b.*?(?:margin|profit)", profit_section, re.IGNORECASE)
            signals["high_margin_products"] = [p.lower().replace(" ", "_") for p in margin_products[:3]]
        
        return signals
    
    def _best_coffee_for(self, product: str, signals: dict) -> str:
        """Pick best coffee pairing based on context."""
        # Simple heuristic: cold drinks for sweet breads, hot drinks for savory
        sweet_breads = {"donut", "chiffon", "croissant_chocolate", "macaron", "brownie", "chocopie", "cookie", "muffin", "pancake", "melon_bread", "chocolate_cake", "apple_pie", "cream_horn", "eggtart"}
        savory_breads = {"croissant", "baguette", "bagel", "bread_roll", "sourdough", "flatbread", "pullman", "pizza_bread", "brioche", "pandesal", "cornbread", "bread_coconut", "mantequilla", "stickbread", "soboru_bread", "tostada"}
        
        if product in sweet_breads:
            return "cold_brew"
        elif product in savory_breads:
            return "latte"
        return "americano"
