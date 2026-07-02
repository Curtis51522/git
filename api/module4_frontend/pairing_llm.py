"""
LLM-generated bread-coffee pairing matrix.

On first use, calls DeepSeek to score all 6 breads x 8 coffees based on
flavor profiles. Result is cached in memory for instant retrieval.

Fallback: hardcoded matrix if DeepSeek unavailable.
"""

import json
import logging
import httpx, os

def _call_deepseek(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") + "/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = httpx.post(url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"), "messages": messages,
              "max_tokens": max_tokens, "temperature": 0.3},
        timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

logger = logging.getLogger("s4.pairing")

# Cache
_pairing_cache: dict | None = None

def _load_bakery_products():
    """Load all bakery products from DB for pairing matrix generation."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from db.mysql_client import get_db
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT product_name FROM products WHERE category='bakery' ORDER BY product_name")
    products = []
    for r in cur.fetchall():
        key = r[0]
        name = key.replace('_', ' ').title()
        products.append({"key": key, "name": name, "desc": name})
    cur.close()
    db.close()
    return products

_BAKERY_CACHE = None
def _get_bakery():
    global _BAKERY_CACHE
    if _BAKERY_CACHE is None:
        _BAKERY_CACHE = _load_bakery_products()
    return _BAKERY_CACHE
BAKERY = None  # Use _get_bakery() instead

def _load_coffee_products():
    """Load all beverage products from DB for pairing matrix generation."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from db.mysql_client import get_db
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT product_name FROM products WHERE category='beverage' ORDER BY product_name")
    products = []
    for r in cur.fetchall():
        key = r[0]
        name = key.replace('_', ' ').title()
        products.append({"key": key, "name": name, "desc": name})
    cur.close()
    db.close()
    return products

_COFFEE_CACHE = None
def _get_coffee():
    global _COFFEE_CACHE
    if _COFFEE_CACHE is None:
        _COFFEE_CACHE = _load_coffee_products()
    return _COFFEE_CACHE
COFFEE = None  # Use _get_coffee() instead

# Hardcoded fallback matrix
def _build_fallback_matrix():
    """Build a knowledge-based pairing matrix using real food pairing principles.
    
    Scoring logic by category:
    - Sweet/dessert breads pair best with bold/bitter coffees and cold brew
    - Buttery/flaky breads pair best with milky coffees
    - Plain/bread-like items pair well with flavored/sweet coffees
    - Chocolate items pair best with cold brew, mocha, and milky coffees
    - Tea-based drinks pair well with subtle breads and pastries
    """
    # Category definitions
    SWEET = {"donut","brownie","chocolate_cake","chocopie","cookie","macaron",
             "apple_pie","cream_horn","eggtart","chiffon","pancake","muffin","melon_bread"}
    BUTTERY = {"croissant","croissant_chocolate","brioche","mantequilla"}
    PLAIN = {"baguette","sourdough","pullman","bagel","bread_roll","flatbread",
             "cornbread","pandesal","stickbread","soboru_bread","tostada","pizza_bread"}
    COCONUT = {"bread_coconut"}
    
    BOLD_COFFEE = {"espresso","americano"}
    MILKY_COFFEE = {"latte","cappuccino","flat_white"}
    SWEET_COFFEE = {"mocha","caramel_macchiato","hot_chocolate","chai_latte","matcha_latte","milk_tea"}
    COLD_LIGHT = {"cold_brew","lemonade"}
    TEA = {"earl_grey","english_breakfast"}
    
    def _pair_score(bread, coffee):
        s = 0.5
        cat_b = "plain"
        if bread in SWEET: cat_b = "sweet"
        elif bread in BUTTERY: cat_b = "buttery"
        elif bread in COCONUT: cat_b = "coconut"
        
        cat_c = "other"
        if coffee in BOLD_COFFEE: cat_c = "bold"
        elif coffee in MILKY_COFFEE: cat_c = "milky"
        elif coffee in SWEET_COFFEE: cat_c = "sweet"
        elif coffee in COLD_LIGHT: cat_c = "cold"
        elif coffee in TEA: cat_c = "tea"
        
        rules = {
            ("sweet","bold"): 0.85, ("sweet","milky"): 0.65,
            ("sweet","sweet"): 0.50, ("sweet","cold"): 0.75, ("sweet","tea"): 0.55,
            ("buttery","bold"): 0.60, ("buttery","milky"): 0.80,
            ("buttery","sweet"): 0.55, ("buttery","cold"): 0.50, ("buttery","tea"): 0.60,
            ("plain","bold"): 0.55, ("plain","milky"): 0.60,
            ("plain","sweet"): 0.75, ("plain","cold"): 0.50, ("plain","tea"): 0.65,
            ("coconut","bold"): 0.70, ("coconut","milky"): 0.75,
            ("coconut","sweet"): 0.60, ("coconut","cold"): 0.65, ("coconut","tea"): 0.55,
        }
        s = rules.get((cat_b, cat_c), 0.50)
        
        overrides = {
            ("croissant_chocolate","espresso"): 0.90, ("croissant_chocolate","americano"): 0.85,
            ("croissant_chocolate","cold_brew"): 0.85, ("croissant_chocolate","mocha"): 0.80,
            ("croissant_chocolate","latte"): 0.75,
            ("brownie","espresso"): 0.90, ("brownie","americano"): 0.85,
            ("brownie","cold_brew"): 0.80, ("brownie","mocha"): 0.85,
            ("chocolate_cake","mocha"): 0.90, ("chocolate_cake","espresso"): 0.85,
            ("chocolate_cake","latte"): 0.75,
            ("chocopie","mocha"): 0.85, ("chocopie","cold_brew"): 0.80,
            ("donut","americano"): 0.90, ("donut","espresso"): 0.85, ("donut","cold_brew"): 0.75,
            ("macaron","espresso"): 0.85, ("macaron","latte"): 0.80, ("macaron","earl_grey"): 0.80,
            ("apple_pie","americano"): 0.85, ("apple_pie","latte"): 0.75, ("apple_pie","chai_latte"): 0.80,
            ("bagel","latte"): 0.70, ("bagel","espresso"): 0.75,
            ("sourdough","espresso"): 0.75, ("sourdough","americano"): 0.75,
            ("pizza_bread","espresso"): 0.70, ("pizza_bread","americano"): 0.70,
            ("melon_bread","matcha_latte"): 0.80, ("melon_bread","latte"): 0.70,
            ("melon_bread","cold_brew"): 0.65,
            ("bread_coconut","latte"): 0.80, ("bread_coconut","cold_brew"): 0.75,
            ("bread_coconut","matcha_latte"): 0.75,
            ("lemonade","cookie"): 0.75, ("lemonade","donut"): 0.60,
            ("lemonade","brownie"): 0.55, ("lemonade","pandesal"): 0.60,
            ("lemonade","bread_coconut"): 0.70, ("lemonade","chiffon"): 0.70,
            ("hot_chocolate","cookie"): 0.85, ("hot_chocolate","brownie"): 0.80,
            ("hot_chocolate","croissant"): 0.75,
            ("chai_latte","apple_pie"): 0.80, ("chai_latte","donut"): 0.70,
            ("chai_latte","cookie"): 0.75,
            ("matcha_latte","melon_bread"): 0.80, ("matcha_latte","chiffon"): 0.75,
            ("matcha_latte","macaron"): 0.75,
            ("earl_grey","croissant"): 0.85, ("earl_grey","sourdough"): 0.80,
            ("earl_grey","baguette"): 0.75, ("earl_grey","chiffon"): 0.80,
            ("english_breakfast","croissant"): 0.80, ("english_breakfast","baguette"): 0.75,
            ("english_breakfast","sourdough"): 0.75, ("english_breakfast","pandesal"): 0.70,
            ("milk_tea","brioche"): 0.75, ("milk_tea","pandesal"): 0.75,
            ("milk_tea","eggtart"): 0.80,
        }
        if (bread, coffee) in overrides:
            s = overrides[(bread, coffee)]
        return round(s, 2)
    
    matrix = {}
    for b in _get_bakery():
        bk = b["key"]
        matrix[bk] = {}
        for c in _get_coffee():
            matrix[bk][c["key"]] = _pair_score(bk, c["key"])
    return matrix
def generate_pairing_matrix() -> dict:
    """Call DeepSeek to score every bread x coffee pair (0.0-1.0)."""
    try:
        prompt = _build_pairing_prompt()
        system = (
            "You are a professional bakery and coffee pairing expert. "
            "Score each bread-coffee pair from 0.0 (terrible match) to 1.0 (perfect match). "
            "Consider flavor complementarity, texture contrast, and traditional pairing wisdom. "
            "Return ONLY valid JSON with no commentary."
        )
        response = _call_deepseek(prompt, system, max_tokens=2000)
        matrix = json.loads(response)
        # Validate structure
        for bread in _get_bakery():
            bk = bread["key"]
            if bk not in matrix:
                raise ValueError(f"Missing bread: {bk}")
            for coffee in _get_coffee():
                ck = coffee["key"]
                if ck not in matrix[bk]:
                    matrix[bk][ck] = 0.3
                matrix[bk][ck] = max(0.0, min(1.0, float(matrix[bk][ck])))
        logger.info("Pairing matrix generated by DeepSeek")
        return matrix
    except Exception as e:
        logger.warning("DeepSeek pairing matrix unavailable (%s), using fallback", e)
        return _build_fallback_matrix()

def get_pairing_matrix(force_refresh: bool = False) -> dict:
    """Get the pairing matrix (cached after first generation)."""
    global _pairing_cache
    if force_refresh or _pairing_cache is None:
        _pairing_cache = generate_pairing_matrix()
    return _pairing_cache


def _build_pairing_prompt() -> str:
    lines = ["Score every bread-coffee pair (0.0-1.0) based on flavor compatibility.\n"]
    lines.append("Breads:")
    for b in _get_bakery():
        lines.append(f"  - {b['key']}: {b['desc']}")
    lines.append("\nCoffees:")
    for c in _get_coffee():
        lines.append(f"  - {c['key']}: {c['desc']}")
    lines.append("\nReturn JSON like:")
    example = {}
    for b in _get_bakery()[:2]:
        example[b["key"]] = {c["key"]: 0.5 for c in COFFEE[:2]}
    lines.append(json.dumps(example, indent=2))
    lines.append("\nInclude ALL breads and ALL coffees listed above. Scores must be 0.0-1.0.")
    return "\n".join(lines)
