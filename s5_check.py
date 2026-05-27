"""S5 Reproducibility Quick-Check Script"""
import sys, os
PASS = FAIL = WARN = 0
def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")

print("=" * 60)
print("S5 BAKERY AI SYSTEM -- REPRODUCIBILITY CHECK")
print("=" * 60)

print()
print("1. PYTHON ENVIRONMENT")
v = sys.version_info
check(f"Python {v.major}.{v.minor}.{v.micro}", v >= (3, 10), "Need Python 3.10+")

print()
print("2. DEPENDENCIES")
for mod in ["fastapi","transformers","torch","xgboost","shap","sklearn","pandas","numpy","httpx","openai","datasets"]:
    try:
        __import__(mod)
        print(f"  [ OK ] {mod}")
    except ImportError:
        FAIL += 1
        print(f"  [FAIL] {mod}")

print()
print("3. TRAINED MODELS")
for path, name in [("models/distilbert/config.json","DistilBERT"),("models/xgboost/croissant_model.json","XGBoost:croissant"),("models/xgboost/donut_model.json","XGBoost:donut"),("models/xgboost/chiffon_model.json","XGBoost:chiffon"),("models/xgboost/bread_roll_model.json","XGBoost:bread_roll"),("models/xgboost/bread_coconut_model.json","XGBoost:bread_coconut"),("models/xgboost/croissant_chocolate_model.json","XGBoost:croissant_choc"),("training/intent_data.json","Intent data")]:
    check(name, os.path.exists(path), f"Missing: {path}")

print()
print("4. ENV VARS")
for var, name in [("DEEPSEEK_API_KEY","DeepSeek"),("WEATHER_API_KEY","Weather")]:
    val = os.getenv(var, "")
    if val:
        print(f"  [ OK ] {name}")
    else:
        WARN += 1
        print(f"  [WARN] {name} -- set in .env")

print()
print("5. MYSQL")
try:
    from db.mysql_client import get_db
    db = get_db()
    c = db.cursor()
    c.execute("SELECT 1")
    c.fetchall()  # consume result
    print(f"  [ OK ] MySQL connected (bakery_ai)")
    c.execute("SHOW TABLES")
    print(f"  [ OK ] Tables: {len(c.fetchall())}")
    c.close()
except Exception as e:
    FAIL += 1
    print(f"  [FAIL] MySQL: {e}")

print()
print("6. INTENT CLASSIFIER")
try:
    from api.module5_agent.intent import get_classifier
    clf = get_classifier()
    print(f"  [ OK ] Model loaded: {clf._model_loaded}")
    for q, exp in [("How many croissants tomorrow?","stock_query"),("Esok nak bake berapa croissant?","stock_query"),("Cuaca hari ini macam mana?","out_of_scope")]:
        i, c = clf.classify(q)
        ok = "PASS" if i == exp else "FAIL"
        print(f"  [{ok}] {q[:50]} -> {i} ({c:.0%})")
except Exception as e:
    FAIL += 1
    print(f"  [FAIL] {e}")

print()
print("7. SHAP CAUSAL REASONING")
try:
    from api.module5_agent.causal_reasoning import compute_shap_attribution
    from datetime import datetime, timedelta
    t = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    attr = compute_shap_attribution("croissant", t)
    if attr.get("error"):
        FAIL += 1
        print(f"  [FAIL] {attr["error"]}")
    else:
        print(f"  [ OK ] Croissant {t}: {attr["predicted_demand"]}u, top driver: {attr["top_driver"]}")
        for c in attr["shap_contributions"][:3]:
            print(f"         {c["effective_sign"]}{c["label"]} ({c["abs_impact"]:.1f})")
except Exception as e:
    FAIL += 1
    print(f"  [FAIL] {e}")

print()
print("=" * 60)
print(f"RESULTS: {PASS} passed, {WARN} warnings, {FAIL} failed")
print("SYSTEM READY. Run: python main.py" if FAIL == 0 else "FIX FAILED CHECKS BEFORE RUNNING")
print("=" * 60)