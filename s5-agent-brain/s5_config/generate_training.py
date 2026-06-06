# -*- coding: utf-8 -*-
import json, os, random, sys, time, httpx, asyncio
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from config.settings import DEEPSEEK_API_KEY
except ImportError:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") + "/chat/completions"

PRODUCT_NAMES = ["croissant", "donut", "chiffon", "bread_roll", "bread_coconut", "croissant_chocolate"]
DISCOUNTS = ["10", "15", "20", "25", "30", "40"]
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data.json")

TEMPLATES = {
    "stock_query": [
        "How many {p} should I bake tomorrow?","How many {p} tomorrow?","What is the stock of {p}?",
        "How is {p} doing?","Should I bake more {p}?","Check {p} status","Is {p} selling well?",
        "Tell me about {p}","Show me {p} forecast","What about {p}?","{p} stock level","{p} demand",
        "Look at {p}","Report on {p}","Any issues with {p}?","Do I need to restock {p}?",
        "How many {p} should I prepare?","Bake {p} or not?","How many {p} do I need?",
        "Forecast for {p}","{p} prediction","What is the demand for {p}?","How much {p} to make?",
        "Production plan for {p}","Do we have enough {p}?","{p} inventory check",
        "Running low on {p}?","Should I order more {p}?","{p} sales forecast",
        "How is my inventory?","What is my stock?","Stock check","Compare {p1} and {p2}",
    ],
    "waste_analysis": [
        "Why is waste high this week?","Why is waste high?","What is causing waste?",
        "How much waste do I have?","Any products expiring?","Why am I throwing away {p}?",
        "Is {p} going to expire?","Waste analysis","Check for expiry risks",
        "Which products have high loss?","Spoilage report","Why are we losing money on waste?",
        "What is going bad?","Anything about to spoil?","Waste reduction ideas",
        "How to reduce waste?","Which items are expiring soon?","Loss analysis",
        "Food waste check","What products are stale?","Day-old stock report",
        "What should I throw out?","Shelf life check","Expiring inventory",
    ],
    "promo_eval": [
        "Run a {d}% promo on {p}","Run a promo on {p}","Should I discount {p}?",
        "What discount for {p}?","Apply discount to {p}","Mark down {p}",
        "Bundle deal for {p}","Promo for {p}","Discount {p}","Sale on {p}",
        "Should I run a sale?","What promo should I run?","Give me a discount for {p}",
        "Can I put {p} on sale?","Best promo for {p}","How much off for {p}?",
        "Special offer for {p}","Clearance on {p}","Price cut for {p}","Deal for {p}",
        "What is the best discount?","Promotional pricing for {p}","Flash sale {p}?",
        "Recommended discount","Suggest a promo",
    ],
    "schedule_audit": [
        "Check schedule for anomalies","Check schedule","Who is working tomorrow?",
        "Any staffing issues?","Schedule audit","Is there enough staff?",
        "Any sick leave today?","Who can swap shifts?","Staff schedule check",
        "Do I have enough bakers?","Schedule overview","Check staff roster",
        "Any shift gaps?","Are we understaffed?","Who is on shift?",
        "Staffing for today","Employee schedule","Shift plan","Coverage check",
        "Workforce check","Who is available?","Baker schedule","Cashier schedule","Team roster",
    ],
    "cross_source_audit": [
        "Run a full store health check","Store health check","Audit the store",
        "Full audit","Cross check all products","Store overview","How is the store doing?",
        "Complete store audit","Health check all products","Run full audit",
        "All products health check","Store status","Overall health","System check",
        "Everything ok?","Store diagnostic","Full report","Comprehensive check",
        "Store analysis","Operations audit",
    ],
    "profit_analysis": [
        "What is my profit margin?","What is my profit margin on {p}?","Profit analysis",
        "How much profit did I make?","What is my revenue?","What are my costs?",
        "Margin check","Financial overview","How much did I earn?","Cost breakdown",
        "Income report","Earnings summary","Bottom line check","How profitable am I?",
        "Revenue report","Cost analysis","Financial health","Money check","Business performance",
    ],
}

OUT_OF_SCOPE = [
    "Tell me a joke","What is the weather?","Who are you?","How old are you?",
    "What is the meaning of life?","Play some music","What time is it?",
    "Can you help me with math?","What is your name?","How to bake a cake?",
    "What is AI?","Tell me a story","Where is the nearest bank?",
    "What is the capital of France?","How do I get to the airport?",
    "Recommend a restaurant","What is the stock market doing?","Who won the game?",
    "Translate this to French","Write me a poem","What is 2+2?",
    "How to fix my car?","Good morning","Thank you","Hello","Bye",
    "What is your favorite color?","Can you sing?","Do you dream?","Are you human?",
]

async def llm_augment(intent, base_examples, n_new=30):
    if not DEEPSEEK_API_KEY:
        return []
    sample = random.sample(base_examples, min(5, len(base_examples)))
    refs = "\n".join("- " + s for s in sample)
    prompt = f"Generate {n_new} diverse bakery AI queries. Intent: {intent}. Products: croissant, donut, chiffon, bread_roll, bread_coconut, croissant_chocolate.\nRefs:\n{refs}\nRules: vary phrasing, include Malaysian English, casual shorthand, typos. One query per line. No numbers, no quotes."
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(LLM_URL,
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 2000, "temperature": 0.9})
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()
            return [q.strip("- ").strip() for q in raw.split("\n") if q.strip() and len(q.strip()) > 3][:n_new]
    except Exception as e:
        print(f"  LLM augment failed for {intent}: {e}")
        return []

async def main():
    all_data = []
    seen = set()

    print("Step 1: Template expansion...")
    for intent, queries in TEMPLATES.items():
        for q in queries:
            if "{p1}" in q and "{p2}" in q:
                for i, p1 in enumerate(PRODUCT_NAMES):
                    for p2 in PRODUCT_NAMES[i+1:]:
                        text = q.replace("{p1}", p1).replace("{p2}", p2)
                        if text not in seen:
                            seen.add(text)
                            all_data.append({"text": text, "intent": intent})
            elif "{p}" in q:
                for p in PRODUCT_NAMES:
                    text = q.replace("{p}", p)
                    if text not in seen:
                        seen.add(text)
                        all_data.append({"text": text, "intent": intent})
            elif "{d}" in q:
                for d in DISCOUNTS:
                    for p in PRODUCT_NAMES:
                        text = q.replace("{d}", d).replace("{p}", p)
                        if text not in seen:
                            seen.add(text)
                            all_data.append({"text": text, "intent": intent})
            else:
                if q not in seen:
                    seen.add(q)
                    all_data.append({"text": q, "intent": intent})

    for q in OUT_OF_SCOPE:
        if q not in seen:
            seen.add(q)
            all_data.append({"text": q, "intent": "out_of_scope"})

    print(f"  Templates: {len(all_data)} examples")

    print("Step 2: LLM augmentation...")
    if DEEPSEEK_API_KEY:
        for intent in TEMPLATES:
            base = [d["text"] for d in all_data if d["intent"] == intent]
            target = 2000 if intent == "stock_query" else 1000
            needed = max(0, target - len(base))
            batches = (needed + 29) // 30
            for b in range(batches):
                n = min(30, needed - b * 30)
                new = await llm_augment(intent, base, n)
                for q in new:
                    if q not in seen:
                        seen.add(q)
                        all_data.append({"text": q, "intent": intent})
                cnt = sum(1 for d in all_data if d["intent"] == intent)
                print(f"  {intent}: {cnt}")
                time.sleep(0.3)

        base_oos = [d["text"] for d in all_data if d["intent"] == "out_of_scope"]
        needed = max(0, 500 - len(base_oos))
        for b in range((needed + 29) // 30):
            n = min(30, needed - b * 30)
            new = await llm_augment("out_of_scope", base_oos, n)
            for q in new:
                if q not in seen:
                    seen.add(q)
                    all_data.append({"text": q, "intent": "out_of_scope"})
            cnt = sum(1 for d in all_data if d["intent"] == "out_of_scope")
            print(f"  out_of_scope: {cnt}")
            time.sleep(0.3)
    else:
        print("  No API key, skipping LLM augmentation")
        print("  Total with templates only:", len(all_data))

    random.shuffle(all_data)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    counts = Counter(d["intent"] for d in all_data)
    print(f"\nFinal total: {len(all_data)} examples")
    for intent, count in counts.most_common():
        print(f"  {intent}: {count}")

if __name__ == "__main__":
    asyncio.run(main())
