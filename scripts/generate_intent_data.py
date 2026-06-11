"""
Generate synthetic intent training data by combining templates with product names.
Produces ~500+ samples across 7 classes.
"""
import json, os, random

PRODUCTS = ["croissant", "donut", "chiffon", "bread roll", "bread coconut",
            "croissant chocolate", "pastry", "cake", "bread"]
PRODUCTS_BM = ["croissant", "donut", "kek chiffon", "roti roll", "roti kelapa",
               "croissant coklat", "pastri", "kek", "roti"]

TEMPLATES = {
    "stock_query": [
        "How many {p} should I bake tomorrow?",
        "How many {p} for {day}?",
        "Check stock for {p}",
        "What is the inventory for {p}?",
        "Do we need to restock {p}?",
        "Prepare how many {p}?",
        "Stock level for {p} please",
        "Should I bake more {p}?",
        "How many units of {p} left?",
        "Check replenishment for {p}",
        "What to prepare for {day}?",
        "Inventory check for {p}",
        "How many {p} do I need?",
        "Restock {p} for today",
        "Do I have enough {p} for {day}?",
        "Tell me stock status for {p}",
        "Bake quantity for {p} {day}",
        "Should I prepare more {p}?",
        "Check {p} availability",
        "What is the restock recommendation for {p}?",
        "How many {p} needed for {day}?",
        "Stock count for {p}",
        "How much {p} stock is remaining?",
        "Is {p} running low?",
        "Forecast demand for {p} {day}",
        "Berapa banyak {p} untuk {day}?",
        "Stok {p} ada berapa?",
        "Bakar berapa {p} untuk {day}?",
        "Nak sediakan berapa {p}?",
        "Ada stok cukup {p} ke?",
        "Tolong check inventory {p}",
        "Berapa {p} perlu dibakar esok?",
        "Cukup ke stok {p} untuk hujung minggu?",
        "Prediksi jualan {p} {day}",
        "Semak baki stok {p}",
    ],
    "waste_analysis": [
        "Why is there so much waste this week?",
        "Analyse the spoilage rate for {p}",
        "What caused the high loss yesterday?",
        "Show me expired items report",
        "Why are we throwing away so many {p}?",
        "Waste analysis for {p} please",
        "Check why so many {p} are rosak",
        "Loss report for this month",
        "Which product has the highest spoilage?",
        "Why is food waste increasing?",
        "Show waste trends for {p}",
        "Analyse overproduction losses for {p}",
        "What is causing high wastage of {p}?",
        "Report on expired {p}",
        "How to reduce spoilage of {p}?",
        "Weekly waste summary for {p}",
        "Kenapa banyak {p} dibuang?",
        "Check throwing away rate for {p}",
        "Loss analysis for {p}",
        "Why are we losing money on waste?",
        "Food waste breakdown for {p}",
        "Overproduction impact on {p} waste",
        "Spoilage check for {p}",
        "Audit waste from {p} batch",
        "Throw away cost for {p} this month",
        "Rosak report for {p}",
        "How much did we waste on {p}?",
        "Mengapa banyak sangat pembaziran {p}?",
        "Buang berapa banyak {p} minggu lepas?",
        "Bazir banyak ke {p} semalam?",
    ],
    "promo_eval": [
        "Was the {p} combo deal effective?",
        "How did the discount on {p} perform?",
        "Evaluate the {p} promotion ROI",
        "Did the 20% off {p} bring more customers?",
        "Promo effectiveness for {p}",
        "Was the last {p} promotion worth it?",
        "Check {p} combo deal sales",
        "ROI of {p} promotion",
        "Should we continue the {p} discount?",
        "How effective was the {p} bundle offer?",
        "Promotion analysis for {p}",
        "Compare {p} sales during promo vs normal",
        "Did the {p} campaign increase revenue?",
        "Evaluate discount impact on {p}",
        "Was the {p} BOGO effective?",
        "Promo ROI for {p}",
        "Check if {p} combo boosted sales",
        "Nilai keberkesanan promosi {p}",
        "Campaign results for {p}",
        "Did the {p} weekend promo drive traffic?",
        "Bundle deal performance for {p}",
        "How much extra revenue from {p} promo?",
        "Promo lift analysis for {p}",
        "Tawaran {p} berkesan ke?",
        "Sales impact of {p} discount campaign",
        "Evaluate {p} promotions",
        "Compare promo period vs baseline for {p}",
        "Adakah jualan {p} meningkat masa promo?",
        "Effectiveness of {p} markdown",
        "Was the {p} flash sale successful?",
    ],
    "schedule_audit": [
        "Who is working {day}?",
        "Show me the staff schedule for {day}",
        "Is there enough staff {day}?",
        "Check if we are understaffed {day}",
        "How many bakers on duty {day}?",
        "Are the baristas scheduled properly {day}?",
        "Staff roster for this week",
        "Any staffing anomalies {day}?",
        "Siapa kerja {day}?",
        "Jadual pekerja {day}",
        "Ada cukup pekerja ke {day}?",
        "Check schedule for cashier {day}",
        "Who is on the morning shift {day}?",
        "Staffing level for {day}",
        "Show me the syif for {day}",
        "Any gaps in the schedule {day}?",
        "Baker coverage for {day}",
        "Who is on duty {day}?",
        "Staff attendance for {day}",
        "Check barista schedule for {day}",
        "Are all roles covered {day}?",
        "Pekerja cukup untuk {day}?",
        "Show cleaning staff roster {day}",
        "Who replaces staff on {day}?",
        "Morning shift staffing {day}",
        "Evening crew schedule {day}",
        "Staff allocation for {day}",
        "Any overtime issues {day}?",
        "Siapa yang kerja syif {day}?",
        "Coverage check for {day} shift",
    ],
    "cross_source_audit": [
        "Run a full store health check",
        "Audit everything please",
        "Any problems with the store today?",
        "Cross check all systems",
        "Full diagnostics report",
        "Store overview and issues",
        "Any alerts or risks today?",
        "Check all modules for problems",
        "Dashboard summary please",
        "Run a sweep of the whole system",
        "KPI check for the store",
        "Operations report for today",
        "Any issues I should know about?",
        "Integrity check all data sources",
        "Compliance check for the bakery",
        "Run diagnostics on all modules",
        "Quick health scan of the store",
        "System-wide audit report",
        "Check for any anomalies across modules",
        "Periksa semua sistem kedai",
        "Status check for all departments",
        "Any warnings I need to address?",
        "Full operational review today",
        "Scan for operational risks",
        "Dashboard health indicators check",
        "Run consistency check on data",
        "Show me all active alerts",
        "Integrated health report please",
        "Laporan penuh operasi kedai",
        "Check if everything is running smoothly",
        "Is the store operating normally?",
        "Daily operations status check",
        "Run a complete system diagnostic",
        "Generate store performance report",
        "Quick operational audit",
    ],
    "profit_analysis": [
        "How much profit did we make {period}?",
        "Show me the revenue breakdown for {period}",
        "What are our earnings {period}?",
        "Profit margin for {p}",
        "Income report for {period}",
        "How much money did we earn {period}?",
        "Gross profit {period}",
        "Net profit analysis for {period}",
        "Berapa untung {period}?",
        "Pendapatan {period}",
        "Sales revenue for {p}",
        "Cost analysis for the bakery",
        "Profit breakdown for {p}",
        "Which product has the best margin?",
        "Keuntungan kedai {period}",
        "How much revenue from {p}?",
        "Show profit vs cost for {p}",
        "Margin analysis for {p}",
        "Income statement for {period}",
        "Revenue report for {p}",
        "Berapa jualan kasar {period}?",
        "Calculate net earnings for {period}",
        "Profit and loss summary for {period}",
        "Which items are most profitable?",
        "Total gross margin {period}",
        "Sales income breakdown for {period}",
        "How are our profit margins trending?",
        "Cost vs revenue analysis for menu",
        "Laporan keuntungan {p}",
        "Compare profit across all products",
    ],
    "out_of_scope": [
        "What is the meaning of life?",
        "How to cook nasi lemak?",
        "Tell me a joke",
        "What is the weather like?",
        "Who won the football match?",
        "What time is it?",
        "Can you write a poem?",
        "How to apply for a loan?",
        "Where is the nearest petrol station?",
        "What is Bitcoin price?",
        "Cerita pasal diri awak",
        "Apa khabar?",
        "Siapa perdana menteri?",
        "How to fix my car?",
        "Play some music",
        "What is the capital of France?",
        "How to bake a cake at home?",
        "Recommend a good restaurant nearby",
        "What movies are playing this weekend?",
        "How do I lose weight fast?",
        "Berapa harga minyak sekarang?",
        "Cerita lawak sikit",
        "How tall is Mount Kinabalu?",
        "Can you translate this to French?",
        "What is the population of Malaysia?",
        "How to start an online business?",
        "Siapa penulis buku tu?",
        "Explain quantum physics simply",
        "What is the best phone to buy?",
        "How do I get to KLCC from here?",
        "What is the best laptop for programming?",
        "How to cook spaghetti?",
        "What are the best travel destinations?",
        "How to learn Python?",
        "Berapa lama nak sampai airport?",
        "Is it going to rain later?",
        "What is the latest iPhone price?",
        "Can you recommend a movie?",
        "How to invest in stocks?",
        "Tell me about Malaysian history",
    ],
}

DAYS = ["today", "tomorrow", "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday", "this weekend", "next week",
        "hari ini", "esok", "minggu depan", "hujung minggu"]

PERIODS = ["today", "yesterday", "this week", "last week", "this month",
           "last month", "June", "hari ini", "semalam", "minggu ini",
           "minggu lepas", "bulan ini", "bulan lepas"]

random.seed(42)
all_data = []

for intent, templates in TEMPLATES.items():
    for tmpl in templates:
        if "{p}" in tmpl and intent in ("stock_query", "waste_analysis", "promo_eval", "profit_analysis"):
            for prod in (PRODUCTS if random.random() > 0.3 else PRODUCTS_BM):
                text = tmpl.replace("{p}", prod)
                if "{day}" in text:
                    text = text.replace("{day}", random.choice(DAYS))
                if "{period}" in text:
                    text = text.replace("{period}", random.choice(PERIODS))
                all_data.append((text, intent))
        elif "{day}" in tmpl:
            for day in random.sample(DAYS, min(5, len(DAYS))):
                text = tmpl.replace("{day}", day)
                all_data.append((text, intent))
        elif "{period}" in tmpl:
            for period in random.sample(PERIODS, min(3, len(PERIODS))):
                text = tmpl.replace("{period}", period)
                all_data.append((text, intent))
        else:
            all_data.append((tmpl, intent))

# Deduplicate
seen = set()
unique = []
for text, intent in all_data:
    if text not in seen:
        seen.add(text)
        unique.append((text, intent))

# Count per class
from collections import Counter
counts = Counter(i for _, i in unique)
for intent, count in sorted(counts.items()):
    print(f"  {intent}: {count} samples")

output_path = os.path.join(os.path.dirname(__file__), "..", "models", "distilbert", "training_data.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(unique, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(unique)} samples to {output_path}")
