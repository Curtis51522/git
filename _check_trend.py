import urllib.request, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

r = urllib.request.urlopen("http://127.0.0.1:8002/s4/revenue/daily?date=2026-06-30", timeout=10)
d = json.loads(r.read())
data = d.get("data", {})
trend = data.get("trend", {})
print("Dates:", trend.get("dates", [])[-7:])
print("Bread:", trend.get("bread", [])[-7:])
print("Orders:", trend.get("orders", [])[-7:])
print("AVG:", trend.get("avg_order", [])[-7:])
print("Total trend entries:", len(trend.get("dates", [])))
