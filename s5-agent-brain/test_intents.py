import sys; sys.path.insert(0, '.')
from server import parse_query, AGENTS

distilbert = ['stock_query','waste_analysis','promo_eval','schedule_audit','cross_source_audit','profit_analysis','out_of_scope']
tests = {
    'How many croissants tomorrow?': 'stock_query',
    'Why is waste high?': 'waste_analysis',
    'Run a 20% promo on donuts': 'promo_eval',
    'Check schedule for anomalies': 'schedule_audit',
    'Run a full store health check': 'cross_source_audit',
    'What is my profit margin?': 'profit_analysis',
    'Tell me a joke': 'out_of_scope',
}

print('=== Intent Routing Validation ===')
phase1_intents = set()
for q, expected in tests.items():
    p = parse_query(q)
    got = p['intent']
    prod = p['product']
    phase1_intents.add(got)
    match = 'OK' if got == expected else 'FAIL'
    print(f'  {match}: "{q[:40]}" -> {got} (product={prod})')

print()
print(f'Phase 1 intents: {sorted(phase1_intents)}')
print(f'DistilBERT 7-class: {distilbert}')
print(f'Exact match: {sorted(phase1_intents) == sorted(distilbert)}')
print(f'Total agents: {len(AGENTS)} -> {[a.name for a in AGENTS]}')
