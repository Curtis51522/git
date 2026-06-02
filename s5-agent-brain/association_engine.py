import json, os, sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)

def get_associations(product_name=None, top_n=5):
    try:
        from db.mysql_client import get_db
        db = get_db()
        cur = db.cursor()
        
        # Get all receipts with items JSON
        cur.execute('SELECT items FROM receipts ORDER BY created_at DESC LIMIT 500')
        rows = cur.fetchall()
        cur.close()
        
        # Parse items and count co-occurrences
        from collections import defaultdict, Counter
        product_count = Counter()
        pair_count = Counter()
        
        for row in rows:
            try:
                items = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(items, list):
                continue
            
            # Extract product names
            names = set()
            for item in items:
                name = item.get('product_name', '') if isinstance(item, dict) else str(item)
                if name:
                    names.add(name)
                    product_count[name] += 1
            
            # Count pairs
            names_list = list(names)
            for i in range(len(names_list)):
                for j in range(i+1, len(names_list)):
                    a, b = sorted([names_list[i], names_list[j]])
                    pair_count[(a, b)] += 1
        
        # Filter for specific product(s) if requested
        if product_name and product_name not in ('all', '-', ''):
            targets = set(p.strip() for p in product_name.split(','))
            results = []
            for (a, b), cnt in pair_count.items():
                for t in targets:
                    if a == t and product_count[a] > 0:
                        confidence = round(cnt / product_count[a] * 100, 1)
                        results.append({'pair': f'{a} + {b}', 'confidence': confidence, 'co_count': cnt, 'query_product': t})
                    elif b == t and product_count[b] > 0:
                        confidence = round(cnt / product_count[b] * 100, 1)
                        results.append({'pair': f'{b} + {a}', 'confidence': confidence, 'co_count': cnt, 'query_product': t})
            results.sort(key=lambda x: -x['confidence'])
            # Deduplicate by pair name
            seen = set()
            deduped = []
            for r in results:
                pair = tuple(sorted(r['pair'].split(' + ')))
                if pair not in seen:
                    seen.add(pair)
                    deduped.append(r)
            return deduped[:top_n]
        else:
            # Return top pairs overall
            results = []
            for (a, b), cnt in pair_count.most_common(top_n * 3):
                base = max(product_count[a], product_count[b])
                if base > 0:
                    confidence = round(cnt / base * 100, 1)
                    results.append({'pair': f'{a} + {b}', 'confidence': confidence, 'co_count': cnt})
            results.sort(key=lambda x: -x['confidence'])
            return results[:top_n]
    except Exception as e:
        return []

def get_period_comparison(days_back=7):
    try:
        from db.mysql_client import get_db
        db = get_db()
        cur = db.cursor()
        from datetime import datetime, timedelta
        target = datetime.now() - timedelta(days=days_back)
        target_str = target.strftime('%Y-%m-%d')
        
        cur.execute(
            'SELECT query, intent, product, data_snapshot, created_at FROM s5_memory_episodic WHERE created_at >= %s ORDER BY created_at DESC LIMIT 10',
            (target_str + ' 00:00:00',))
        rows = cur.fetchall()
        cur.close()
        
        if not rows:
            return None
        
        # Extract the most relevant past snapshot
        snapshots = []
        for row in rows:
            try:
                snap = json.loads(row[3]) if isinstance(row[3], str) else row[3]
            except (json.JSONDecodeError, TypeError):
                snap = {}
            if snap:
                snapshots.append({'date': str(row[4]), 'data': snap, 'intent': row[1]})
        
        return snapshots[:3] if snapshots else None
    except Exception:
        return None
