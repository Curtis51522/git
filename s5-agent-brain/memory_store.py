# Session Memory Store - JSON-based + MySQL dual-write
import json, os, logging, time, sys
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger('s5.memory')

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'memory')
MAX_TURNS = 20

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

os.makedirs(MEMORY_DIR, exist_ok=True)

def _session_path(session_id):
    safe = ''.join(c for c in session_id if c.isalnum() or c in '_-')
    return os.path.join(MEMORY_DIR, f'{safe}.json')

def _mysql_save(session_id, turn):
    try:
        from db.mysql_client import get_db
        db = get_db()
        cur = db.cursor()
        cur.execute(
            'INSERT INTO s5_memory_episodic (session_id, query, intent, product, target_date, response, data_snapshot, importance) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)',
            (session_id,
             turn.get('query','')[:500],
             turn.get('intent','general')[:32],
             turn.get('product','')[:64],
             turn.get('target_date','')[:16],
             turn.get('decision', turn.get('response',''))[:2000],
             json.dumps(turn.get('key_data',{}), default=str),
             float(turn.get('importance',0.5))))
        db.commit()
        cur.close()
    except Exception:
        pass

def load_session(session_id):
    path = _session_path(session_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def save_turn(session_id, turn):
    history = load_session(session_id)
    turn['timestamp'] = datetime.now().isoformat()
    history.append(turn)
    if len(history) > MAX_TURNS:
        history = history[-MAX_TURNS:]
    path = _session_path(session_id)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, default=str)
    _mysql_save(session_id, turn)

def get_recent_context(session_id, n=5):
    history = load_session(session_id)
    if not history:
        return 'No prior conversation.'
    recent = history[-n:]
    lines = []
    for i, turn in enumerate(recent, 1):
        dec = turn.get('decision', turn.get('response', '?'))
        lines.append(
            '[Turn -' + str(len(recent)-i) + '] '
            'Q: ' + turn.get('query','?')[:80] + ' | '
            'Intent: ' + turn.get('intent','?') + ' | '
            'Decision: ' + dec[:100]
        )
    return '\n'.join(lines)

def get_key_metrics(session_id, n=3):
    history = load_session(session_id)
    if not history:
        return {}
    recent = history[-n:]
    metrics = {'forecast_history': [], 'inventory_history': [], 'decisions': [], 'product_scopes': []}
    for turn in recent:
        data = turn.get('key_data', {})
        if 'forecast' in data:
            metrics['forecast_history'].append(data['forecast'])
        if 'inventory' in data:
            metrics['inventory_history'].append(data['inventory'])
        metrics['decisions'].append(turn.get('decision', turn.get('response',''))[:80])
        scope = data.get('product_scope', turn.get('product', ''))
        metrics['product_scopes'].append(scope)
    return metrics

def clear_session(session_id):
    path = _session_path(session_id)
    if os.path.exists(path):
        os.remove(path)
