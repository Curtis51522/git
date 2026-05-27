from supabase import create_client, Client
from config.settings import SUPABASE_URL, SUPABASE_ANON_KEY

_client = None

def get_supabase():
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _client
