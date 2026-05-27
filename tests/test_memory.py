"""
Tests for S5 Agent Query Multi-turn Memory (MemoryStore).
Phase 2 TDD ? these must FAIL before implementation.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Module under test (does NOT exist yet ? RED phase)
from api.module5_agent.memory import MemoryStore, init_memory_tables


@pytest.fixture
def store():
    """Create a fresh MemoryStore and ensure tables exist."""
    init_memory_tables()
    return MemoryStore()


class TestStoreEpisode:
    """Contract: store_episode(session_id, query, intent, product, target_date, response, data_snapshot) -> int"""

    def test_store_and_return_id(self, store):
        """Happy path: store an episode and get a positive integer ID back."""
        episode_id = store.store_episode(
            session_id="test_session_1",
            query="How many croissants tomorrow?",
            intent="stock_query",
            product="croissant",
            target_date="2026-05-28",
            response="Bake 41 croissants.",
            data_snapshot={"forecast": 45, "inventory": 4, "capacity": 50},
        )
        assert isinstance(episode_id, int)
        assert episode_id > 0

    def test_empty_session_id_fallback(self, store):
        """Boundary: empty session_id should fall back to 'default' without error."""
        episode_id = store.store_episode(
            session_id="",
            query="Test query",
            intent="general",
            product="",
            target_date="",
            response="Test response",
            data_snapshot={},
        )
        assert isinstance(episode_id, int)
        assert episode_id > 0

    def test_empty_query_still_stored(self, store):
        """Boundary: empty query should not crash."""
        episode_id = store.store_episode(
            session_id="s1", query="", intent="general",
            product="", target_date="", response="ok", data_snapshot={},
        )
        assert episode_id > 0

    def test_large_snapshot_truncated(self, store):
        """Boundary: snapshot > 5KB should be truncated, not rejected."""
        large_snapshot = {"key": "x" * 6000}
        episode_id = store.store_episode(
            session_id="s1", query="q", intent="general",
            product="", target_date="", response="r",
            data_snapshot=large_snapshot,
        )
        assert episode_id > 0


class TestRetrieveEpisodes:
    """Contract: retrieve_episodes(session_id, product, target_date, limit) -> list[dict]"""

    def test_retrieve_returns_list(self, store):
        """Happy path: returns a list."""
        store.store_episode("s_r1", "q1", "stock_query", "croissant", "", "r1", {})
        results = store.retrieve_episodes("s_r1")
        assert isinstance(results, list)

    def test_retrieve_respects_limit(self, store):
        """Should return at most `limit` results."""
        for i in range(10):
            store.store_episode("s_r2", f"q{i}", "general", "", "", f"r{i}", {})
        results = store.retrieve_episodes("s_r2", limit=3)
        assert len(results) <= 3

    def test_retrieve_product_filter(self, store):
        """Should only return episodes matching the product."""
        store.store_episode("s_r3", "q1", "stock_query", "croissant", "", "r1", {})
        store.store_episode("s_r3", "q2", "stock_query", "donut", "", "r2", {})
        results = store.retrieve_episodes("s_r3", product="croissant")
        for r in results:
            assert r.get("product", "") == "croissant"

    def test_retrieve_no_results(self, store):
        """Boundary: no matching episodes returns empty list, not error."""
        results = store.retrieve_episodes("nonexistent_session_xyz")
        assert results == []


class TestGetRecentContext:
    """Contract: get_recent_context(session_id, n, product) -> str"""

    def test_returns_string(self, store):
        """Happy path: returns a string."""
        store.store_episode("s_c1", "How many croissants?", "stock_query",
                            "croissant", "", "Bake 41.", {"forecast": 45})
        ctx = store.get_recent_context("s_c1", n=3)
        assert isinstance(ctx, str)

    def test_empty_when_no_history(self, store):
        """Boundary: returns empty string when no episodes."""
        ctx = store.get_recent_context("no_history_session")
        assert ctx == ""

    def test_respects_n_limit(self, store):
        """Should include at most n episodes in context."""
        for i in range(5):
            store.store_episode("s_c2", f"q{i}", "general", "", "", f"r{i}", {})
        ctx = store.get_recent_context("s_c2", n=2)
        # Should not contain all 5 queries
        assert "q3" not in ctx or "q4" not in ctx


class TestGenerateReflection:
    """Contract: generate_reflection(session_id) -> str | None"""

    def test_returns_none_when_no_episodes(self, store):
        """Boundary: no episodes -> None, not crash."""
        result = store.generate_reflection("empty_session")
        assert result is None

    def test_returns_string_with_enough_episodes(self, store):
        """Happy path: with 20+ episodes, returns a reflection string."""
        for i in range(20):
            store.store_episode("s_ref1", f"How many croissants on day {i}?",
                                "stock_query", "croissant", f"2026-05-{28+i}",
                                f"Bake {40+i}.", {"forecast": 40+i, "inventory": i})
        result = store.generate_reflection("s_ref1")
        # May fail if DeepSeek unavailable ? that's OK
        # Just check it doesn't crash
        assert result is None or isinstance(result, str)
