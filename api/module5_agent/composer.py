"""
Composer Agent -- natural-language output synthesis for S5.

Two consumption paths:
1. Store-manager summary (POST /s5/query)  -- Fusion data + AuditWarnings -> readable text
2. Staff upselling script (POST /s5/script) -- combo scores -> suggestion text

LLM strategy:
- DeepSeek API (primary) -- OpenAI-compatible, ~$0.27/M tokens
- Mock fallback (secondary) -- when API key missing or call fails
"""

import logging
from typing import Optional

from api.mock_llm import mock_composer
from api.module5_agent.llm_client import compose_summary_real, compose_script_real
from api.module5_agent.sql_templates import get_template

logger = logging.getLogger("s5.composer")


class ComposerAgent:
    """Composer -- translates structured results into natural language."""

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compose_summary(self, result: dict, query: str = "", intent: str = "stock_query") -> str:
        """Generate a store-manager-facing decision summary.
        
        Tries DeepSeek API first; falls back to mock if unavailable.
        """
        if self.use_mock:
            return mock_composer(result, "summary")

        # Try real DeepSeek first
        summary = compose_summary_real(result, query, intent)
        if summary:
            return summary

        # Fallback to mock
        logger.info("Composer: DeepSeek unavailable, using mock fallback")
        return mock_composer(result, "summary")

    def compose_script(self, combo_data):
        """Generate a staff-facing upselling script.
        
        Tries DeepSeek API first; falls back to mock if unavailable.
        """
        if self.use_mock:
            return [{"products": "Combo", "script": mock_composer(combo_data, "script")}]

        # Try real DeepSeek first
        scripts = compose_script_real(combo_data)
        if scripts:
            return scripts

        # Fallback to mock
        logger.info("Composer: DeepSeek unavailable, using mock fallback")
        return [{"products": "Combo", "script": mock_composer(combo_data, "script")}]

    # ------------------------------------------------------------------
    # SQL strategy
    # ------------------------------------------------------------------
    def build_sql(self, intent: str, params: dict) -> Optional[str]:
        """Return a parameterized SQL query for *standard* intents.

        Returns None if the intent has no template, signalling the caller
        to fall back to LLM-generated SQL.
        """
        template = get_template(intent)
        if template is None:
            return None

        return template.strip()

    @staticmethod
    def has_template(intent: str) -> bool:
        """Check whether a parameterized SQL template exists for an intent."""
        return get_template(intent) is not None
