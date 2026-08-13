"""Configuration for the optional OpenAI-compatible LLM provider."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class LLMProviderConfig:
    """Resolved credentials and endpoint details for an optional LLM provider."""

    api_key: str
    base_url: str
    model: str

    @property
    def configured(self) -> bool:
        """Return whether all values required for an API request are present."""
        return bool(self.api_key and self.base_url and self.model)


def get_llm_provider_config() -> LLMProviderConfig:
    """Resolve generic variables first, then legacy DeepSeek variables."""
    return LLMProviderConfig(
        api_key=os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=(
            os.getenv("LLM_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com/v1"
        ).rstrip("/"),
        model=(
            os.getenv("LLM_MODEL")
            or os.getenv("DEEPSEEK_MODEL")
            or "deepseek-chat"
        ),
    )
