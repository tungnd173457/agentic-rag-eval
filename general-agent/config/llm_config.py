"""LLM / vision provider settings (text + multimodal generation)."""

from pydantic import Field, PositiveFloat
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """Text + vision LLM provider configuration.

    Fields shared with EmbedConfig (OPENAI_API_KEY/OPENAI_BASE_URL/
    OLLAMA_BASE_URL) collapse to one field under AppConfig's MRO.
    Only ``AppConfig`` is instantiated, so this carries no ``model_config``.
    ``LLM_TIMEOUT`` covers chat, vision and embedding requests alike.
    """

    LLM_PROVIDER: str = Field(default="ollama", description="Chat provider: 'ollama', 'openai' or 'openrouter'.")
    OLLAMA_BASE_URL: str = Field(default="", description="Ollama host base URL (OpenAI-compatible /v1 is appended).")
    OLLAMA_LLM_MODEL: str = Field(default="qwen2.5vl:7b", description="Chat model id for the ollama provider.")
    OPENAI_API_KEY: str = Field(default="", description="API key for the openai provider.")
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1", description="Base URL for the openai provider (ends in /v1).")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", description="Chat model id for the openai provider.")
    OPENAI_VISION_MODEL: str = Field(default="gpt-4o-mini", description="Vision model id for the openai provider.")
    OLLAMA_VISION_MODEL: str = Field(default="qwen2.5vl:7b", description="Vision model id for the ollama provider.")
    LLM_TIMEOUT: PositiveFloat = Field(default=120.0, description="Request timeout (seconds) for chat, vision and embedding.")
    OPENROUTER_API_KEY: str = Field(default="", description="API key for the openrouter provider (shared with rerank).")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1", description="Base URL for the openrouter provider (ends in /v1).")
    OPENROUTER_MODEL: str = Field(default="openai/gpt-4o-mini", description="Chat model id for the openrouter provider.")
    OPENROUTER_VISION_MODEL: str = Field(default="openai/gpt-4o-mini", description="Vision model id for the openrouter provider.")
