"""Embedding provider settings (vectorization of chunks/queries)."""

from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings


class EmbedConfig(BaseSettings):
    """Embedding provider configuration. OPENAI_*/OLLAMA_* fields are shared with
    LLMConfig and collapse via AppConfig's MRO. Embedding requests reuse
    LLM_TIMEOUT (from LLMConfig) — there is no separate embed timeout."""

    EMBED_PROVIDER: str = Field(default="ollama", description="Embedding provider: 'ollama' or 'openai'.")
    OLLAMA_BASE_URL: str = Field(default="", description="Ollama host base URL (shared with LLMConfig).")
    OLLAMA_EMBED_MODEL: str = Field(default="qwen3-embedding:4b", description="Embedding model id for the ollama provider.")
    OPENAI_API_KEY: str = Field(default="", description="API key for the openai provider (shared with LLMConfig).")
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1", description="Base URL for the openai provider (shared).")
    OPENAI_EMBED_MODEL: str = Field(default="text-embedding-3-small", description="Embedding model id for the openai provider.")
    EMBED_BATCH_SIZE: PositiveInt = Field(default=32, description="Texts per embedding request.")
    EMBED_MAX_CHARS: PositiveInt = Field(default=2000, description="Max chars per text sent to the embedder.")
