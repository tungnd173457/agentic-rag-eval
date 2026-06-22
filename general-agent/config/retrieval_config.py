"""Retrieval-side tunables (agent loop, kb_search, get_document, reranker).

Folded into ``AppConfig`` — no standalone singleton. Read via ``app_config``.
"""
from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings


class RetrievalConfig(BaseSettings):
    # ---- Agent loop ----
    MAX_TOOL_ROUNDS: PositiveInt = Field(default=3, description="Max tool-calling rounds before a forced answer.")
    # ---- kb_search ----
    KB_SEARCH_TOP_K: PositiveInt = Field(default=8, description="Parents returned post-rerank.")
    KB_SEARCH_WIDE_K: PositiveInt = Field(default=50, description="Children fetched pre-rerank.")
    KB_SEARCH_HYBRID_ALPHA: float = Field(default=0.5, ge=0.0, le=1.0, description="0=BM25 only, 1=vector only.")
    # ---- kb_get_document ----
    KB_GET_DOC_MAX_TOKENS: PositiveInt = Field(default=4000, description="get_document token budget.")
    KB_GET_DOC_MAX_POSITIONS: PositiveInt = Field(default=8, description="Max sections per get_document call.")
    # ---- Reranker ----
    COHERE_API_KEY: str | None = Field(default=None, description="Cohere rerank API key; None disables reranking.")
    USE_RERANKING: bool = Field(default=True, description="Enable Cohere reranking.")
    RERANK_PROVIDER: str = Field(default="cohere", description="Rerank backend: 'cohere' or 'openrouter'.")
    COHERE_RERANK_MODEL: str = Field(default="rerank-multilingual-v3.0", description="Rerank model id when RERANK_PROVIDER='cohere'.")
    OPENROUTER_RERANK_MODEL: str = Field(default="cohere/rerank-v3.5", description="Rerank model id when RERANK_PROVIDER='openrouter'.")
    # ---- Chat (multi-turn conversation + compaction) ----
    CHAT_MAX_CONTEXT_TOKENS: PositiveInt = Field(
        default=4000, description="Token ceiling that triggers LLM summarize compaction."
    )
    CHAT_KEEP_RECENT_TURNS: PositiveInt = Field(
        default=4, description="Recent turns always kept verbatim (never summarized)."
    )
    # ---- Runtime ----
    LOG_LEVEL: str = Field(default="INFO", description="Runtime log level (shared with ApiConfig).")
