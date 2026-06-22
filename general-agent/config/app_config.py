"""Unified application config — the single source of truth.

``AppConfig`` multi-inherits every config group; duplicate fields collapse via
MRO (OPENAI_API_KEY/OPENAI_BASE_URL/OLLAMA_BASE_URL across LLM↔Embed;
LOG_LEVEL across Api↔Retrieval). Only this class is instantiated, so
it carries the single ``model_config``.
"""
from pydantic_settings import SettingsConfigDict

from config.api_config import ApiConfig
from config.connector_config import ConnectorConfig
from config.embed_config import EmbedConfig
from config.gdrive_config import GDriveConfig
from config.ingestion_config import IngestionConfig
from config.llm_config import LLMConfig
from config.mongo_config import MongoConfig
from config.retrieval_config import RetrievalConfig
from config.storage_config import StorageConfig
from config.weaviate_config import WeaviateConfig


class AppConfig(
    LLMConfig,
    EmbedConfig,
    WeaviateConfig,
    ApiConfig,
    IngestionConfig,
    RetrievalConfig,
    MongoConfig,
    ConnectorConfig,
    GDriveConfig,
    StorageConfig,
):
    """Every setting in one object; read via the ``app_config`` singleton."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
