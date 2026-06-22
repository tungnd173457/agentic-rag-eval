"""Consolidated application configuration package.

One singleton — ``app_config`` — holds every setting (LLM, embedding, Weaviate,
API, ingestion, retrieval), read once at import (env > .env > defaults). Access
fields directly, e.g. ``app_config.WEAVIATE_URL``. There is no ``get_config()``.
"""

from config.app_config import AppConfig

app_config = AppConfig()

__all__ = ["AppConfig", "app_config"]
