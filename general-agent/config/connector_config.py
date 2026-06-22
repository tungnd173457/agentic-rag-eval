"""Generic ingest-source scan settings (provider-agnostic).

Per-provider credentials live in their own config (e.g. ``GDriveConfig``).
Folded into ``AppConfig``; read via ``app_config``. UPPERCASE field = env var.
"""
from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings


class ConnectorConfig(BaseSettings):
    CONNECTOR_SCAN_INTERVAL: PositiveInt = Field(default=120, description="Celery-Beat cadence for the source scan (seconds).")
    CONNECTOR_MAX_FILE_MB: PositiveInt = Field(default=50, description="Skip remote files larger than this (MB) before download when the size is known.")
