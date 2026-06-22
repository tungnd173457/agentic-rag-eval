"""Backend HTTP API (FastAPI) settings."""

from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings


class ApiConfig(BaseSettings):
    """FastAPI server host/port/CORS + cross-cutting log level."""

    API_HOST: str = Field(default="0.0.0.0", description="Bind host for uvicorn.")
    API_PORT: PositiveInt = Field(default=8000, description="Bind port for uvicorn.")
    API_CORS_ALLOW_ORIGINS: list[str] = Field(default=["*"], description="CORS allowed origins.")
    LOG_LEVEL: str = Field(default="INFO", description="Runtime log level (shared with retrieval).")
