"""MongoDB connection settings (3A full-text store + ingestion lifecycle + conversations)."""

from pydantic import Field
from pydantic_settings import BaseSettings


class MongoConfig(BaseSettings):
    """MongoDB connection — one record per ``doc_id`` in ``MONGODB_DB.MONGODB_COLLECTION``."""

    MONGODB_URL: str = Field(
        default="",
        description="Connection string, e.g. mongodb+srv://user:pass@host/...?retryWrites=true&w=majority.",
    )
    MONGODB_DB: str = Field(default="rag", description="Database name.")
    MONGODB_COLLECTION: str = Field(default="documents", description="Documents collection name.")
