"""Weaviate vector-store connection + class-name settings."""

from pydantic import Field, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings


class WeaviateConfig(BaseSettings):
    """Weaviate connection and collection (class) configuration."""

    WEAVIATE_URL: str = Field(default="", description="Weaviate HTTP URL (scheme://host[:port]).")
    WEAVIATE_API_KEY: str = Field(default="", description="Weaviate API key (empty = anonymous).")
    WEAVIATE_TIMEOUT: PositiveFloat = Field(default=30.0, description="Connect/query/insert timeout (seconds).")
    WEAVIATE_DOC_CLASS: str = Field(default="Document", description="Collection (class) name for the document registry.")
    WEAVIATE_CHUNK_CLASS: str = Field(default="Chunk", description="Collection (class) name for chunks.")
    WEAVIATE_GRPC_HOST: str = Field(default="", description="gRPC host; '' derives from WEAVIATE_URL host.")
    WEAVIATE_GRPC_PORT: PositiveInt = Field(default=50051, description="gRPC port (v4 SDK requires gRPC).")
    WEAVIATE_GRPC_SECURE: bool | None = Field(default=None, description="gRPC TLS; None derives from WEAVIATE_URL scheme.")
    WEAVIATE_BATCH_SIZE: PositiveInt = Field(default=25, description="Objects per insert_many gRPC batch. Keep small: large embeddings (e.g. 3072-dim) can exceed a fronting proxy's body limit and return HTTP 413.")
