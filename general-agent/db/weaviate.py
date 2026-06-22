"""Weaviate v4 SDK connection management. Connection comes from env.

3A no longer writes the document registry to DynamoDB — it upserts a document
metadata object into Weaviate instead (see ``repo``). This module owns the
single, reused ``WeaviateClient``; ``repo`` and ``retrieval.tools._store`` get
collection handles from it.

The v4 client talks to Weaviate over **gRPC** for all queries/batch (REST only
for schema + CRUD-by-id), so a gRPC endpoint must be reachable — see
``docs/research/weaviate-python-sdk-v4.md``.

Connection comes from ``config.app_config`` (env > .env > defaults):
  WEAVIATE_URL          — HTTP base URL incl. scheme/host[/port], e.g.
                          https://weaviate.example.com  (a trailing /v1 is tolerated)
  WEAVIATE_API_KEY      — optional; passed as ``Auth.api_key`` bearer credential
  WEAVIATE_GRPC_HOST    — optional; defaults to the WEAVIATE_URL host
  WEAVIATE_GRPC_PORT    — optional; defaults to 50051
  WEAVIATE_GRPC_SECURE  — optional; defaults to the WEAVIATE_URL scheme (https→True)
  WEAVIATE_TIMEOUT      — query/init timeout seconds (default 30)
"""
from __future__ import annotations

import logging
from urllib.parse import urlparse

import weaviate
from weaviate.classes.init import AdditionalConfig, Auth, Timeout
from weaviate.client import WeaviateClient
from weaviate.exceptions import WeaviateBaseError

from config import AppConfig, app_config

logger = logging.getLogger(__name__)


class WeaviateError(RuntimeError):
    """Any failure talking to Weaviate (connection, gRPC, or query error)."""


# Module-level singleton — connecting opens a gRPC channel + HTTP session, so we
# create the client once and reuse it across Lambda invocations / HTTP requests.
_client: WeaviateClient | None = None


def _conn_params(cfg: AppConfig) -> dict:
    """Translate the env config into ``connect_to_custom`` kwargs."""
    raw = cfg.WEAVIATE_URL.strip().rstrip("/")
    if not raw:
        raise WeaviateError("WEAVIATE_URL is not set")
    if raw.endswith("/v1"):
        raw = raw[: -len("/v1")]
    parsed = urlparse(raw)
    if not parsed.hostname:
        raise WeaviateError(f"WEAVIATE_URL is not a valid URL: {cfg.WEAVIATE_URL!r}")

    http_secure = parsed.scheme == "https"
    http_host = parsed.hostname
    http_port = parsed.port or (443 if http_secure else 80)

    grpc_secure = http_secure if cfg.WEAVIATE_GRPC_SECURE is None else cfg.WEAVIATE_GRPC_SECURE
    return {
        "http_host": http_host,
        "http_port": http_port,
        "http_secure": http_secure,
        "grpc_host": cfg.WEAVIATE_GRPC_HOST.strip() or http_host,
        "grpc_port": cfg.WEAVIATE_GRPC_PORT,
        "grpc_secure": grpc_secure,
    }


def get_client() -> WeaviateClient:
    """Return the shared, connected ``WeaviateClient`` (lazily created)."""
    global _client
    if _client is not None:
        return _client
    cfg = app_config
    params = _conn_params(cfg)
    key = cfg.WEAVIATE_API_KEY.strip()
    timeout = int(cfg.WEAVIATE_TIMEOUT)
    try:
        client = weaviate.connect_to_custom(
            **params,
            auth_credentials=Auth.api_key(key) if key else None,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=timeout, query=timeout, insert=timeout)
            ),
        )
    except WeaviateBaseError as exc:
        raise WeaviateError(f"could not connect to Weaviate: {exc}") from exc
    _client = client
    logger.info(
        "Connected to Weaviate http=%s:%s grpc=%s:%s",
        params["http_host"], params["http_port"],
        params["grpc_host"], params["grpc_port"],
    )
    return _client


def close_client() -> None:
    """Close the shared client (releases the gRPC channel + HTTP session)."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except WeaviateBaseError as exc:
            logger.warning("Error closing Weaviate client: %s", exc)
        _client = None
