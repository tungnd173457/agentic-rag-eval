"""Provider-dispatched text embedding, shared by ingestion (indexing children)
and retrieval (embedding the agent's search queries with the SAME model).

Dispatch key: EMBED_PROVIDER ("ollama" | "openai"), independent of LLM_PROVIDER.
Both providers speak the OpenAI embeddings protocol (ollama via its /v1
endpoint), so there is a single code path — only the client (base_url/key/model)
differs. The client is built in ``services.llm.client.resolve_embed`` from the
same env (via config.app_config) that configures the rest of the pipeline —
this is what prevents the query-embedded-with-a-different-model-than-the-index
bug class.

Raises EmbeddingError on any failure — embedding is fatal for both indexing
and search, never best-effort.
"""
from __future__ import annotations

import logging

from openai import OpenAIError

from config import app_config
from services.llm.client import ClientConfigError, resolve_embed

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Transport failure, bad response shape, or misconfiguration."""


def embed_texts(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Embed ``texts`` with the EMBED_PROVIDER backend. One vector per input.

    Texts are capped at EMBED_MAX_CHARS and sent in EMBED_BATCH_SIZE batches;
    each batch's vectors are reordered by the API-returned index before being
    appended, so output order matches input order.
    """
    if not texts:
        return []
    cfg = app_config
    try:
        client, default_model = resolve_embed(cfg)
    except ClientConfigError as exc:
        raise EmbeddingError(str(exc)) from exc

    mdl = model or default_model
    prepared = [t[: cfg.EMBED_MAX_CHARS] for t in texts]
    batch_size = max(1, cfg.EMBED_BATCH_SIZE)

    out: list[list[float]] = []
    for i in range(0, len(prepared), batch_size):
        batch = prepared[i : i + batch_size]
        try:
            resp = client.embeddings.create(model=mdl, input=batch)
        except OpenAIError as exc:
            raise EmbeddingError(f"embed {mdl} → {exc}") from exc
        items = resp.data
        if len(items) != len(batch):
            raise EmbeddingError(
                f"embed {mdl} returned {len(items)} vectors for {len(batch)} texts"
            )
        items = sorted(items, key=lambda d: d.index)
        out.extend(item.embedding for item in items)
    return out
