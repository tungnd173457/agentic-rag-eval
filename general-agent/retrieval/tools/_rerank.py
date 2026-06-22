"""Rerank used inside kb_search. Failure ⇒ keep hybrid order.

Two backends share the same request body and response shape (``results[].index``
+ ``relevance_score``), so they differ only in URL / model / api_key — one row
each in ``RERANKERS``, selected by ``RERANK_PROVIDER``. Add a backend = add a row.

Returns INDEXES into the input list (best first) so callers rerank whatever
objects the texts came from. The fallback (disabled, unknown provider, missing
key, API error) is the identity order truncated to top_n — search stays alive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import requests
import structlog

from config import AppConfig, app_config

log = structlog.get_logger(__name__)

TIMEOUT = 30.0


@dataclass(frozen=True)
class RerankSpec:
    """A rerank backend: where to POST, which model, and which key to send."""

    url: str
    model: Callable[[AppConfig], str]
    api_key: Callable[[AppConfig], str]


RERANKERS: dict[str, RerankSpec] = {
    "cohere": RerankSpec(
        url="https://api.cohere.com/v2/rerank",
        model=lambda c: c.COHERE_RERANK_MODEL,
        api_key=lambda c: c.COHERE_API_KEY or "",
    ),
    "openrouter": RerankSpec(
        url="https://openrouter.ai/api/v1/rerank",
        model=lambda c: c.OPENROUTER_RERANK_MODEL,
        api_key=lambda c: c.OPENROUTER_API_KEY,
    ),
}


def rerank(query: str, documents: list[str], *, top_n: int) -> list[int]:
    """Indexes into ``documents``, most relevant first, at most ``top_n``."""
    if not documents:
        return []
    fallback = list(range(min(top_n, len(documents))))
    if len(documents) == 1 or not app_config.USE_RERANKING:
        return fallback

    spec = RERANKERS.get(app_config.RERANK_PROVIDER.strip().lower())
    if spec is None:
        log.warning("rerank.unknown_provider_fallback", provider=app_config.RERANK_PROVIDER)
        return fallback
    api_key = spec.api_key(app_config)
    if not api_key:
        return fallback

    try:
        resp = requests.post(
            spec.url,
            json={
                "model": spec.model(app_config),
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        # Defend the "search stays alive" guarantee: a malformed response must
        # not surface as an IndexError in the caller — drop out-of-range and
        # duplicate indexes instead.
        order: list[int] = []
        for r in results:
            i = int(r["index"])
            if 0 <= i < len(documents) and i not in order:
                order.append(i)
        return order[:top_n]
    except Exception as exc:
        log.warning("rerank.failed_fallback_to_hybrid_order", provider=app_config.RERANK_PROVIDER, error=str(exc))
        return fallback
