"""Read-side access to the Weaviate ``Chunk`` collection (and Mongo full text).

This is the REAL index ingestion writes (services.weaviate Chunk schema):
children carry the BYO embedding and are the hybrid-search unit; parents carry
the context text and a sequential doc-order ``position``. All queries go through
the v4 SDK collection handle from services.weaviate.

Every public function raises StoreError on backend failure — the tool layer
converts that into guidance text for the agent (never an exception up the loop).
"""
from __future__ import annotations

import json

import structlog
from weaviate.classes.query import Filter, MetadataQuery, Sort
from weaviate.exceptions import WeaviateBaseError

from config import app_config
from db import mongo
from db import weaviate as wc
from services import weaviate as weaviate_repo

log = structlog.get_logger(__name__)


class StoreError(RuntimeError):
    """Weaviate unreachable or returned a query error."""


CHILD_FIELDS = [
    "doc_id", "chunk_id", "parent_id", "domain_level_2", "title", "text",
    "position", "filename", "updated_at",
]
PARENT_FIELDS = [
    "doc_id", "chunk_id", "domain_level_2", "title", "text", "position",
    "char_count", "filename", "updated_at",
]


def _chunk_collection():
    return weaviate_repo.get_collection(app_config.WEAVIATE_CHUNK_CLASS)


def _row(obj, *, with_score: bool = False) -> dict:
    """Flatten a v4 ``Object`` to the flat dict shape the tool layer expects.

    Old GraphQL returned ``{prop: value, _additional: {score}}``; we mirror that
    so downstream code (search_kb / get_document) is unchanged."""
    row = dict(obj.properties)
    if with_score:
        score = getattr(obj.metadata, "score", None)
        row["_additional"] = {"score": score}
    return row


def hybrid_search_children(
    query: str,
    vector: list[float],
    *,
    alpha: float,
    limit: int,
    domains: list[str] | None = None,
) -> list[dict]:
    """Hybrid (BM25 + vector) search over child chunks, best first."""
    operands = [Filter.by_property("kind").equal("child")]
    if domains:
        operands.append(Filter.by_property("domain_level_2").contains_any(domains))
    try:
        result = _chunk_collection().query.hybrid(
            query=query,
            vector=vector,
            alpha=alpha,
            filters=Filter.all_of(operands),
            limit=limit,
            return_metadata=MetadataQuery(score=True),
            return_properties=CHILD_FIELDS,
        )
    except WeaviateBaseError as exc:
        raise StoreError(str(exc)) from exc
    return [_row(o, with_score=True) for o in result.objects]


def fetch_parents(refs: list[tuple[str, str]]) -> list[dict]:
    """Parent rows for ``[(doc_id, parent_chunk_id), …]`` (order NOT guaranteed —
    chunk_id alone is per-document, so each ref is scoped by doc_id)."""
    if not refs:
        return []
    pairs = [
        Filter.all_of([
            Filter.by_property("doc_id").equal(doc_id),
            Filter.by_property("chunk_id").equal(pid),
        ])
        for doc_id, pid in refs
    ]
    where = Filter.all_of([
        Filter.by_property("kind").equal("parent"),
        Filter.any_of(pairs),
    ])
    try:
        result = _chunk_collection().query.fetch_objects(
            filters=where, limit=len(refs), return_properties=PARENT_FIELDS,
        )
    except WeaviateBaseError as exc:
        raise StoreError(str(exc)) from exc
    return [_row(o) for o in result.objects]


def fetch_parents_by_position(doc_id: str, positions: list[int]) -> list[dict]:
    """Parent rows of ``doc_id`` whose ``position`` is in ``positions``, sorted in
    doc order. Positions are sequential ints from the splitter."""
    if not positions:
        return []
    where = Filter.all_of([
        Filter.by_property("doc_id").equal(doc_id),
        Filter.by_property("kind").equal("parent"),
        Filter.any_of([
            Filter.by_property("position").equal(p) for p in positions
        ]),
    ])
    try:
        result = _chunk_collection().query.fetch_objects(
            filters=where,
            sort=Sort.by_property("position", ascending=True),
            limit=len(positions),
            return_properties=PARENT_FIELDS,
        )
    except WeaviateBaseError as exc:
        raise StoreError(str(exc)) from exc
    return [_row(o) for o in result.objects]


def resolve_doc_id(prefix: str) -> str | None:
    """Full doc_id for a citation prefix (chunk refs are ``doc_id[:8]#p<pos>``)."""
    try:
        return weaviate_repo.resolve_doc_id_prefix(prefix)
    except wc.WeaviateError as exc:
        raise StoreError(str(exc)) from exc


def count_parents(doc_id: str) -> int:
    """Total parent sections of ``doc_id`` (for "section X of Y" footers)."""
    where = Filter.all_of([
        Filter.by_property("doc_id").equal(doc_id),
        Filter.by_property("kind").equal("parent"),
    ])
    try:
        result = _chunk_collection().aggregate.over_all(filters=where, total_count=True)
    except WeaviateBaseError as exc:
        raise StoreError(str(exc)) from exc
    return result.total_count or 0


def doc_info(doc_id: str) -> dict | None:
    """Document-registry metadata (filename, domain, page/char counts) or None."""
    try:
        props = weaviate_repo.get_document(doc_id)
    except wc.WeaviateError as exc:
        raise StoreError(str(exc)) from exc
    if not props:
        return None
    payload: dict = {}
    try:
        payload = json.loads(props.get("payload") or "{}")
    except ValueError:
        pass
    return {
        "doc_id": doc_id,
        "filename": props.get("filename", ""),
        "domain_level_2": props.get("domain_level_2", ""),
        "updated_at": props.get("updated_at", ""),
        "page_count": payload.get("page_count"),
        "char_count": payload.get("char_count"),
    }


def get_full_text(doc_id: str) -> str | None:
    """Full extracted markdown from MongoDB — fallback when Weaviate has no
    parents for the doc. Best-effort: any Mongo problem returns None."""
    try:
        record = mongo.get_content(doc_id)
    except Exception as exc:  # MongoConfigError, network, …
        log.warning("store.mongo_fallback_failed", doc_id=doc_id, error=str(exc))
        return None
    return (record or {}).get("content") or None
