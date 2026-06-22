"""Full-text store for 3A extractions in MongoDB.

3A writes the extraction markdown to S3 and the metadata record to Weaviate;
this additionally stores the FULL extracted text in MongoDB so downstream
consumers can read the whole document body by ``doc_id`` without an S3 round
trip — 3B does exactly that (``get_content``, with the S3 .md as fallback).
One document per ``doc_id`` (idempotent upsert), with fields:

  doc_id, extracted_s3_key, content, file_type, domain_level_1, domain_level_2,
  metadata (+ updated_at).

``pymongo`` is imported lazily so this module (and anything that imports it, e.g.
the 3A extractor) loads even where pymongo is not installed — the import only
happens the first time ``upsert_content`` actually writes.

Connection comes from ``config.app_config`` (env > .env > defaults):
  MONGODB_URL        — required, e.g. mongodb+srv://user:pass@host/...?...
  MONGODB_DB         — database name   (default "rag")
  MONGODB_COLLECTION — collection name (default "documents")
"""
from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from enum import Enum

from config import app_config

logger = logging.getLogger(__name__)


class MongoConfigError(RuntimeError):
    """MONGODB_URL missing or the client could not be created."""


# Cached client (pymongo manages its own connection pool, so one is enough).
_client = None


def get_database():
    """Return the configured database, creating the cached client on first use."""
    global _client
    if _client is None:
        url = app_config.MONGODB_URL.strip()
        if not url:
            raise MongoConfigError("MONGODB_URL is not set")
        try:
            from pymongo import MongoClient
        except ImportError as exc:  # pragma: no cover - depends on install
            raise MongoConfigError(
                "pymongo is not installed (pip install 'pymongo[srv]')"
            ) from exc
        _client = MongoClient(url)
    return _client[app_config.MONGODB_DB]


def _collection():
    """Return the configured documents collection, creating the client on first use."""
    return get_database()[app_config.MONGODB_COLLECTION]


def get_content(doc_id: str) -> dict | None:
    """Return the full-text record for ``doc_id`` (content, s3_key, file_type,
    domain_level_1/2, metadata, updated_at) or ``None`` when absent.

    3B reads the 3A markdown through this instead of fetching the S3 artifact
    (the .md on S3 stays as the audit copy / fallback). Raises MongoConfigError
    when MONGODB_URL is unset — callers decide whether that is fatal.
    """
    return _collection().find_one({"doc_id": doc_id}, {"_id": False})


def upsert_content(
    doc_id: str,
    extracted_s3_key: str,
    content: str,
    *,
    file_type: str = "",
    domain_level_1: str = "",
    domain_level_2: str = "",
    metadata: dict | None = None,
) -> None:
    """Create-or-replace the full-text record for ``doc_id`` (idempotent).

    Stores the processed-artifact key under ``extracted_s3_key`` — NOT ``s3_key``,
    which is the lifecycle layer's raw-upload key (``mark_uploaded``) that the
    worker re-ingests from. Writing it here would clobber that key and break
    retries/reconcile (dedup rejects a ``processed/…`` key).
    """
    _collection().update_one(
        {"doc_id": doc_id},
        {"$set": {
            "doc_id": doc_id,
            "extracted_s3_key": extracted_s3_key,
            "content": content,
            "file_type": file_type,
            "domain_level_1": domain_level_1,
            "domain_level_2": domain_level_2,
            "metadata": metadata or {},
            "updated_at": datetime.now(UTC).isoformat(),
        }},
        upsert=True,
    )
    logger.info(
        "Upserted MongoDB content doc_id=%s extracted_s3_key=%s type=%s domain=%s/%s "
        "metadata_fields=%d chars=%d",
        doc_id, extracted_s3_key, file_type, domain_level_1, domain_level_2,
        len(metadata or {}), len(content),
    )


# =====================================================================
# Orchestration lifecycle — same rag.documents collection, disjoint fields.
#
# The fields below (status/attempts/error/counts/… + the raw ``s3_key``) are
# written by the upload API, the Celery worker and the reconciler via $set/$inc.
# They never touch ``content``/``extracted_s3_key`` (3A's fields), and
# ``upsert_content`` never touches ``status`` or the raw ``s3_key`` — so the two
# layers merge into one record per doc_id without clobbering. (The worker
# re-ingests from this raw ``s3_key``; 3A's processed key lives separately in
# ``extracted_s3_key``.)
# =====================================================================


class IngestStatus(str, Enum):
    """Orchestration lifecycle of the whole ingestion flow (Mongo rag.documents).

    Distinct from the Weaviate registry's DocStatus (pipeline-internal step
    status) — different store, different consumer.
    """

    UPLOADED = "uploaded"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


_indexes_ready = False


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_indexes() -> None:
    """Index (status, updated_at) for the reconciler query — once per process."""
    global _indexes_ready
    if _indexes_ready:
        return
    _collection().create_index([("status", 1), ("updated_at", 1)])
    _indexes_ready = True


def mark_uploaded(
    doc_id: str,
    s3_key: str,
    filename: str,
    source: str,
    content_type: str,
    size_bytes: int,
) -> None:
    """Create-or-reset the lifecycle record at upload time (status=uploaded).

    Merges with any existing 3A content via $set on a disjoint field set;
    $inc bumps attempts, $setOnInsert preserves the original created_at.
    """
    _ensure_indexes()
    now = _now_iso()
    _collection().update_one(
        {"doc_id": doc_id},
        {
            "$set": {
                "doc_id": doc_id,
                "s3_key": s3_key,
                "filename": filename,
                "source": source,
                "content_type": content_type,
                "size_bytes": size_bytes,
                "status": IngestStatus.UPLOADED.value,
                "error": None,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
            "$inc": {"attempts": 1},
        },
        upsert=True,
    )
    logger.info("Mongo lifecycle uploaded doc_id=%s s3_key=%s", doc_id, s3_key)


def set_status(doc_id: str, status: str) -> None:
    _collection().update_one(
        {"doc_id": doc_id}, {"$set": {"status": status, "updated_at": _now_iso()}}
    )


def mark_indexed(doc_id: str, parent_count: int, child_count: int) -> None:
    _collection().update_one(
        {"doc_id": doc_id},
        {"$set": {
            "status": IngestStatus.INDEXED.value,
            "parent_count": parent_count,
            "child_count": child_count,
            "error": None,
            "updated_at": _now_iso(),
        }},
    )


def mark_failed(doc_id: str, error: str) -> None:
    _collection().update_one(
        {"doc_id": doc_id},
        {"$set": {"status": IngestStatus.FAILED.value, "error": error, "updated_at": _now_iso()}},
    )


def inc_attempts(doc_id: str) -> None:
    _collection().update_one(
        {"doc_id": doc_id}, {"$inc": {"attempts": 1}, "$set": {"updated_at": _now_iso()}}
    )


def delete_document(doc_id: str) -> int:
    """Delete the record for ``doc_id``. Returns the deleted count (0 if absent).

    Idempotent — deleting an already-gone record is a no-op, so the cross-store
    document-delete flow can be retried safely after a partial failure.
    """
    deleted = _collection().delete_one({"doc_id": doc_id}).deleted_count
    if deleted:
        logger.info("Deleted MongoDB record doc_id=%s", doc_id)
    return deleted


def find_stuck(statuses: list[str], before_iso: str) -> list[dict]:
    """Records whose status is in ``statuses`` and were last touched before
    ``before_iso`` — the reconciler's candidate set."""
    return list(
        _collection().find(
            {"status": {"$in": list(statuses)}, "updated_at": {"$lt": before_iso}},
            {"_id": False},
        )
    )


# Active lifecycle states worth polling (still in progress).
ACTIVE_STATUSES = [IngestStatus.UPLOADED.value, IngestStatus.INDEXING.value]

# Fields safe + cheap to return for list/poll views — excludes the heavy
# `content`/`payload`/`metadata` blobs and internal `s3_key`/`attempts`.
_STATUS_PROJECTION = {
    "_id": False,
    "doc_id": True, "status": True, "error": True,
    "filename": True, "created_at": True, "updated_at": True,
    "domain_level_1": True, "domain_level_2": True,
    "parent_count": True, "child_count": True,
}


def _status_query(statuses: list[str] | None, q: str | None = None) -> dict:
    query: dict = {} if statuses is None else {"status": {"$in": statuses}}
    if q:
        query["filename"] = {"$regex": re.escape(q), "$options": "i"}
    return query


def list_status(
    statuses: list[str] | None, limit: int, offset: int, q: str | None = None
) -> list[dict]:
    """Documents newest-first by ``updated_at``, optionally restricted to a set of
    lifecycle statuses (``None`` = all) and/or a filename substring ``q``.
    Heavy/internal fields are projected out."""
    return list(
        _collection()
        .find(_status_query(statuses, q), _STATUS_PROJECTION)
        .sort("updated_at", -1)
        .skip(offset)
        .limit(limit)
    )


def count_status(statuses: list[str] | None, q: str | None = None) -> int:
    """Total documents matching ``statuses`` (``None`` = all) and optional ``q``,
    ignoring pagination."""
    return _collection().count_documents(_status_query(statuses, q))


# =====================================================================
# Connector cursors — one tiny ``connector_state`` doc per ingest source.
# Holds the last delta token (Drive's max modifiedTime, etc.) so a poll only
# fetches what changed; the content-hash dedup remains the authoritative guard.
# =====================================================================

def _connector_state():
    return get_database()["connector_state"]


def get_connector_cursor(source: str) -> str | None:
    """Last persisted scan cursor for ``source`` (None on the first ever scan)."""
    rec = _connector_state().find_one({"source": source}, {"_id": False})
    return rec.get("cursor") if rec else None


def set_connector_cursor(source: str, cursor: str) -> None:
    """Persist the scan cursor for ``source`` (idempotent upsert)."""
    _connector_state().update_one(
        {"source": source},
        {"$set": {"source": source, "cursor": cursor, "updated_at": _now_iso()}},
        upsert=True,
    )
    logger.info("connector cursor source=%s -> %s", source, cursor)
