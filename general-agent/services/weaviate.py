"""3A document registry in Weaviate (replaces the old DynamoDB ROOT entry).

One object per document, keyed by ``uuid5(doc_id)`` so writes are idempotent
upserts and lookups need no secondary index. The collection has no vectorizer
(``self_provided``) — this is a metadata record, NOT a searchable chunk;
embedding happens later (3B/3C), not here.

A handful of scalar properties (status / domain / content_hash / …) are mirrored
as typed fields for filtering; the full extraction meta (incl. ``headings`` and
the per-domain ``metadata`` dict) is kept verbatim in the ``payload`` text field
so arbitrary fields survive without schema churn.

All access goes through the v4 SDK via ``db.weaviate`` collection handles.

Environment:
  WEAVIATE_DOC_CLASS — collection name (default "Document")
  (+ the WEAVIATE_* connection vars read by ``db.weaviate``)
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime

from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter
from weaviate.collections.collection import Collection
from weaviate.exceptions import WeaviateBaseError

from config import app_config
from db import weaviate as wc

logger = logging.getLogger(__name__)


def get_collection(name: str) -> Collection:
    """Handle for collection ``name`` (no network call; used for data/query ops)."""
    return wc.get_client().collections.get(name)

# Fixed namespace for deterministic object ids (value is arbitrary but stable).
_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")

_schema_ready = False
_chunk_schema_ready = False


def _doc_class() -> str:
    return app_config.WEAVIATE_DOC_CLASS


def _chunk_class() -> str:
    return app_config.WEAVIATE_CHUNK_CLASS


def _field(name: str) -> Property:
    """Exact-match text property (whole value = one token, for Equal filters)."""
    return Property(name=name, data_type=DataType.TEXT, tokenization=Tokenization.FIELD)


def _blob(name: str) -> Property:
    return Property(name=name, data_type=DataType.TEXT)


def _doc_properties() -> list[Property]:
    return [
        _field("doc_id"),
        _field("status"),
        _field("content_hash"),
        _field("source"),
        _field("file_type"),
        _field("domain_level_1"),
        _field("domain_level_2"),
        _blob("filename"),
        _field("updated_at"),
        _blob("payload"),  # full extraction meta as JSON
    ]


def _chunk_properties() -> list[Property]:
    """The Chunk collection: parents (context, no vector) + children (BYO
    vector, embed/search unit) of every document, discriminated by ``kind``.
    level2/level3 are filter properties — they replace the old per-level2
    Bedrock KB routing."""
    return [
        _field("doc_id"),
        _field("kind"),                 # "parent" | "child"
        _field("chunk_id"),             # parent_NNNNNNNNNN / child_NNNNNNNNNN
        _field("parent_id"),            # children: owner parent; parents: ""
        _field("doc_hash"),             # sha256 of the chunk body
        _field("domain_level_1"),
        _field("domain_level_2"),
        _field("domain_level_3"),       # primary level3
        Property(name="domains_level_3", data_type=DataType.TEXT_ARRAY),
        Property(name="domain_tags_level_3", data_type=DataType.TEXT_ARRAY),
        Property(name="position", data_type=DataType.INT),
        Property(name="char_count", data_type=DataType.INT),
        _blob("title"),
        _blob("text"),
        _blob("s3_key"),
        _blob("filename"),
        _field("updated_at"),
    ]


def ensure_schema() -> None:
    """Create the document collection if it does not already exist (idempotent)."""
    global _schema_ready
    if _schema_ready:
        return
    name = _doc_class()
    try:
        client = wc.get_client()
        if not client.collections.exists(name):
            client.collections.create(
                name=name,
                description="3A document/extraction registry (status, domain, meta).",
                vector_config=Configure.Vectors.self_provided(),
                properties=_doc_properties(),
            )
            logger.info("Created Weaviate collection %s", name)
    except WeaviateBaseError as exc:
        raise wc.WeaviateError(f"ensure_schema {name}: {exc}") from exc
    _schema_ready = True


def ensure_chunk_schema() -> None:
    """Create the Chunk collection if it does not already exist (idempotent)."""
    global _chunk_schema_ready
    if _chunk_schema_ready:
        return
    name = _chunk_class()
    try:
        client = wc.get_client()
        if not client.collections.exists(name):
            client.collections.create(
                name=name,
                description="3B parent/child chunks: parent=context, child=embed/search.",
                vector_config=Configure.Vectors.self_provided(
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE
                    ),
                ),
                properties=_chunk_properties(),
            )
            logger.info("Created Weaviate collection %s", name)
    except WeaviateBaseError as exc:
        raise wc.WeaviateError(f"ensure_chunk_schema {name}: {exc}") from exc
    _chunk_schema_ready = True


def _doc_uuid(doc_id: str) -> str:
    return str(uuid.uuid5(_NS, doc_id))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _s(v) -> str:
    return "" if v is None else str(v)


def upsert_document(meta: dict, *, status: str, content_hash: str = "") -> None:
    """Create-or-replace the document's registry object from its 3A meta.

    ``meta`` is the extraction_meta dict (doc_id, source, filename, file_type,
    extraction_method, page_count, char_count, domain_level_1/2, metadata,
    headings). ``status`` and ``content_hash`` are operational fields not in the
    S3 meta file.
    """
    ensure_schema()
    class_name = _doc_class()
    doc_id = _s(meta.get("doc_id"))
    obj_id = _doc_uuid(doc_id)

    entry = {
        **meta,
        "status": status,
        "content_hash": content_hash or meta.get("content_hash", ""),
        "updated_at": _now_iso(),
    }
    props = {
        "doc_id": doc_id,
        "status": status,
        "content_hash": _s(entry["content_hash"]),
        "source": _s(meta.get("source")),
        "file_type": _s(meta.get("file_type")),
        "domain_level_1": _s(meta.get("domain_level_1")),
        "domain_level_2": _s(meta.get("domain_level_2")),
        "filename": _s(meta.get("filename")),
        "updated_at": entry["updated_at"],
        "payload": json.dumps(entry, ensure_ascii=False, default=str),
    }
    coll = get_collection(class_name)
    try:
        if coll.data.exists(obj_id):
            coll.data.replace(uuid=obj_id, properties=props)
        else:
            coll.data.insert(properties=props, uuid=obj_id)
    except WeaviateBaseError as exc:
        raise wc.WeaviateError(f"upsert_document doc_id={doc_id}: {exc}") from exc
    logger.info("Upserted Weaviate %s doc_id=%s status=%s", class_name, doc_id, status)


def get_document(doc_id: str) -> dict | None:
    """Return the registry object's properties for ``doc_id``, or None if absent.

    Used by the pre-3A dedup check to detect a re-upload of the same doc_id.
    """
    ensure_schema()
    class_name = _doc_class()
    obj_id = _doc_uuid(doc_id)
    coll = get_collection(class_name)
    try:
        obj = coll.query.fetch_object_by_id(obj_id)
    except WeaviateBaseError as exc:
        raise wc.WeaviateError(f"get_document doc_id={doc_id}: {exc}") from exc
    return dict(obj.properties) if obj else None


def resolve_doc_id_prefix(prefix: str) -> str | None:
    """Full ``doc_id`` for a citation prefix, or None if nothing matches.

    Citations/source chips expose only ``doc_id[:8]`` (see search_kb's
    ``_chunk_ref``); this maps that short prefix back to the stored full id so a
    detail lookup can succeed. Queries the Chunk collection (parents carry the
    full doc_id). Single source of truth — ``retrieval._store.resolve_doc_id``
    delegates here.
    """
    prefix = (prefix or "").strip()
    if not prefix:
        return None
    where = Filter.all_of([
        Filter.by_property("doc_id").like(prefix + "*"),
        Filter.by_property("kind").equal("parent"),
    ])
    coll = get_collection(_chunk_class())
    try:
        result = coll.query.fetch_objects(
            filters=where, limit=1, return_properties=["doc_id"],
        )
    except WeaviateBaseError as exc:
        raise wc.WeaviateError(f"resolve_doc_id_prefix prefix={prefix}: {exc}") from exc
    if not result.objects:
        return None
    value = result.objects[0].properties.get("doc_id")
    return str(value) if value is not None else None


def chunk_uuid(doc_id: str, chunk_id: str) -> str:
    """Deterministic Weaviate object id for one chunk: uuid5(doc_id/chunk_id).

    Deterministic so a re-split UPSERTS the same objects instead of duplicating,
    and so split_meta's ``vector_id`` can be derived without a lookup.
    """
    return str(uuid.uuid5(_NS, f"{doc_id}/{chunk_id}"))


def upsert_chunks(objects: list[dict], batch_size: int | None = None) -> int:
    """Batch-upsert chunk objects (``{"id", "properties", "vector"?}``).

    Parents carry no vector (context only); children carry the BYO embedding.
    Uses ``insert_many`` in slices of ``batch_size`` (defaults to
    ``app_config.WEAVIATE_BATCH_SIZE``); ids are deterministic (``chunk_uuid``)
    so re-runs overwrite. Raises WeaviateError if any object in a batch reports
    an error — indexing must not be partial silently.
    """
    if not objects:
        return 0
    if batch_size is None:
        batch_size = app_config.WEAVIATE_BATCH_SIZE
    ensure_chunk_schema()
    class_name = _chunk_class()
    coll = get_collection(class_name)
    total = 0
    for i in range(0, len(objects), batch_size):
        batch = objects[i : i + batch_size]
        data_objects = [
            DataObject(
                properties=o.get("properties") or {},
                uuid=o.get("id"),
                vector=o.get("vector"),
            )
            for o in batch
        ]
        try:
            result = coll.data.insert_many(data_objects)
        except WeaviateBaseError as exc:
            raise wc.WeaviateError(f"batch upsert {class_name}: {exc}") from exc
        # insert_many returns per-object errors keyed by batch index rather than
        # raising — surface the first one WITH the failing object's identity
        # (doc_id / chunk_id / object id) so the log points at the bad chunk.
        if result.has_errors:
            idx, err = next(iter(result.errors.items()))
            props = (batch[idx].get("properties") or {}) if idx < len(batch) else {}
            detail = (
                f"batch upsert {class_name} failed: doc_id={props.get('doc_id', '')} "
                f"chunk_id={props.get('chunk_id', '')} kind={props.get('kind', '')} "
                f"object_id={batch[idx].get('id', '') if idx < len(batch) else ''} "
                f"error={err.message}"
            )
            logger.error(detail)
            raise wc.WeaviateError(detail)
        total += len(batch)
    logger.info("Upserted %d Weaviate %s chunks", total, class_name)
    return total


def delete_chunks(doc_id: str) -> int:
    """Delete every chunk object of ``doc_id`` from the Chunk collection.

    The vector-store half of 3B's delete-then-reload (develop's
    ``delete_document`` convention): when a document is re-split, the old
    parent/child chunk set is invalid — purge it so the search index never
    serves stale text, duplicate hits, or chunks whose ``s3_key``/``parent_id``
    no longer exist. 3C then embeds + upserts the fresh set.

    Returns the number deleted. Returns 0 (no-op) when the Chunk collection
    does not exist yet — 3C may not be deployed.
    """
    if not doc_id:
        return 0
    class_name = _chunk_class()
    client = wc.get_client()
    try:
        if not client.collections.exists(class_name):
            return 0
        coll = client.collections.get(class_name)
        where = Filter.by_property("doc_id").equal(doc_id)
        deleted = 0
        # Batch delete is capped per call (default 10k) — loop until nothing matches.
        while True:
            result = coll.data.delete_many(where=where)
            matches = result.matches or 0
            successful = result.successful or 0
            deleted += successful
            if matches == 0 or successful == 0:
                break
    except WeaviateBaseError as exc:
        raise wc.WeaviateError(f"delete_chunks doc_id={doc_id}: {exc}") from exc
    if deleted:
        logger.info("Deleted %d Weaviate %s chunks for doc_id=%s", deleted, class_name, doc_id)
    return deleted


def delete_document(doc_id: str) -> None:
    """Delete the registry object for ``doc_id`` from the Document collection.

    The registry half of a full document delete (the Chunk half is
    ``delete_chunks``). Idempotent: a missing object — or a not-yet-created
    collection — is a no-op, so retries after a partial failure are safe.
    Called LAST in the delete flow: the registry payload is the fallback source
    of ``source``/``domain_level_2`` used to build the S3 prefixes, so it must
    outlive the other steps.
    """
    if not doc_id:
        return
    class_name = _doc_class()
    client = wc.get_client()
    try:
        if not client.collections.exists(class_name):
            return
        client.collections.get(class_name).data.delete_by_id(_doc_uuid(doc_id))
    except WeaviateBaseError as exc:
        raise wc.WeaviateError(f"delete_document doc_id={doc_id}: {exc}") from exc
    logger.info("Deleted Weaviate %s registry object for doc_id=%s", class_name, doc_id)


def find_by_content_hash(content_hash: str) -> list[dict]:
    """Return ``[{"doc_id", "status"}, …]`` for documents sharing ``content_hash``.

    Used by the dedup check to spot a different doc_id with identical content.
    """
    if not content_hash:
        return []
    ensure_schema()
    class_name = _doc_class()
    coll = get_collection(class_name)
    try:
        result = coll.query.fetch_objects(
            filters=Filter.by_property("content_hash").equal(content_hash),
            return_properties=["doc_id", "status"],
            limit=100,
        )
    except WeaviateBaseError as exc:
        raise wc.WeaviateError(f"find_by_content_hash: {exc}") from exc
    return [
        {"doc_id": o.properties.get("doc_id", ""), "status": o.properties.get("status", "")}
        for o in result.objects
    ]
