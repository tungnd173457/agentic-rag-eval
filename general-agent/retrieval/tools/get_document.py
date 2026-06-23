"""kb_get_document — positions-list document reader (the "fetch" of the
search-then-fetch pair).

Given a doc_id (the prefix shown in kb_search refs) and an explicit list of
section positions, returns those parent sections in doc order, each labelled
[doc_id#p<position>] so the agent can cite them. kb_search already returns the
hit's own full text, so the agent asks here only for the NEIGHBOURING positions
it has not read (e.g. [3, 5] for a hit at p4). Stateless — every call is
independent. Mongo full text is the fallback when Weaviate has no parents.
"""
from __future__ import annotations

import structlog

from config import app_config
from retrieval.tools import _store
from retrieval.tools.base import ToolResult, ToolSpec

log = structlog.get_logger(__name__)

CHARS_PER_TOKEN = 3          # same heuristic as search_kb

BAD_DOC_ID = (
    "doc_id not found. Use a doc_id exactly as shown in kb_search results — the "
    "part before '#p' in a ref like 'a1b2c3d4#p4'."
)
BACKEND_DOWN = (
    "Document backend unavailable right now — do not retry. Answer from the "
    "information you already have and mention the limitation."
)
NO_POSITIONS = (
    "Provide at least one section position from a kb_search ref, e.g. "
    "positions=[4] (or [3,4,5] to include neighbours)."
)

DESCRIPTION = (
    "Read the NEIGHBOURING knowledge-base sections around a kb_search hit — the "
    "context you do NOT already have. Pass the doc_id (the part before '#p' in a "
    "kb_search ref like 'a1b2c3d4#p4') and a list of positions (the '#p<n>' "
    "numbers).\n\n"
    "The position in a kb_search ref is the section you have ALREADY READ — "
    "kb_search returned its full text — so never pass it back here; that only "
    "returns text you already have and wastes the call. Request only the "
    "neighbours you have NOT seen: hit at p4 → positions=[3,5] (before and after, "
    "NOT [4] and NOT [3,4,5]); hit at p1 → positions=[2] since p1 is the first "
    "section and there is nothing before it (NOT [1]).\n\n"
    "The response header shows how many sections the document has, so you know the "
    "valid range. Use this only when a hit looks relevant but you need the "
    "surrounding clauses, table rows, or context to answer precisely — if the "
    "matched section alone is enough, do not call this tool at all."
)


def spec() -> ToolSpec:
    return ToolSpec(
        name="kb_get_document",
        description=DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": (
                        "Document id as shown in a kb_search ref, e.g. the "
                        "'a1b2c3d4' part of 'a1b2c3d4#p4'."
                    ),
                },
                "positions": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "description": (
                        "Neighbouring section positions you have NOT read yet "
                        "(see the tool description for which positions to pass)."
                    ),
                },
            },
            "required": ["doc_id", "positions"],
            "additionalProperties": False,
        },
        execute=kb_get_document,
    )


def kb_get_document(
    doc_id: str, positions: list[int], *, doc_id_map: dict[str, str] | None = None
) -> ToolResult:
    prefix = (doc_id or "").strip().lower()
    if not prefix:
        return ToolResult(text=BAD_DOC_ID)

    # The agent works in 1-based positions (p1..pN); storage is 0-based. Convert
    # to 0-based here, then everything below stays 0-based and only display sites
    # add +1 back. Normalize: positive ints only, dedup, sort ascending.
    wanted = sorted({
        p - 1 for p in positions
        if isinstance(p, int) and not isinstance(p, bool) and p >= 1
    })
    if not wanted:
        return ToolResult(text=NO_POSITIONS)

    # Cap, remembering what was dropped so the footer can offer a follow-up.
    cap = app_config.KB_GET_DOC_MAX_POSITIONS
    dropped = wanted[cap:]
    wanted = wanted[:cap]

    # Exact resolution from the orchestrator-injected map (prefix -> full id);
    # fuzzy Weaviate `like` is only a fallback for cold cross-turn prefixes.
    full_id = (doc_id_map or {}).get(prefix)
    if full_id is None:
        try:
            full_id = _store.resolve_doc_id(prefix)
        except _store.StoreError as exc:
            log.warning("kb_get_document.resolve_failed", error=str(exc))
            return ToolResult(text=BACKEND_DOWN)
    if full_id is None:
        return ToolResult(text=BAD_DOC_ID)

    try:
        window = _store.fetch_parents_by_position(full_id, wanted)
        total = _store.count_parents(full_id)
    except _store.StoreError as exc:
        log.warning("kb_get_document.fetch_failed", error=str(exc))
        return ToolResult(text=BACKEND_DOWN)

    if not window:
        if total == 0:
            return _mongo_fallback(full_id, prefix)
        return ToolResult(text=(
            "No sections at positions "
            + ", ".join(f"p{p + 1}" for p in wanted)
            + f". This document has sections p1–p{total}."
        ))

    return _render(full_id, window, total, wanted, dropped)


def _render(
    full_id: str, window: list[dict], total: int, wanted: list[int],
    dropped: list[int],
) -> ToolResult:
    budget = app_config.KB_GET_DOC_MAX_TOKENS * CHARS_PER_TOKEN
    per_section = max(400, budget // max(1, len(window)))
    prefix = full_id[:8]

    info = None
    try:
        info = _store.doc_info(full_id)
    except _store.StoreError:
        pass  # metadata is nice-to-have; the sections still render
    meta = info or {}
    approx_tokens = (meta.get("char_count") or 0) // CHARS_PER_TOKEN
    header = (
        f"{_store.display_filename(meta.get('filename') or window[0].get('filename'))} "
        f"(doc {prefix}, domain: "
        f"{meta.get('domain_level_2') or window[0].get('domain_level_2', '')}, "
    )
    if meta.get("page_count"):
        header += f", {meta['page_count']} pages"
    if approx_tokens:
        header += f", ~{approx_tokens} tokens"
    header += f", {total} sections)"

    parts: list[str] = [header]
    sources: list[dict] = []
    for p in window:
        body = p["text"]
        if len(body) > per_section:
            body = body[:per_section] + "\n… (section truncated)"
        ref = f"{prefix}#p{p['position'] + 1}"
        parts.append(f"[{ref}] {p.get('title', '')}\n{body}")
        sources.append({
            "chunk_id": ref,
            "doc_id": full_id,
            "position": p["position"] + 1,  # 1-based, matching the ref
            "title": p.get("title", ""),
            "domain": p.get("domain_level_2", ""),
        })

    found = {p["position"] for p in window}
    missing = [p for p in wanted if p not in found]
    footer = f"Sections present: p1–p{total}."
    if missing:
        footer += "\nNo section at: " + ", ".join(f"p{p + 1}" for p in missing) + "."
    if dropped:
        footer += (
            f"\nShowing first {len(window)} of {len(wanted) + len(dropped)} "
            "requested positions; call again for: "
            + ", ".join(f"p{p + 1}" for p in dropped) + "."
        )
    parts.append(footer)
    return ToolResult(text="\n\n".join(parts), sources=sources)


def _mongo_fallback(full_id: str, prefix: str) -> ToolResult:
    content = _store.get_full_text(full_id)
    if not content:
        return ToolResult(text=BAD_DOC_ID)
    cap = app_config.KB_GET_DOC_MAX_TOKENS * CHARS_PER_TOKEN
    body = content[:cap]
    note = (
        "\n\n(Archive copy: section navigation is unavailable for this document; "
        "this is the beginning of its full text"
        + (", truncated." if len(content) > cap else ".")
        + ")"
    )
    return ToolResult(text=f"[{prefix}] document text (archive copy)\n{body}{note}")
