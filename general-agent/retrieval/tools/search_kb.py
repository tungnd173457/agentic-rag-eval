"""kb_search — the agent's knowledge-base search tool.

Internals (invisible to the agent): embed the query with the SAME provider
that indexed (embedding) → Weaviate hybrid (BM25+vector) over child
chunks → group children by parent (in-call span dedupe) → rerank the parent
texts (provider-selected: Cohere or OpenRouter) → top-k → render the full parent
text as lightly structured plain
text. The only result cap is ``KB_SEARCH_TOP_K`` (post-rerank); there is no
per-section snippet truncation and no token budget. No raw scores (rank order
suffices), stable citeable chunk refs ``doc_id[:8]#p<position>``, guidance text
on empty results — never an error.
"""
from __future__ import annotations

import structlog

from config import app_config
from retrieval.tools import _store
from retrieval.tools._rerank import rerank
from retrieval.tools.base import ToolResult, ToolSpec
from services.embedding import EmbeddingError, embed_texts
from utils.taxonomy import load_domain_config

log = structlog.get_logger(__name__)

BACKEND_DOWN = (
    "Search backend unavailable right now — do not retry this search. "
    "Answer from any information you already have and tell the user the "
    "knowledge base could not be reached."
)

DESCRIPTION = """Search Laoscitec's internal knowledge base of import-export business \
documents (contracts, invoices, purchase orders, project proposals, customs and \
logistics records, company policies — mostly Vietnamese, some English). It does NOT \
contain general world knowledge, news, or law texts beyond company documents.

Matching is hybrid: short concept phrases work best (e.g. "điều khoản thanh toán hợp \
đồng" rather than a full sentence); exact codes/IDs like "PO-2024-001" also match via \
keyword search. Prefer several small targeted searches over one broad one.

Results are ranked sections with a [chunk_id] in brackets — cite these ids in your \
answer. To read a full section and its neighbours, call kb_get_document with the \
chunk_id. Use the domains filter ONLY when the question clearly maps to specific \
document types; leave it empty to search everything."""


def _domain_enum() -> list[str]:
    # general_text is the standalone classification fallback (taxonomy.taxonomy
    # FALLBACK_LEVEL2) — chunks DO carry it, but it is not in any level2_list.
    codes = {
        l2["level2"]
        for dom in load_domain_config()["domains"]
        for l2 in dom["level2_list"]
    }
    codes.add("general_text")
    return sorted(codes)


def _domain_gloss() -> str:
    pairs = sorted(
        (l2["level2"], l2.get("level2_vi", ""))
        for dom in load_domain_config()["domains"]
        for l2 in dom["level2_list"]
    )
    glosses = "; ".join(f"{code} ({vi})" for code, vi in pairs)
    return glosses + "; general_text (unclassified/general documents)"


def spec() -> ToolSpec:
    return ToolSpec(
        name="kb_search",
        description=DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search phrase. Short concept phrases in the document's "
                        "language work best, e.g. 'phạt vi phạm hợp đồng ABC' or "
                        "'INV-2024-117 payment status'."
                    ),
                },
                "domains": {
                    "type": "array",
                    "items": {"type": "string", "enum": _domain_enum()},
                    "description": (
                        "Optional document-type filter (empty = all). Values: "
                        + _domain_gloss()
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        execute=kb_search,
    )


def _chunk_ref(parent: dict) -> str:
    return f"{parent['doc_id'][:8]}#p{parent['position']}"


def kb_search(
    query: str,
    domains: list[str] | None = None,
    *,
    exclude_chunk_ids: set[str] | None = None,
) -> ToolResult:
    # exclude_chunk_ids is orchestrator-injected (NOT a model arg, not in the
    # input_schema): sections already returned earlier in this agent loop, so the
    # next search surfaces only NEW sections regardless of how the query is phrased.
    try:
        vector = embed_texts([query])[0]
    except EmbeddingError as exc:
        log.warning("kb_search.embed_failed", error=str(exc))
        return ToolResult(text=BACKEND_DOWN)

    try:
        children = _store.hybrid_search_children(
            query,
            vector,
            alpha=app_config.KB_SEARCH_HYBRID_ALPHA,
            limit=app_config.KB_SEARCH_WIDE_K,
            domains=domains or None,
        )
    except _store.StoreError as exc:
        log.warning("kb_search.weaviate_failed", error=str(exc))
        return ToolResult(text=BACKEND_DOWN)

    if not children:
        suffix = (
            " or (3) removing the domain filter"
            if domains
            else ""
        )
        return ToolResult(text=(
            f'No results for "{query}"'
            + (f" in domains {domains}" if domains else "")
            + ". Try: (1) different or shorter phrasing, (2) the document's "
            "language (Vietnamese for most documents)" + suffix + "."
        ))

    # Group children by parent, preserving best-hit order (in-call dedupe).
    refs: list[tuple[str, str]] = []
    for child in children:
        ref = (child["doc_id"], child["parent_id"])
        if ref not in refs:
            refs.append(ref)

    try:
        fetched = _store.fetch_parents(refs)
    except _store.StoreError as exc:
        log.warning("kb_search.parent_fetch_failed", error=str(exc))
        return ToolResult(text=BACKEND_DOWN)

    by_ref = {(p["doc_id"], p["chunk_id"]): p for p in fetched}
    parents = [by_ref[r] for r in refs if r in by_ref]
    if not parents:
        return ToolResult(text=(
            f'Matching sections for "{query}" could not be loaded. '
            "Try a different phrasing."
        ))

    if exclude_chunk_ids:
        parents = [p for p in parents if _chunk_ref(p) not in exclude_chunk_ids]
        if not parents:
            return ToolResult(text=(
                f'Every section matching "{query}" was already returned by an '
                "earlier search in this conversation — no new information. Try a "
                "different query, or answer the user with what you already have."
            ))

    order = rerank(query, [p["text"] for p in parents], top_n=app_config.KB_SEARCH_TOP_K)
    top = [parents[i] for i in order]

    # Best-effort: how many sections each hit's document has, so the agent knows
    # the valid position range for kb_get_document (p0..N-1) without a probe call.
    # A failure here just drops the hint — it never sinks the search.
    try:
        counts = _store.count_parents_by_doc([p["doc_id"] for p in top])
    except _store.StoreError as exc:
        log.warning("kb_search.section_count_failed", error=str(exc))
        counts = {}

    return _render(query, domains, top, total=len(parents), counts=counts)


def _render(
    query: str,
    domains: list[str] | None,
    parents: list[dict],
    *,
    total: int,
    counts: dict[str, int],
) -> ToolResult:
    suffix = f" (domains: {', '.join(domains)})" if domains else ""

    lines: list[str] = []
    sources: list[dict] = []
    shown = 0
    for i, p in enumerate(parents, 1):
        text = " ".join(p["text"].split())
        date = (p.get("updated_at") or "")[:10]
        fields = [f'Chunk retrieved in file: {p.get("filename") or "?"}']
        fields.append(f'Domain: {p.get("domain_level_2", "")}')
        fields.append(f"Updated: {date}")
        section_count = counts.get(p["doc_id"])
        if section_count is not None:
            fields.append(f"Sections in document: {section_count}")
        entry = f"{i}. [{_chunk_ref(p)}] " + ", ".join(fields) + f"\n   {text}"
        lines.append(entry)
        shown += 1
        sources.append({
            "chunk_id": _chunk_ref(p),
            "doc_id": p["doc_id"],
            "position": p["position"],
            # Same fallback as the rendered entry so source chips never show
            # blank where the text showed a filename.
            "title": p.get("title") or p.get("filename") or "",
            "domain": p.get("domain_level_2", ""),
        })

    header = f'Results 1-{shown} of {total} for "{query}"{suffix}\n'
    footer = ""
    hidden = total - shown
    if hidden > 0:
        footer += f"\n{hidden} more matches not shown. Narrow the query or add a domain filter."
    footer += (
        "\nCite sources using the [chunk_id] shown in brackets. "
        "Use kb_get_document with the doc_id and positions (e.g. [3,4,5]) to read a "
        "section and its neighbours. 'Sections in document: N' is that document's "
        "valid position range — p0 to pN-1; do not request positions outside it."
    )
    return ToolResult(text=header + "\n".join(lines) + footer, sources=sources)
