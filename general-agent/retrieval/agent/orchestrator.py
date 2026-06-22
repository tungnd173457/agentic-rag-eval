"""The agent loop: an async generator of AgentEvent driving chat_with_tools.

Layered stopping (design spec §Agent loop):
  1. hard cap app_config.MAX_TOOL_ROUNDS — then a FORCED no-tools answer turn,
  2. repeat-query + already-seen-chunk annotations steer the model to stop,
  3. budget exhaustion NEVER errors — answer-with-gaps.

LLM failure: one retry (recoverable error event), then a terminal error event.
Tool errors never surface here — the registry returns guidance text.

``chat`` and ``registry`` are injectable; under a scripted fake chat the loop
is a deterministic state machine (see tests).
"""
from __future__ import annotations

import re
import time
import uuid
from collections.abc import AsyncIterator, Callable

import anyio
import anyio.to_thread
import structlog

from config import app_config
from retrieval.agent.events import AgentEvent
from retrieval.agent.prompts import FORCED_ANSWER_PROMPT
from retrieval.tools import ToolRegistry, ToolResult, default_registry
from schemas.retrieval import AgentResult, Source, ToolTraceEntry
from services.llm.stream import ChatError, ToolCall, chat_with_tools

log = structlog.get_logger(__name__)

CITATION_RE = re.compile(r"\[([0-9a-fA-F]{6,}#p\d+)\]")
_MAX_DISPLAY_CHARS = 80


def _display_args(call: ToolCall) -> dict:
    """Condensed args for the tool_call event chip (strings truncated)."""
    if call.arguments is None:
        return {"raw": call.raw_arguments[:_MAX_DISPLAY_CHARS]}
    out = {}
    for key, value in call.arguments.items():
        out[key] = value[:_MAX_DISPLAY_CHARS] if isinstance(value, str) else value
    return out


def _search_key(call: ToolCall) -> tuple | None:
    if call.name != "kb_search" or not call.arguments:
        return None
    domains = call.arguments.get("domains") or []
    if not isinstance(domains, list):
        domains = []
    return (
        str(call.arguments.get("query", "")).strip().lower(),
        # str() each item: schema-invalid shapes (e.g. [{}]) get a guidance
        # result from the registry, but the dedupe key must never raise.
        tuple(sorted(str(d) for d in domains)),
    )


def _annotate(
    call: ToolCall,
    result: ToolResult,
    seen_chunk_ids: set[str],
    seen_queries: set[tuple],
) -> str:
    """Append stop-steering notes; update the cross-round dedupe memory."""
    text = result.text
    key = _search_key(call)
    if key is not None:
        if key in seen_queries:
            text += "\n\nNote: identical to a previous search; results unchanged."
        seen_queries.add(key)
    ids = [s["chunk_id"] for s in result.sources]
    if ids and all(i in seen_chunk_ids for i in ids):
        text += (
            "\n\nNote: all sections above were already returned by earlier tool "
            "calls — no new information. Answer with what you have."
        )
    seen_chunk_ids.update(ids)
    return text


async def run_agent(
    messages: list[dict],
    *,
    seed_chunk_ids: set[str] | None = None,
    seed_sources: list[dict] | None = None,
    registry: ToolRegistry | None = None,
    chat: Callable | None = None,
) -> AsyncIterator[AgentEvent]:
    chat = chat or chat_with_tools
    registry = registry or default_registry()
    tools = registry.openai_tools()
    started = time.perf_counter()
    start_len = len(messages)

    yield AgentEvent("message_start", {"request_id": uuid.uuid4().hex})

    seen_chunk_ids: set[str] = set(seed_chunk_ids or ())
    seen_queries: set[tuple] = set()
    # source_index = chunks surfaced THIS run (drives dedup/previews/persistence).
    # cite_index = the citation-resolution map: prior turns' surfaced sources
    # (seed_sources) PLUS this run's, so a follow-up can cite a chunk retrieved
    # in an earlier turn without re-searching it. Kept separate so the persisted
    # surfaced set stays per-run and doesn't grow unboundedly.
    source_index: dict[str, dict] = {}
    cite_index: dict[str, dict] = {s["chunk_id"]: s for s in (seed_sources or [])}
    tool_trace: list[ToolTraceEntry] = []
    thinking_parts: list[str] = []
    final_text = ""
    rounds = 0
    round_no = 0
    retried = False

    while True:
        forced = round_no >= app_config.MAX_TOOL_ROUNDS
        if forced and (not messages or messages[-1].get("content") != FORCED_ANSWER_PROMPT):
            messages.append({"role": "user", "content": FORCED_ANSWER_PROMPT})

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        try:
            async for delta in chat(messages, [] if forced else tools, temperature=0.0):
                if delta.kind == "thinking" and delta.text:
                    thinking_parts.append(delta.text)
                    yield AgentEvent("thinking_delta", {"step": round_no, "text": delta.text})
                elif delta.kind == "text" and delta.text:
                    text_parts.append(delta.text)
                    yield AgentEvent("text_delta", {"step": round_no, "text": delta.text})
                elif delta.kind == "tool_call" and delta.tool_call:
                    calls.append(delta.tool_call)
        except ChatError as exc:
            log.warning("agent.llm_failed", error=str(exc), retried=retried)
            if not retried:
                retried = True
                yield AgentEvent(
                    "error", {"message": "LLM call failed; retrying", "recoverable": True}
                )
                continue
            # Generic message only: ChatError embeds the model name and
            # backend URL — client-visible events must not leak infra detail
            # (specifics already went to the log above).
            yield AgentEvent(
                "error", {"message": "LLM unavailable", "recoverable": False}
            )
            return

        final_text = "".join(text_parts)
        if not calls or forced:
            break

        rounds += 1
        messages.append({
            "role": "assistant",
            "content": final_text or None,
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.raw_arguments or "{}"},
                }
                for c in calls
            ],
        })

        for call in calls:
            display = _display_args(call)
            yield AgentEvent(
                "tool_call",
                {"step": round_no, "call_id": call.id, "name": call.name, "display": display},
            )
            tool_started = time.perf_counter()
            if call.arguments is None:
                result = ToolResult(text=registry.invalid_args_guidance(call.name))
            else:
                # Orchestrator-injected context, merged in AFTER validation (not
                # model args — see registry.execute):
                #  - kb_search dedupes against sections already returned this loop;
                #  - kb_get_document resolves a citation prefix to the full doc_id
                #    from sources already surfaced this run (exact, no Weaviate
                #    `like`); the tool falls back to `like` only on a cold prefix.
                if call.name == "kb_search":
                    injected = {"exclude_chunk_ids": seen_chunk_ids}
                elif call.name == "kb_get_document":
                    injected = {"doc_id_map": {
                        s["chunk_id"].split("#")[0]: s["doc_id"]
                        for s in source_index.values()
                        if s.get("doc_id")
                    }}
                else:
                    injected = None
                # Tools do blocking I/O (Weaviate hybrid, Cohere rerank, query
                # embedding) — offload so the event loop is never stalled.
                result = await anyio.to_thread.run_sync(
                    registry.execute, call.name, call.arguments, injected
                )
            content = _annotate(call, result, seen_chunk_ids, seen_queries)
            latency_ms = round((time.perf_counter() - tool_started) * 1000, 1)

            for source in result.sources:
                source_index.setdefault(source["chunk_id"], source)
                cite_index.setdefault(source["chunk_id"], source)
            tool_trace.append(ToolTraceEntry(
                tool=call.name, args=display,
                result_count=len(result.sources), latency_ms=latency_ms,
                previews=[Source(**s) for s in result.sources],
                step=round_no,
            ))
            yield AgentEvent("tool_result", {
                "call_id": call.id,
                "summary": {
                    "result_count": len(result.sources),
                    "sources": result.sources,
                },
            })
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )
        round_no += 1

    answer = final_text.strip()
    # Tools emit lowercase refs; normalize so an uppercased model citation
    # still maps to its source instead of being flagged unknown.
    cited = list(dict.fromkeys(c.lower() for c in CITATION_RE.findall(answer)))
    sources = [Source(**cite_index[c]) for c in cited if c in cite_index]
    flags: list[str] = []
    if tool_trace and not cited:
        flags.append("no_citations")
    if any(c not in cite_index for c in cited):
        flags.append("unknown_citation")

    new_messages = messages[start_len:] + [{"role": "assistant", "content": answer}]
    new_messages = [m for m in new_messages if m.get("content") != FORCED_ANSWER_PROMPT]

    response = AgentResult(
        answer=answer,
        reasoning="".join(thinking_parts),
        sources=sources,
        tool_trace=tool_trace,
        rounds=rounds,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        flags=flags,
        new_messages=new_messages,
        surfaced_chunk_ids=list(source_index),
        surfaced_sources=[Source(**s) for s in source_index.values()],
    )
    log.info(
        "agent.done",
        rounds=rounds,
        latency_ms=response.latency_ms,
        flags=flags,
        tool_trace=[t.model_dump() for t in tool_trace],
    )
    yield AgentEvent("done", response.model_dump())
