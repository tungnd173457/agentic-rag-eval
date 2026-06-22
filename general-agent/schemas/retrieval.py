"""Pydantic v2 schemas for the retrieval/chat module — single source of truth.

Persistence document model lives in models/conversation.py; these are the HTTP
view DTOs + the internal AgentResult produced by the agent loop.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Source(BaseModel):
    """One cited knowledge-base section ([chunk_id] in the answer)."""

    chunk_id: str            # display/citation token "<doc_id[:8]>#p<position>"
    doc_id: str = ""         # full content-hash id; "" only for legacy persisted sources
    position: int = -1       # parent section index; -1 only for legacy
    title: str = ""
    domain: str = ""


class ToolTraceEntry(BaseModel):
    """One tool call made by the agent (observability; logged per request).

    `previews` holds every section the call returned so the tool timeline
    can be replayed in full when a past conversation is reopened.
    """

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result_count: int = 0
    latency_ms: float = 0.0
    previews: list[Source] = Field(default_factory=list)
    step: int = 0            # agent round this call ran in (interleaved timeline replay)


class AgentResult(BaseModel):
    """Internal result of one agent run — payload of the orchestrator `done` event.

    `new_messages` are the raw OpenAI blocks generated this run (assistant
    tool_calls + tool results + final assistant), persisted verbatim by the chat
    service. `surfaced_chunk_ids` are all chunks kb_search returned (cross-turn
    dedup seed). Neither is forwarded to the HTTP client.
    """

    answer: str
    reasoning: str = ""
    sources: list[Source] = Field(default_factory=list)
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)
    rounds: int = 0
    latency_ms: float = 0.0
    flags: list[str] = Field(default_factory=list)
    new_messages: list[dict[str, Any]] = Field(default_factory=list)
    surfaced_chunk_ids: list[str] = Field(default_factory=list)
    # Full metadata of every chunk surfaced this run, persisted per turn so a
    # later turn can resolve citations back to chunks retrieved earlier in the
    # conversation (seeded into the next run's cite_index).
    surfaced_sources: list[Source] = Field(default_factory=list)


class ChatRequest(BaseModel):
    user_id: str
    message: str = Field(min_length=1)
    conversation_id: str | None = None   # None → server creates a new conversation
    stream: bool = True
    # Per-request LLM selection from the chat screen; None → env defaults.
    provider: str | None = None   # "openai" | "ollama" | "openrouter"; None → env LLM_PROVIDER
    model: str | None = None      # model id; None → env chat model for the provider


class ChatResponse(BaseModel):
    """Client-facing answer — payload of the chat `done` event / non-stream JSON."""

    conversation_id: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)
    rounds: int = 0
    latency_ms: float = 0.0
    flags: list[str] = Field(default_factory=list)


class CreateConversationRequest(BaseModel):
    user_id: str
    title: str | None = None


class RenameConversationRequest(BaseModel):
    title: str


class ConversationMeta(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class DisplayTurn(BaseModel):
    turn_id: str
    role: str                       # "user" | "assistant"
    content: str
    created_at: datetime
    # The fields below are populated for assistant turns only, so a reopened
    # conversation can replay the reasoning block + tool timeline + turn footer.
    sources: list[Source] = Field(default_factory=list)
    # Per-round reasoning (round -> text), so the client replays the interleaved
    # thinking→tools→thinking timeline. `reasoning` is kept as a flat fallback for
    # older clients; new clients prefer `thoughts`.
    thoughts: dict[int, str] = Field(default_factory=dict)
    reasoning: str = ""
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)
    rounds: int = 0
    latency_ms: float = 0.0
    flags: list[str] = Field(default_factory=list)


class ConversationDetail(ConversationMeta):
    turns: list[DisplayTurn] = Field(default_factory=list)
