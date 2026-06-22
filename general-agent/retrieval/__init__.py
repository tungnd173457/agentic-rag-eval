"""Retrieval (read-side) module: query → agentic answer.

Public surface: the I/O schemas. The agent loop lives in retrieval.agent,
tools in retrieval.tools, the SSE API in retrieval.api.
"""
from schemas.retrieval import (
    AgentResult,
    ChatRequest,
    ChatResponse,
    Source,
    ToolTraceEntry,
)

__all__ = [
    "AgentResult",
    "ChatRequest",
    "ChatResponse",
    "Source",
    "ToolTraceEntry",
]
