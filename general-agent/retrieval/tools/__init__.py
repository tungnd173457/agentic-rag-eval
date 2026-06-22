"""Agent tools — agent-agnostic plain Python, registered for the loop."""
from retrieval.tools import get_document, search_kb
from retrieval.tools.base import ToolRegistry, ToolResult, ToolSpec


def default_registry() -> ToolRegistry:
    """The agent's tool set: search-then-fetch, nothing else (tool-selection
    accuracy degrades with count; kb_list_documents is a documented future
    extension — see the design spec)."""
    return ToolRegistry([search_kb.spec(), get_document.spec()])


__all__ = ["ToolRegistry", "ToolResult", "ToolSpec", "default_registry"]
