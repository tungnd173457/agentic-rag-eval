"""LLM ops dùng cho RAG retrieval (bản rút gọn cho eval).

Bản gốc còn export structured/vision; ở đây chỉ giữ phần cần cho agent loop:
  * client — factory OpenAI client theo provider (openai/ollama qua /v1)
  * stream — streaming tool-calling chat cho agent loop
"""
from services.llm.client import (
    ClientConfigError,
    resolve_chat,
    resolve_chat_async,
    resolve_embed,
    resolve_vision,
)
from services.llm.stream import ChatDelta, ChatError, ToolCall, chat_with_tools

__all__ = [
    "ChatDelta",
    "ChatError",
    "ToolCall",
    "chat_with_tools",
    "ClientConfigError",
    "resolve_chat",
    "resolve_chat_async",
    "resolve_vision",
    "resolve_embed",
]
