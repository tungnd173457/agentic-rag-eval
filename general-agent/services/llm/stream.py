"""Streaming tool-calling chat for the agent loop — one OpenAI-SDK code path
for both providers (see ``services.llm.client`` for how each provider's client
is built; ollama targets its OpenAI-compatible /v1 endpoint).

chat_with_tools() is an ASYNC generator of ChatDelta (<think> routing is
delegated to ``utils.think.ThinkSplitter``):
  - "thinking": qwen <think>…</think> content (tags stripped, split-tag safe)
  - "text":     answer content
  - "tool_call": a COMPLETE tool call — args already assembled from deltas and
                 parsed (arguments=None when the model emitted invalid JSON;
                 the orchestrator answers with guidance so it self-corrects)
  - "finish":   end of the model turn, with finish_reason

Non-stream mode is simply consuming the iterator to completion. Raises
ChatError on transport/HTTP/config failure — retry policy belongs to the
orchestrator, not here. NOTE: this is a generator, so config/transport errors
surface on the FIRST next(), not at call time — keep construction and
iteration inside the same try/except ChatError. The timeout is connect +
between-chunks read time, not a total-stream budget.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

from openai import OpenAIError, omit

from config import app_config
from services.llm.client import ClientConfigError, resolve_chat_async
from utils.think import ThinkSplitter


class ChatError(RuntimeError):
    """LLM endpoint unreachable, non-2xx, or misconfigured."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None       # None ⇒ raw_arguments was not valid JSON
    raw_arguments: str


@dataclass
class ChatDelta:
    kind: str                    # "thinking" | "text" | "tool_call" | "finish"
    text: str = ""
    tool_call: ToolCall | None = None
    finish_reason: str = ""


async def chat_with_tools(
    messages: list[dict],
    tools: list[dict],
    *,
    temperature: float = 0.0,
    provider: str | None = None,
    model: str | None = None,
) -> AsyncIterator[ChatDelta]:
    cfg = app_config
    try:
        client, model = resolve_chat_async(cfg, provider=provider, model=model)
    except ClientConfigError as exc:
        raise ChatError(str(exc)) from exc

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=cast(Any, messages),
            stream=True,
            temperature=temperature,
            tools=cast(Any, tools) if tools else omit,
        )
    except OpenAIError as exc:
        raise ChatError(f"chat {model} → {exc}") from exc

    splitter = ThinkSplitter()
    pending: dict[int, dict] = {}     # index → {"id", "name", "args"}
    finish_reason = ""

    try:
        async for chunk in stream:
            choice = (chunk.choices or [None])[0]
            if choice is None:
                continue
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta
            if delta is None:
                continue

            content = delta.content
            if content:
                for kind, text in splitter.feed(content):
                    yield ChatDelta(kind=kind, text=text)

            for part in delta.tool_calls or []:
                slot = pending.setdefault(
                    part.index or 0, {"id": "", "name": "", "args": ""}
                )
                if part.id:
                    slot["id"] = part.id
                fn = part.function
                if fn and fn.name:
                    slot["name"] = fn.name
                if fn and fn.arguments:
                    slot["args"] += fn.arguments
    except OpenAIError as exc:
        raise ChatError(f"chat {model} stream → {exc}") from exc

    for kind, text in splitter.flush():
        yield ChatDelta(kind=kind, text=text)

    for index in sorted(pending):
        slot = pending[index]
        raw_args = slot["args"]
        try:
            arguments = json.loads(raw_args) if raw_args.strip() else {}
            if not isinstance(arguments, dict):
                arguments = None
        except ValueError:
            arguments = None
        yield ChatDelta(
            kind="tool_call",
            tool_call=ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments=arguments,
                raw_arguments=raw_args,
            ),
        )

    yield ChatDelta(kind="finish", finish_reason=finish_reason or "stop")
