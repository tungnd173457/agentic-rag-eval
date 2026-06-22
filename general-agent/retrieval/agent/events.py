"""Agent events — the orchestrator's output unit.

The answer is a sequence of ORDERED MESSAGE PARTS (text and tool activity in
the order they happened; the last text part is the answer). The API layer
serializes these as SSE; non-stream mode folds them into a ChatResponse.

Types and payloads (see design spec §API):
  message_start  {request_id}
  thinking_delta {step, text}
  text_delta     {step, text}
  tool_call      {step, call_id, name, display}
  tool_result    {call_id, summary: {result_count, sources}}
  error          {message, recoverable}
  done           full response dump (AgentResult internally; ChatResponse to the client)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentEvent:
    type: str
    data: dict
