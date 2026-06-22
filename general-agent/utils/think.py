"""ThinkSplitter — route streamed model content into thinking vs answer,
stripping <think>…</think> tags even when a tag is split across chunks.

Model-agnostic and dependency-free: used by the retrieval chat stream to pull
reasoning-model chain-of-thought out of the answer channel. A suffix that could
start a tag is held back until the next chunk resolves it.
"""
from __future__ import annotations


class ThinkSplitter:
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._thinking = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        self._buf += text
        out: list[tuple[str, str]] = []
        while True:
            tag = self.CLOSE if self._thinking else self.OPEN
            i = self._buf.find(tag)
            if i == -1:
                emit = self._emit_safe(tag)
                if emit:
                    out.append(("thinking" if self._thinking else "text", emit))
                return out
            if i:
                out.append(("thinking" if self._thinking else "text", self._buf[:i]))
            self._buf = self._buf[i + len(tag):]
            self._thinking = not self._thinking

    def _emit_safe(self, tag: str) -> str:
        held = 0
        for k in range(1, len(tag)):
            if self._buf.endswith(tag[:k]):
                held = k
        if held:
            emit, self._buf = self._buf[:-held], self._buf[-held:]
        else:
            emit, self._buf = self._buf, ""
        return emit

    def flush(self) -> list[tuple[str, str]]:
        if not self._buf:
            return []
        emit, self._buf = self._buf, ""
        return [("thinking" if self._thinking else "text", emit)]
