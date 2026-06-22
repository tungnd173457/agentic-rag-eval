"""Text cleaning before chunking (ported from develop ``common/parent_child``
``CleanProcessor``, Dify origin).

Runs at the top of ``split_document`` so stages 1-4 never see control chars or
runaway blank runs from 3A. Two layers:

  * always-on   — strip ``<|``/``|>`` markers, control / invalid unicode chars;
  * extra-spaces — collapse 3+ newlines to a blank line and runs of horizontal
    whitespace to one space (markdown headings/tables/pipes are untouched).

Develop's optional ``remove_urls_emails`` rule is intentionally NOT applied:
document URLs / emails are real content here (contracts, contacts).
"""
from __future__ import annotations

import re

# always-on: pseudo-tag markers + control / invalid unicode (Dify rules).
_MARKER_OPEN = re.compile(r"<\|")
_MARKER_CLOSE = re.compile(r"\|>")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f￾]")

# remove_extra_spaces: 3+ newlines → blank line; 2+ horizontal spaces → one.
_EXTRA_NEWLINES = re.compile(r"\n{3,}")
_EXTRA_SPACES = re.compile(r"[\t\f\r\x20  ᠎ -   　]{2,}")


def clean_markdown(text: str) -> str:
    """Clean 3A markdown before block parsing (always-on + extra-spaces)."""
    text = _MARKER_OPEN.sub("<", text)
    text = _MARKER_CLOSE.sub(">", text)
    text = _CONTROL.sub("", text)
    text = _EXTRA_NEWLINES.sub("\n\n", text)
    text = _EXTRA_SPACES.sub(" ", text)
    return text.strip()
