"""Parse 3A markdown into a flat list of typed blocks for the splitter.

3A emits ATX ``#`` headings, GFM tables (``| ... |`` with a ``| --- |`` rule)
and plain paragraphs. We turn the raw markdown into ``Block``s so stage 1
(structural split) can walk content with table boundaries already detected and
kept intact. Figures are NOT inline tags in 3A output — captions are appended as
a trailing bullet list — so only tables are treated as atomic here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class Block:
    """One markdown block. ``md`` is the text to re-emit into a parent body."""

    kind: str            # "heading" | "table" | "para"
    level: int           # heading level (1..6); 0 for table/para
    text: str            # clean text (heading title / table / paragraph)
    md: str              # markdown to write into the parent body
    atomic: bool = False  # tables are atomic — never split across children mid-row


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return bool(s) and "|" in s


def _is_separator_row(line: str) -> bool:
    """A GFM rule row, e.g. ``| --- | :--: |`` — only dashes/colons/pipes/space."""
    s = line.strip()
    return "-" in s and set(s) <= set("-:| ")


def parse_blocks(markdown: str) -> list[Block]:
    """Split ``markdown`` into heading / table / paragraph blocks, in order."""
    lines = markdown.splitlines()
    blocks: list[Block] = []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            blocks.append(Block("heading", level, text, "#" * level + " " + text))
            i += 1
            continue

        # A run of consecutive pipe-rows is a candidate table.
        if _is_table_row(line) and i + 1 < n and _is_table_row(lines[i + 1]):
            start = i
            while i < n and _is_table_row(lines[i]):
                i += 1
            rows = [r.rstrip() for r in lines[start:i]]
            if len(rows) >= 2 and any(_is_separator_row(r) for r in rows):
                md = "\n".join(rows)
                blocks.append(Block("table", 0, md, md, atomic=True))
            else:  # pipes but no rule row → ordinary paragraph
                md = "\n".join(rows)
                blocks.append(Block("para", 0, md, md))
            continue

        # Paragraph: accumulate until a blank line, heading, or table starts.
        start = i
        while i < n and lines[i].strip() and not _HEADING_RE.match(lines[i]):
            if _is_table_row(lines[i]) and i + 1 < n and _is_table_row(lines[i + 1]):
                break
            i += 1
        md = "\n".join(r.rstrip() for r in lines[start:i])
        blocks.append(Block("para", 0, md, md))

    return blocks


def table_header(table_md: str) -> tuple[str, list[str]]:
    """Return (header+rule markdown, data-row lines) for a GFM table.

    The header + separator are repeated in every table child/cap-piece so each
    fragment stays a valid, self-describing table.
    """
    rows = [r for r in table_md.splitlines() if r.strip()]
    if len(rows) < 2:
        return "", rows
    # header is row 0; the rule row is the first separator row after it.
    rule_idx = next((j for j, r in enumerate(rows) if _is_separator_row(r)), 1)
    header = "\n".join(rows[: rule_idx + 1])
    data = rows[rule_idx + 1 :]
    return header, data
