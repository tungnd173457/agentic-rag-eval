"""Parent/child split kiểu Dify (recursive-character, 2 tầng).

  - parent: cắt theo PARENT_MODE ("paragraph" → split_text; "full-doc" → 1 parent).
  - child : mỗi parent split_text tiếp với CHILD_* config.

Children embed/search; parents là ngữ cảnh trả cho LLM. Sizes đo bằng KÝ TỰ,
overlap = 0. Contract (Parent/Child fields, id format) giữ nguyên cho downstream.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from config import app_config

from .recursive_char import clean_text, split_text


def _validate_thresholds() -> None:
    """RELATIONAL guard (+ positivity, + PARENT_MODE hợp lệ). Không có bound tuyệt đối."""
    for name, value in (
        ("PARENT_MAX_CHARS", app_config.PARENT_MAX_CHARS),
        ("CHILD_MAX_CHARS", app_config.CHILD_MAX_CHARS),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if app_config.CHILD_MAX_CHARS > app_config.PARENT_MAX_CHARS:
        raise ValueError(
            f"CHILD_MAX_CHARS ({app_config.CHILD_MAX_CHARS}) must not exceed "
            f"PARENT_MAX_CHARS ({app_config.PARENT_MAX_CHARS}) — a child cannot outgrow its parent."
        )
    if app_config.PARENT_MODE not in ("paragraph", "full-doc"):
        raise ValueError(
            f'PARENT_MODE must be "paragraph" or "full-doc", got {app_config.PARENT_MODE!r}.'
        )


_validate_thresholds()


def char_len(text: str) -> int:
    """Chunk size theo ký tự — đơn vị sizing duy nhất."""
    return len(text)


def text_hash(text: str) -> str:
    """sha256 của body chunk (per-chunk ``doc_hash``)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Parent:
    parent_id: str = ""
    position: int = 0
    text: str = field(default="")
    char_count: int = 0
    doc_hash: str = ""
    child_ids: list[str] = field(default_factory=list)
    title: str = ""


@dataclass
class Child:
    parent_id: str
    position: int  # index trong parent
    text: str
    char_count: int
    child_id: str = ""
    doc_hash: str = ""


def split_document(markdown: str) -> tuple[list[Parent], list[Child]]:
    """Clean → tạo parent theo PARENT_MODE → cắt child mỗi parent. Trả
    (parents, children) với id, position, char_count và per-chunk ``doc_hash``."""
    text = clean_text(markdown)
    separators = list(app_config.RECURSIVE_SEPARATORS)

    if app_config.PARENT_MODE == "full-doc":
        parent_texts = [text] if text.strip() else []
    else:
        parent_texts = split_text(
            text, app_config.PARENT_MAX_CHARS, app_config.PARENT_SEPARATOR, separators
        )

    parents: list[Parent] = []
    children: list[Child] = []
    child_seq = 0
    for parent_text in parent_texts:
        if not parent_text.strip():
            continue
        position = len(parents)
        parent_id = f"parent_{position + 1:010d}"
        parent = Parent(
            parent_id=parent_id,
            position=position,
            text=parent_text,
            char_count=char_len(parent_text),
            doc_hash=text_hash(parent_text),
            child_ids=[],
            title="",
        )
        for child_text in split_text(
            parent_text, app_config.CHILD_MAX_CHARS, app_config.CHILD_SEPARATOR, separators
        ):
            if not child_text.strip():
                continue
            child_seq += 1
            child = Child(
                parent_id=parent_id,
                position=len(parent.child_ids),
                text=child_text,
                char_count=char_len(child_text),
                child_id=f"child_{child_seq:010d}",
                doc_hash=text_hash(child_text),
            )
            parent.child_ids.append(child.child_id)
            children.append(child)
        parents.append(parent)

    return parents, children
