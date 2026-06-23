"""Recursive-character splitter (port sạch từ Dify FixedRecursiveCharacterTextSplitter).

Đo bằng KÝ TỰ, overlap = 0. Hai hành vi cốt lõi được giữ:
  1. Tách theo ``fixed_separator`` ở tầng trên; mảnh ≤ ``chunk_size`` giữ nguyên,
     mảnh dài hơn mới đệ quy.
  2. Đệ quy theo ``separators`` ưu tiên (separator đầu tiên xuất hiện trong text);
     separator rỗng ``""`` ⇒ cắt theo từng ký tự. Các mảnh nhỏ liền nhau được gộp
     tới sát ``chunk_size``.
"""
from __future__ import annotations

import re

# --- clean (port từ Dify CleanProcessor.clean: default-clean + remove_extra_spaces) ---
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f￾]")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(
    r"[\t\f\r\x20  ᠎ -   　]{2,}"
)


def clean_text(text: str) -> str:
    """Đổi ``<| |>`` về ``< >``, xoá control char, gộp ``\\n{3,}``→``\\n\\n`` và
    chuỗi khoảng trắng ngang ≥2 → một space."""
    text = text.replace("<|", "<").replace("|>", ">")
    text = _CONTROL.sub("", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _choose_separator(text: str, separators: list[str]) -> tuple[str, list[str]]:
    """Separator đầu tiên xuất hiện trong text (kèm các separator còn lại sau nó).
    ``""`` là sentinel cắt-theo-ký-tự và luôn khớp."""
    for i, sep in enumerate(separators):
        if sep == "":
            return "", []
        if sep in text:
            return sep, list(separators[i + 1:])
    return (separators[-1] if separators else ""), []


def _merge(splits: list[str], chunk_size: int) -> list[str]:
    """Gộp greedy các mảnh liền nhau tới sát ``chunk_size`` (overlap = 0)."""
    out: list[str] = []
    cur = ""
    for s in splits:
        if cur and len(cur) + len(s) > chunk_size:
            out.append(cur)
            cur = s
        else:
            cur += s
    if cur:
        out.append(cur)
    return out


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    sep, rest = _choose_separator(text, separators)
    if sep == "":
        splits = list(text)
    else:
        # Re-attach the separator to every piece (incl. " ") so the greedy
        # merge reconstructs the original spacing — Dify keeps separators via
        # _merge_splits(join=separator); this is the plan-consistent equivalent.
        parts = text.split(sep)
        splits = [p + sep for p in parts[:-1]] + [parts[-1]]
    splits = [s for s in splits if s not in ("", "\n")]

    out: list[str] = []
    buf: list[str] = []
    for s in splits:
        if len(s) < chunk_size:
            buf.append(s)
            continue
        if buf:
            out.extend(_merge(buf, chunk_size))
            buf = []
        if rest:
            out.extend(_recursive_split(s, chunk_size, rest))
        else:
            out.append(s)  # không chia nhỏ hơn được — giữ nguyên (như Dify)
    if buf:
        out.extend(_merge(buf, chunk_size))
    return out


def split_text(
    text: str, chunk_size: int, fixed_separator: str, separators: list[str]
) -> list[str]:
    pieces = text.split(fixed_separator) if fixed_separator else [text]
    out: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            if piece.strip():
                out.append(piece)
        else:
            out.extend(_recursive_split(piece, chunk_size, separators))
    return out
