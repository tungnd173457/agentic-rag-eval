# Dify-style Parent-Child Splitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thay hẳn splitter parent-child 4-stage (heading/table-aware) trong `general-agent/` bằng splitter recursive-character 2 tầng kiểu Dify, giữ nguyên contract `split_document`.

**Architecture:** Một module thuần `recursive_char.py` (split theo `fixed_separator`, mảnh quá dài đệ quy theo danh sách separator ưu tiên xuống tới ký tự; overlap = 0) + một bước clean port từ `CleanProcessor.clean` của Dify. `split_document` dựng parent (theo `PARENT_MODE`) rồi child từ mỗi parent, gán id/position/doc_hash giữ đúng các field downstream đang dùng. Embedding / Weaviate / retrieval KHÔNG đổi.

**Tech Stack:** Python 3, pydantic-settings (`config.app_config`), pytest (mới thêm cho dev). Tất cả sizing đo bằng KÝ TỰ.

## Global Constraints

- Mọi kích thước đo bằng **ký tự** (`len(text)`), không phải token.
- **Overlap cố định = 0** — không có tham số overlap nào.
- Config field đặt tên **trung lập, KHÔNG chứa chữ "DIFY"**.
- Contract bất biến: `split_document(markdown: str) -> tuple[list[Parent], list[Child]]`.
  - `Parent` phải có: `parent_id`, `position`, `text`, `char_count`, `doc_hash`, `child_ids`, `title`.
  - `Child` phải có: `child_id`, `parent_id`, `position`, `text`, `char_count`, `doc_hash`.
  - `title` luôn `""` (Dify không sinh title từ splitter).
- Id format giữ nguyên: `parent_{n:010d}` (n bắt đầu 1), `child_{n:010d}` (n tuần tự toàn document, bắt đầu 1).
- KHÔNG đụng: `services/embedding.py`, `services/weaviate.py`, `retrieval/*`, `eval/ingest_gold_docs.py` (logic), schema Weaviate.
- Validation duy nhất giữ lại: `CHILD_MAX_CHARS ≤ PARENT_MAX_CHARS` (+ positivity, + `PARENT_MODE` hợp lệ). KHÔNG check khoảng `[50, 4000]`.
- Lệnh test (chạy từ `general-agent/`): `.venv/bin/python -m pytest tests/ -v`.
- Tất cả lệnh git chạy trong repo `EnterpriseRAG-Bench` (gốc `/home/boltbolt/Desktop/EnterpriseRAG-Bench`). `general-agent/` nằm trong repo này.

---

## File Structure

- Create `general-agent/ingestion/splitters/recursive_char.py` — splitter thuần + `clean_text`. Không phụ thuộc config/embedding/weaviate.
- Modify `general-agent/config/ingestion_config.py` — thay `SplitConfig` bằng bộ field mới.
- Rewrite `general-agent/ingestion/splitters/parent_child.py` — `Parent`/`Child` rút gọn + `_validate_thresholds` đơn giản + `split_document` dùng `recursive_char`.
- Delete `general-agent/ingestion/splitters/blocks.py` (chỉ `parent_child.py` dùng).
- Delete `general-agent/ingestion/splitters/clean.py` (chỉ `parent_child.py` dùng; thay bằng `clean_text` trong `recursive_char.py`).
- Create `general-agent/tests/conftest.py`, `general-agent/tests/test_recursive_char.py`, `general-agent/tests/test_split_document.py`.
- Modify `general-agent/requirements.txt` — thêm `pytest` (dev).
- Modify `EnterpriseRAG-Bench/docs/dify-parent-child-chunking.md` — thêm mục so sánh thuật toán cũ↔mới.

`general-agent/ingestion/splitters/__init__.py` **không đổi** (vẫn `from .parent_child import Child, Parent, split_document`).

---

## Task 1: Core recursive-character splitter + clean (pure module)

**Files:**
- Create: `general-agent/ingestion/splitters/recursive_char.py`
- Create: `general-agent/tests/conftest.py`
- Create: `general-agent/tests/test_recursive_char.py`
- Modify: `general-agent/requirements.txt`

**Interfaces:**
- Produces:
  - `split_text(text: str, chunk_size: int, fixed_separator: str, separators: list[str]) -> list[str]`
  - `clean_text(text: str) -> str`

- [ ] **Step 1: Thêm pytest vào requirements và cài**

Sửa `general-agent/requirements.txt` — thêm dòng cuối:

```
pytest>=8.0
```

Rồi cài vào venv:

Run: `cd /home/boltbolt/Desktop/EnterpriseRAG-Bench/general-agent && .venv/bin/pip install pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 2: Tạo conftest đưa project dir vào sys.path**

Create `general-agent/tests/conftest.py`:

```python
"""Cho pytest import được config/ingestion/... của bản copy general-agent."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # general-agent/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 3: Viết test thất bại cho `recursive_char`**

Create `general-agent/tests/test_recursive_char.py`:

```python
from ingestion.splitters.recursive_char import clean_text, split_text

SEPS = ["\n\n", "。", ". ", " ", ""]


def test_short_text_single_chunk():
    assert split_text("hello world", 100, "\n\n", SEPS) == ["hello world"]


def test_fixed_separator_splits_paragraphs():
    assert split_text("a\n\nb\n\nc", 100, "\n\n", SEPS) == ["a", "b", "c"]


def test_oversized_piece_split_by_space_into_equal_chunks():
    text = " ".join(["aaaa"] * 10)  # 49 chars, no newline
    out = split_text(text, 20, "", SEPS)
    assert out == ["a" * 20, "a" * 20]


def test_char_level_fallback_when_no_separator():
    out = split_text("a" * 50, 20, "", [""])
    assert out == ["a" * 20, "a" * 20, "a" * 10]


def test_clean_collapses_and_strips_markers():
    assert clean_text("<|x|>") == "<x>"
    assert clean_text("a\n\n\n\nb") == "a\n\nb"
    assert clean_text("a    b") == "a b"
    assert clean_text("a\x00b") == "ab"
```

- [ ] **Step 4: Chạy test để xác nhận FAIL**

Run: `cd /home/boltbolt/Desktop/EnterpriseRAG-Bench/general-agent && .venv/bin/python -m pytest tests/test_recursive_char.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.splitters.recursive_char'`

- [ ] **Step 5: Viết `recursive_char.py`**

Create `general-agent/ingestion/splitters/recursive_char.py`:

```python
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
    r"[\t\f\r\x20  ᠎ -   　]{2,}"
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
    elif sep == " ":
        splits = text.split(" ")
    else:
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
```

- [ ] **Step 6: Chạy test để xác nhận PASS**

Run: `cd /home/boltbolt/Desktop/EnterpriseRAG-Bench/general-agent && .venv/bin/python -m pytest tests/test_recursive_char.py -v`
Expected: PASS — 5 passed

- [ ] **Step 7: Commit**

```bash
cd /home/boltbolt/Desktop/EnterpriseRAG-Bench
git add general-agent/ingestion/splitters/recursive_char.py general-agent/tests/conftest.py general-agent/tests/test_recursive_char.py general-agent/requirements.txt
git commit -m "feat(splitter): recursive-char split_text + clean_text (Dify-style core)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: New SplitConfig + rewrite split_document + delete dead files

**Files:**
- Modify: `general-agent/config/ingestion_config.py:31-46` (class `SplitConfig`) và docstring tham chiếu `PC_PARENT_MAX_CHARS` ở dòng 4.
- Rewrite: `general-agent/ingestion/splitters/parent_child.py`
- Delete: `general-agent/ingestion/splitters/blocks.py`
- Delete: `general-agent/ingestion/splitters/clean.py`
- Create: `general-agent/tests/test_split_document.py`

**Interfaces:**
- Consumes: `split_text`, `clean_text` từ Task 1; `config.app_config` với các field mới của `SplitConfig`.
- Produces:
  - `Parent` dataclass: `parent_id: str`, `position: int`, `text: str`, `char_count: int`, `doc_hash: str`, `child_ids: list[str]`, `title: str`.
  - `Child` dataclass: `child_id: str`, `parent_id: str`, `position: int`, `text: str`, `char_count: int`, `doc_hash: str`.
  - `split_document(markdown: str) -> tuple[list[Parent], list[Child]]`.
  - `_validate_thresholds() -> None`.
  - app_config fields: `PARENT_MODE: str`, `PARENT_MAX_CHARS: int`, `PARENT_SEPARATOR: str`, `CHILD_MAX_CHARS: int`, `CHILD_SEPARATOR: str`, `RECURSIVE_SEPARATORS: list[str]`.

- [ ] **Step 1: Thay class `SplitConfig` bằng bộ field mới**

Trong `general-agent/config/ingestion_config.py`, thay toàn bộ class `SplitConfig` (dòng 31-45) bằng:

```python
class SplitConfig(BaseSettings):
    """3B chunking kiểu Dify (recursive-character, CHARACTER-based).

    Quan hệ chéo (``CHILD_MAX_CHARS ≤ PARENT_MAX_CHARS``) được kiểm ở use-time
    trong ``ingestion/splitters/parent_child.py:_validate_thresholds``.
    """

    PARENT_MODE: str = Field(default="paragraph", description='Cách tạo parent: "paragraph" (cắt nhiều parent) hoặc "full-doc" (cả tài liệu = 1 parent).')
    PARENT_MAX_CHARS: PositiveInt = Field(default=1024, description="chunk_size khi cắt parent (paragraph mode).")
    PARENT_SEPARATOR: str = Field(default="\n\n", description="fixed_separator tầng parent.")
    CHILD_MAX_CHARS: PositiveInt = Field(default=512, description="chunk_size khi cắt child.")
    CHILD_SEPARATOR: str = Field(default="\n", description="fixed_separator tầng child.")
    RECURSIVE_SEPARATORS: list[str] = Field(default_factory=lambda: ["\n\n", "。", ". ", " ", ""], description="Danh sách separator đệ quy dùng chung cho cả 2 tầng.")
```

Đồng thời sửa docstring đầu file (dòng 4) — đổi ví dụ `app_config.PC_PARENT_MAX_CHARS` thành `app_config.PARENT_MAX_CHARS`.

- [ ] **Step 2: Viết test thất bại cho `split_document`**

Create `general-agent/tests/test_split_document.py`:

```python
import pytest

import config
from ingestion.splitters import split_document
from ingestion.splitters.parent_child import _validate_thresholds

SEPS = ["\n\n", "。", ". ", " ", ""]


def _set(monkeypatch, **kw):
    for k, v in kw.items():
        monkeypatch.setattr(config.app_config, k, v)


def _paragraph(monkeypatch):
    _set(
        monkeypatch,
        PARENT_MODE="paragraph",
        PARENT_MAX_CHARS=50,
        CHILD_MAX_CHARS=20,
        PARENT_SEPARATOR="\n\n",
        CHILD_SEPARATOR="\n",
        RECURSIVE_SEPARATORS=SEPS,
    )


def test_paragraph_mode_multiple_parents(monkeypatch):
    _paragraph(monkeypatch)
    parents, _ = split_document("alpha\n\nbeta\n\ngamma")
    assert [p.text for p in parents] == ["alpha", "beta", "gamma"]
    assert [p.parent_id for p in parents] == [
        "parent_0000000001",
        "parent_0000000002",
        "parent_0000000003",
    ]
    assert [p.position for p in parents] == [0, 1, 2]
    assert all(p.title == "" for p in parents)
    assert all(p.char_count == len(p.text) for p in parents)


def test_full_doc_mode_single_parent(monkeypatch):
    _set(
        monkeypatch,
        PARENT_MODE="full-doc",
        PARENT_MAX_CHARS=1024,
        CHILD_MAX_CHARS=20,
        CHILD_SEPARATOR="\n",
        RECURSIVE_SEPARATORS=SEPS,
    )
    parents, _ = split_document("alpha\n\nbeta\n\ngamma")
    assert len(parents) == 1
    assert parents[0].text == "alpha\n\nbeta\n\ngamma"


def test_child_parent_linkage_and_ids(monkeypatch):
    _paragraph(monkeypatch)
    parents, children = split_document("alpha\n\nbeta")
    pids = {p.parent_id for p in parents}
    assert all(c.parent_id in pids for c in children)
    for p in parents:
        owned = [c.child_id for c in children if c.parent_id == p.parent_id]
        assert p.child_ids == owned
    assert [c.child_id for c in children] == [
        f"child_{i:010d}" for i in range(1, len(children) + 1)
    ]


def test_leading_separator_stripped(monkeypatch):
    _set(
        monkeypatch,
        PARENT_MODE="full-doc",
        PARENT_MAX_CHARS=1024,
        CHILD_MAX_CHARS=1024,
        CHILD_SEPARATOR="\n",
        RECURSIVE_SEPARATORS=SEPS,
    )
    parents, _ = split_document(". leading dot")
    assert parents[0].text == "leading dot"


def test_validate_rejects_child_larger_than_parent(monkeypatch):
    _set(monkeypatch, CHILD_MAX_CHARS=2000, PARENT_MAX_CHARS=1000)
    with pytest.raises(ValueError):
        _validate_thresholds()
```

- [ ] **Step 3: Chạy test để xác nhận FAIL**

Run: `cd /home/boltbolt/Desktop/EnterpriseRAG-Bench/general-agent && .venv/bin/python -m pytest tests/test_split_document.py -v`
Expected: FAIL — `AttributeError`/`ValueError` do `parent_child.py` còn dùng `PC_*` cũ (validate fail ở import) hoặc field mới chưa được đọc.

- [ ] **Step 4: Viết lại `parent_child.py`**

Thay TOÀN BỘ nội dung `general-agent/ingestion/splitters/parent_child.py` bằng:

```python
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


def _strip_leading_sep(text: str) -> str:
    """Bỏ dấu '.'/'。' lạc ở đầu chunk (như Dify), giữ nguyên còn lại."""
    if text.startswith(".") or text.startswith("。"):
        return text[1:].strip()
    return text


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
    for raw_parent in parent_texts:
        parent_text = _strip_leading_sep(raw_parent)
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
        for raw_child in split_text(
            parent_text, app_config.CHILD_MAX_CHARS, app_config.CHILD_SEPARATOR, separators
        ):
            child_text = _strip_leading_sep(raw_child)
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
```

- [ ] **Step 5: Xoá hai file dead code**

Run:
```bash
cd /home/boltbolt/Desktop/EnterpriseRAG-Bench/general-agent && rm ingestion/splitters/blocks.py ingestion/splitters/clean.py
```
Expected: không lỗi (file biến mất).

- [ ] **Step 6: Chạy toàn bộ test để xác nhận PASS**

Run: `cd /home/boltbolt/Desktop/EnterpriseRAG-Bench/general-agent && .venv/bin/python -m pytest tests/ -v`
Expected: PASS — tất cả test ở `test_recursive_char.py` + `test_split_document.py` (10 passed).

- [ ] **Step 7: Smoke check import downstream còn resolve**

Run:
```bash
cd /home/boltbolt/Desktop/EnterpriseRAG-Bench/general-agent && .venv/bin/python -c "import sys; sys.path.insert(0,'.'); from ingestion.splitters import split_document, Parent, Child; p,c = split_document('A\n\nB'); print(len(p), len(c), p[0].parent_id, p[0].title=='')"
```
Expected: in ra ví dụ `2 2 parent_0000000001 True` (không ImportError; `eval/ingest_gold_docs.py` import `split_document` vẫn hợp lệ).

- [ ] **Step 8: Commit**

```bash
cd /home/boltbolt/Desktop/EnterpriseRAG-Bench
git add general-agent/config/ingestion_config.py general-agent/ingestion/splitters/parent_child.py general-agent/tests/test_split_document.py
git add -A general-agent/ingestion/splitters/blocks.py general-agent/ingestion/splitters/clean.py
git commit -m "feat(splitter): thay 4-stage splitter bằng recursive-char parent-child kiểu Dify

- SplitConfig mới (PARENT_MODE/PARENT_MAX_CHARS/CHILD_MAX_CHARS/separators), bỏ PC_*_MIN, overlap, LEVEL3_*
- split_document dùng recursive_char, giữ nguyên contract Parent/Child + id format
- xoá blocks.py, clean.py (dead code sau khi thay)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Tài liệu so sánh thuật toán cũ ↔ mới

**Files:**
- Modify: `EnterpriseRAG-Bench/docs/dify-parent-child-chunking.md` (thêm mục mới ở cuối)

**Interfaces:**
- Consumes: hành vi splitter mới (Task 1, 2) + mô tả splitter 4-stage cũ.
- Produces: không có code; chỉ tài liệu.

- [ ] **Step 1: Thêm mục so sánh vào cuối doc**

Mở `EnterpriseRAG-Bench/docs/dify-parent-child-chunking.md`, thêm vào cuối file:

````markdown
## 9. So sánh: splitter cũ (4-stage) ↔ splitter mới (recursive-char kiểu Dify)

> Áp dụng cho `general-agent/`. Kiến trúc parent-child cốt lõi không đổi (chỉ child
> được embed, parent là ngữ cảnh, đo bằng ký tự). Chỉ **thuật toán cắt** thay đổi.

| Khía cạnh | Cũ (4-stage, `blocks.py`+`parent_child.py`) | Mới (recursive-char, `recursive_char.py`) |
|---|---|---|
| Parse cấu trúc | Có — Block heading/table/para | Không — text thuần |
| Ranh giới parent | Cắt theo H1/H2; H3+ gộp vào | Cắt theo `PARENT_SEPARATOR` + size; không biết heading |
| Bảng (table) | Atomic, cắt theo ROW, lặp header | Coi như text thường |
| Chống chunk nhỏ | min-size merge + floor-fold + rescue + consolidate (tới fixpoint) | Không có MIN; có thể sinh nhiều parent nhỏ |
| Cân kích thước | balanced packing (tránh runt) | greedy merge tới sát `chunk_size` |
| Cấu hình | `PC_PARENT_MIN/MAX`, `PC_CHILD_MIN/MAX`, `PC_CHILD_OVERLAP`, `LEVEL3_*` | `PARENT_MODE`, `PARENT_MAX_CHARS`, `CHILD_MAX_CHARS`, `PARENT/CHILD_SEPARATOR`, `RECURSIVE_SEPARATORS` |
| Overlap | có (`PC_CHILD_OVERLAP`, mặc định 40) | không (cố định 0) |
| parent_mode | không (luôn theo heading) | có (`paragraph` / `full-doc`) |
| title | suy ra từ heading | luôn `""` |

**Map config cũ → mới:**

| Cũ | Mới | Ghi chú |
|---|---|---|
| `PC_PARENT_MAX_CHARS=10000` | `PARENT_MAX_CHARS=1024` | đổi default, vẫn là trần parent |
| `PC_PARENT_MIN_CHARS=2000` | (bỏ) | không còn khái niệm min |
| `PC_CHILD_MAX_CHARS=512` | `CHILD_MAX_CHARS=512` | giữ |
| `PC_CHILD_MIN_CHARS=300` | (bỏ) | — |
| `PC_CHILD_OVERLAP=40` | (bỏ) | overlap = 0 |
| `LEVEL3_*` | (bỏ) | không thuộc splitter Dify |
| — | `PARENT_MODE`, `PARENT_SEPARATOR`, `CHILD_SEPARATOR`, `RECURSIVE_SEPARATORS` | mới |

**Ảnh hưởng kỳ vọng:** splitter mới đơn giản, nhanh, không phụ thuộc cấu trúc
markdown; đổi lại mất khả năng giữ bảng nguyên vẹn và có thể tạo nhiều parent nhỏ
(không gộp ở tầng trên). Trên benchmark, recall/precision có thể thay đổi tuỳ
phân bố tài liệu — chạy `eval/local_metrics.py` + LLM judge để đo (xem CLAUDE.md).
````

- [ ] **Step 2: Commit**

```bash
cd /home/boltbolt/Desktop/EnterpriseRAG-Bench
git add docs/dify-parent-child-chunking.md
git commit -m "docs: so sánh splitter 4-stage cũ vs recursive-char mới

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Core recursive splitter (spec 3.1) → Task 1. ✓
- Clean port từ Dify (spec 3.2) → Task 1 (`clean_text`). ✓
- `split_document` + parent_mode + giữ contract + title="" (spec 3.3) → Task 2. ✓
- Config trung lập, bỏ min/overlap/level3, validation chỉ child≤parent (spec 3.4) → Task 2. ✓
- Xoá `blocks.py`/`clean.py` (spec 4) → Task 2 Step 5. ✓
- Testing (spec 5) → Task 1 + Task 2 tests. ✓
- Tài liệu so sánh (spec 6) → Task 3. ✓
- YAGNI: không pluggable, không đụng embedding/weaviate/retrieval (spec 7). ✓

**Placeholder scan:** không có TBD/TODO; mọi step có code/lệnh cụ thể. ✓

**Type consistency:** `split_text(text, chunk_size, fixed_separator, separators)` dùng nhất quán Task 1↔2. `Parent`/`Child` field khớp contract Global Constraints. `_validate_thresholds` đọc đúng các field mới của `SplitConfig`. ✓
