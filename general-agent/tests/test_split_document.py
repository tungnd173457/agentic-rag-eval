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


def test_leading_punctuation_preserved(monkeypatch):
    # Dify-faithful: the splitter does NOT strip leading "."/"。".
    _set(
        monkeypatch,
        PARENT_MODE="full-doc",
        PARENT_MAX_CHARS=1024,
        CHILD_MAX_CHARS=1024,
        CHILD_SEPARATOR="\n",
        RECURSIVE_SEPARATORS=SEPS,
    )
    parents, _ = split_document(". leading dot")
    assert parents[0].text == ". leading dot"


def test_validate_rejects_child_larger_than_parent(monkeypatch):
    _set(monkeypatch, CHILD_MAX_CHARS=2000, PARENT_MAX_CHARS=1000)
    with pytest.raises(ValueError):
        _validate_thresholds()
