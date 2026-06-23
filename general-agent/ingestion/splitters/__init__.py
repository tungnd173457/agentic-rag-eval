"""Parent/child splitter (recursive-character, Dify-style).

Two tiers, sizes measured in CHARACTERS, overlap = 0:
  - parent: split per ``PARENT_MODE`` ("paragraph" → recursive split_text;
    "full-doc" → whole document as one parent).
  - child : each parent re-split with the CHILD_* knobs; only children are
    embedded/searched, parents are returned as LLM context.

Public API: ``split_document(markdown) -> (list[Parent], list[Child])``,
plus the ``Parent`` and ``Child`` dataclasses.
"""
from .parent_child import Child, Parent, split_document

__all__ = ["split_document", "Parent", "Child"]
