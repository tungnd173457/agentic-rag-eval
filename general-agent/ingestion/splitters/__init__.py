"""3B parent/child structural splitter (stages 1-4).

``split_document`` turns 3A markdown into parent chunks (context) and child
chunks (embed/search), sized in characters (develop ``common/parent_child``
convention). level3 classification (stage 5) lives in
``..classifiers.level3_classifier``.
"""
from .parent_child import Child, Parent, split_document

__all__ = ["split_document", "Parent", "Child"]
