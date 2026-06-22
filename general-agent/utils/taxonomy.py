"""Loader for the domain classification config (domains.json).

The JSON is bundled into the Lambda asset via `Code.from_asset("src")`, so the
path resolves the same way locally and inside the Lambda runtime.

domains.json is the single source of truth for the domain taxonomy — there is
no Python enum mirroring it. Taxonomy is 3-level: every chunk carries
(level1, level2, level3). A chunk that matches no domain falls back to the
standalone global `(general_text, general_text, general_text)`.

The KB / S3 routing unit is **level2** — each level2 code maps to one Bedrock
KB and one `domains/<level2>/` prefix. level3 stays in the taxonomy as a
per-chunk classification leaf only; it is no longer a KB unit (too granular).
Every level2 also carries a `metadata` field describing the per-doc-type fields
to extract.
"""

import json
from functools import lru_cache
from pathlib import Path

_TAXONOMY_FILE = Path(__file__).resolve().parents[1] / "config" / "domains.json"
_CONFIG_PATH = _TAXONOMY_FILE

# Standalone global fallback — used when classification matches nothing.
FALLBACK_LEVEL1 = "general_text"
FALLBACK_LEVEL2 = "general_text"
FALLBACK_LEVEL3 = "general_text"
GLOBAL_FALLBACK_PATH = (FALLBACK_LEVEL1, FALLBACK_LEVEL2, FALLBACK_LEVEL3)


@lru_cache(maxsize=1)
def load_domain_config() -> dict:
    """Return the parsed domains.json (read once per cold start)."""
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def level1_codes() -> frozenset[str]:
    """All valid level1 codes — used to build the S3 routing prefix."""
    config = load_domain_config()
    return frozenset(domain["level1"] for domain in config["domains"])


@lru_cache(maxsize=1)
def level2_codes() -> frozenset[str]:
    """All distinct level2 codes — each maps to one KB / S3 routing prefix."""
    config = load_domain_config()
    return frozenset(
        level2["level2"]
        for domain in config["domains"]
        for level2 in domain["level2_list"]
    )


@lru_cache(maxsize=1)
def level2_to_level1() -> dict[str, str]:
    """Map each level2 code → its (unique) level1 code.

    level2 is unique across the taxonomy and belongs to exactly one level1, so
    the file-level classifier (3A) can ask the LLM for level2 alone and derive
    level1 from this map — avoiding the mismatched-(level1, level2)-pair
    fallback that would otherwise discard a correctly-chosen level2.
    """
    config = load_domain_config()
    return {
        level2["level2"]: domain["level1"]
        for domain in config["domains"]
        for level2 in domain["level2_list"]
    }


def is_valid_level2(level2: str) -> bool:
    """True if `level2` is a known taxonomy code."""
    return level2 in level2_codes()


def level1_for_level2(level2: str) -> str | None:
    """Return the level1 that owns `level2`, or None if `level2` is unknown."""
    return level2_to_level1().get(level2)


@lru_cache(maxsize=1)
def level3_codes() -> frozenset[str]:
    """All distinct level3 codes — per-chunk taxonomy leaves (not a KB unit)."""
    config = load_domain_config()
    return frozenset(
        level3["level3"]
        for domain in config["domains"]
        for level2 in domain["level2_list"]
        for level3 in level2["level3_list"]
    )


@lru_cache(maxsize=1)
def _level2_to_level3_list() -> dict[str, list[dict]]:
    """Map each level2 code → its raw `level3_list` from domains.json."""
    config = load_domain_config()
    return {
        level2["level2"]: list(level2.get("level3_list", []))
        for domain in config["domains"]
        for level2 in domain["level2_list"]
    }


def level3_for_level2(level2: str) -> list[dict]:
    """Return the level3 entries [{level3, description}] for `level2`.

    Used by step 3B to narrow the per-chunk classifier's candidate set to the
    level3 leaves of the file's (already-fixed) level2. The trailing
    `general_text` fallback leaf is included. Empty list if `level2` is unknown.
    """
    return _level2_to_level3_list().get(level2, [])


@lru_cache(maxsize=1)
def level2_metadata() -> dict[str, dict | None]:
    """Map each level2 code → its metadata-extraction schema.

    The schema comes from the `metadata` field on each level2 in domains.json:
    a JSON object `{field: "how to find it"}` (values may be nested objects or
    arrays for line items). It is `None` for level2s with no schema, e.g. the
    general_text fallback.
    """
    config = load_domain_config()
    return {
        level2["level2"]: level2.get("metadata")
        for domain in config["domains"]
        for level2 in domain["level2_list"]
    }


@lru_cache(maxsize=1)
def valid_level1_level2() -> frozenset[tuple[str, str]]:
    """Valid (level1, level2) pairs — validate the 3A_2 file-level classifier."""
    config = load_domain_config()
    return frozenset(
        (domain["level1"], level2["level2"])
        for domain in config["domains"]
        for level2 in domain["level2_list"]
    )


@lru_cache(maxsize=1)
def valid_paths() -> frozenset[tuple[str, str, str]]:
    """Every valid (level1, level2, level3) triple — validate per-chunk output."""
    config = load_domain_config()
    return frozenset(
        (domain["level1"], level2["level2"], level3["level3"])
        for domain in config["domains"]
        for level2 in domain["level2_list"]
        for level3 in level2["level3_list"]
    )


@lru_cache(maxsize=1)
def classification_catalog() -> str:
    """Render the level1/level2 taxonomy as compact text for the LLM prompt."""
    cfg = load_domain_config()
    lines: list[str] = []
    for dom in cfg["domains"]:
        lines.append(f"- {dom['level1']} ({dom.get('level1_vi', '')}):")
        for l2 in dom["level2_list"]:
            desc = (l2.get("description") or "").strip()
            lines.append(
                f"    * {l2['level2']} ({l2.get('level2_vi', '')}): {desc}"
            )
    return "\n".join(lines)


def is_valid_level1_level2(level1: str, level2: str) -> bool:
    """True if (level1, level2) is a known taxonomy pair."""
    return (level1, level2) in valid_level1_level2()


def is_valid_path(level1: str, level2: str, level3: str) -> bool:
    """True if (level1, level2, level3) is a known taxonomy path."""
    return (level1, level2, level3) in valid_paths()
