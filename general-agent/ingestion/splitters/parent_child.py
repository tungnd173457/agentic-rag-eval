"""Parent/child structural split (3B, stages 1-4).

Pipeline (level3 classification is stage 5, in ``classifiers.level3_classifier``):

  1. STRUCTURAL SPLIT — cut PARENT chunks at H1/H2 headings; H3+ stay inside a
     parent; tables stay atomic; a heading with no body folds into the next
     parent. No-heading docs are cut by paragraph to the parent target.
  2. MIN-SIZE MERGE   — merge tiny (< PC_PARENT_MIN_CHARS) SIBLING parents (sections that
     share the same PARENT heading — the anchor excludes the section's own
     heading, so H2 sections under one H1 are siblings; top-level sections are
     chapters and never merge with each other), forward then backward.
  2b. FLOOR FOLD      — a non-atomic parent still under PC_PARENT_MIN_CHARS
     after the sibling merge folds into an adjacent non-atomic parent EVEN
     across sections (prev preferred, else next) — last resort so no runt
     chunk reaches the index. Content is never dropped: if both neighbours
     are atomic/over-budget, the runt stays.
  3. SIZE CAP         — split over-long parents block-aware: prefer H3
     boundaries, then pack whole blocks toward a BALANCED target (even pieces,
     so the cap never strands a sub-MIN tail); only oversized PROSE blocks are
     sentence-split. Tables — standalone or inside a mixed parent — split by
     ROW with the header repeated, never mid-row.
  3b. RESCUE RUNTS    — a sub-MIN parent stuck between two near-MAX neighbours
     (folding would overflow MAX) is merged with a neighbour and re-capped
     balanced, redrawing the boundary into pieces that each clear MIN. Only
     commits when every piece lands in [MIN, MAX]; an indivisible union stays.
  3c. CONSOLIDATE     — every remaining sub-MIN parent (text OR atomic table,
     incl. first/last) folds into its smaller neighbour to a fixpoint; a union
     containing a table may overflow PARENT_MAX up to PARENT_MAX+PARENT_MIN
     (text-only stays ≤ PARENT_MAX). Kept only when nothing fits — whole doc
     < MIN (bill) or both neighbours too big.
  4. CHILD CHUNKS     — per parent, sliding character windows (overlap within
     the parent only); table parents emit one child per row group, header repeated.

Children embed/search; parents give the LLM context. Sizes are measured in
CHARACTERS (develop ``common/parent_child`` convention — not tokens). All
thresholds are env-tunable.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from config import app_config

from .blocks import Block, parse_blocks, table_header
from .clean import clean_markdown

# Stage thresholds (CHARACTERS) — read off ``app_config`` (env > .env >
# defaults). Names/values follow develop's parent_child config (PC_*_CHARS);
# tests monkeypatch ``app_config.PC_*`` to exercise the relational guards.


def _validate_thresholds() -> None:
    """Fail fast on inconsistent env values.

    Purely RELATIONAL checks between the real knobs (+ positivity) — no
    arbitrary absolute bounds: the sizes come from the operator's trusted
    .env, so internal consistency is the only thing worth enforcing.
    """
    for name, value in (
        ("PC_PARENT_MAX_CHARS", app_config.PC_PARENT_MAX_CHARS),
        ("PC_CHILD_MAX_CHARS", app_config.PC_CHILD_MAX_CHARS),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    for name, value in (
        ("PC_PARENT_MIN_CHARS", app_config.PC_PARENT_MIN_CHARS),
        ("PC_CHILD_MIN_CHARS", app_config.PC_CHILD_MIN_CHARS),
        ("PC_CHILD_OVERLAP", app_config.PC_CHILD_OVERLAP),
    ):
        if value < 0:
            raise ValueError(f"{name} must not be negative, got {value}.")
    if app_config.PC_CHILD_OVERLAP > app_config.PC_CHILD_MAX_CHARS:
        raise ValueError(
            f"Got a larger chunk overlap ({app_config.PC_CHILD_OVERLAP}) than chunk size "
            f"({app_config.PC_CHILD_MAX_CHARS}), should be smaller."
        )
    if app_config.PC_CHILD_MIN_CHARS > app_config.PC_CHILD_MAX_CHARS:
        raise ValueError(
            f"PC_CHILD_MIN_CHARS ({app_config.PC_CHILD_MIN_CHARS}) must not exceed "
            f"PC_CHILD_MAX_CHARS ({app_config.PC_CHILD_MAX_CHARS})."
        )
    if app_config.PC_CHILD_MAX_CHARS > app_config.PC_PARENT_MAX_CHARS:
        raise ValueError(
            f"PC_CHILD_MAX_CHARS ({app_config.PC_CHILD_MAX_CHARS}) must not exceed "
            f"PC_PARENT_MAX_CHARS ({app_config.PC_PARENT_MAX_CHARS}) — a child cannot outgrow its parent."
        )
    if app_config.PC_PARENT_MIN_CHARS > app_config.PC_PARENT_MAX_CHARS:
        raise ValueError(
            f"PC_PARENT_MIN_CHARS ({app_config.PC_PARENT_MIN_CHARS}) must not exceed "
            f"PC_PARENT_MAX_CHARS ({app_config.PC_PARENT_MAX_CHARS})."
        )


_validate_thresholds()


def char_len(text: str) -> int:
    """Chunk size in characters — the single sizing measure for 3B."""
    return len(text)


def text_hash(text: str) -> str:
    """sha256 of the chunk body (develop's per-chunk ``doc_hash`` convention)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Sentence boundary: end punctuation (incl. Vietnamese) or a newline.
_SENT_RE = re.compile(r"(?<=[.!?;…])\s+|\n+")


@dataclass
class Parent:
    blocks: list[Block]
    title: str
    parent_stack: list[str]
    anchor: tuple                      # identity of the H1/H2 ancestor (merge key)
    is_atomic: bool = False
    parent_id: str = ""
    position: int = 0
    text: str = field(default="")
    char_count: int = 0
    doc_hash: str = ""                 # sha256(text) — per-chunk identity/dedup
    child_ids: list[str] = field(default_factory=list)


@dataclass
class Child:
    parent_id: str
    position: int                      # index within the parent
    text: str
    char_count: int
    child_id: str = ""
    doc_hash: str = ""                 # sha256(text) — per-chunk identity/dedup


def _join(blocks: list[Block]) -> str:
    return "\n\n".join(b.md for b in blocks).strip()


def _sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENT_RE.split(text)) if s]


# ---------------------------------------------------------------- stage 1
def _structural_split(blocks: list[Block]) -> list[Parent]:
    """Cut parents at H1/H2 boundaries; fold body-less headings forward."""
    parents: list[Parent] = []
    stack: list[tuple[int, str]] = []   # (level, text) heading ancestors
    cur: Parent | None = None
    has_content = False
    pending: list[Block] = []           # body-less section → opens the NEXT parent

    def flush():
        nonlocal cur, has_content
        if cur is not None:
            if has_content:
                parents.append(cur)
            else:
                # Body-less section (heading blocks only): never drop it —
                # fold its blocks forward as the next parent's opening lines.
                pending.extend(cur.blocks)
        cur = None
        has_content = False

    def open_parent():
        nonlocal pending
        anc = [t for lvl, t in stack if lvl <= 2]
        # SIBLINGS share the same PARENT heading: the anchor EXCLUDES the
        # section's own heading (anc[:-1]) so e.g. tiny H2 sections under one
        # H1 can merge in stage 2. A TOP-LEVEL section keeps its own title as
        # anchor — chapters never merge with each other.
        anchor = tuple(anc[:-1]) if len(anc) >= 2 else tuple(anc)
        p = Parent(blocks=list(pending), title=anc[-1] if anc else "",
                   parent_stack=list(anc), anchor=anchor)
        pending = []
        return p

    for b in blocks:
        if b.kind == "heading":
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.text))
            if b.level <= 2:
                flush()
                cur = open_parent()
                cur.blocks.append(b)
                cur.title = b.text
            else:
                if cur is None:
                    cur = open_parent()
                cur.blocks.append(b)
                if not has_content:        # title = heading nearest the content
                    cur.title = b.text
        else:
            if cur is None:
                cur = open_parent()
            cur.blocks.append(b)
            has_content = True
    flush()
    if pending:
        # Trailing body-less heading(s) — no next parent to fold into; keep
        # them as a runt parent (stage 2b folds it backward) so no text is lost.
        parents.append(open_parent())

    for p in parents:
        p.is_atomic = any(b.kind == "table" for b in p.blocks) and not any(
            b.kind == "para" for b in p.blocks
        )
        p.text = _join(p.blocks)
        p.char_count = char_len(p.text)
    return parents


# ---------------------------------------------------------------- stage 2
def _merge_small(parents: list[Parent]) -> list[Parent]:
    """Merge tiny SIBLING parents — sections sharing the same parent heading
    (anchor excludes the section's own heading) — forward, then backward.
    Top-level sections have distinct anchors, so chapters never merge here;
    leftovers below PC_PARENT_MIN_CHARS are swept cross-section by ``_floor_fold``."""
    merged: list[Parent] = []
    for p in parents:
        if (
            merged
            and not p.is_atomic
            and not merged[-1].is_atomic
            and merged[-1].anchor == p.anchor
            and merged[-1].char_count < app_config.PC_PARENT_MIN_CHARS
            and merged[-1].char_count + p.char_count <= app_config.PC_PARENT_MAX_CHARS
        ):
            _absorb(merged[-1], p)
        else:
            merged.append(p)

    # Backward pass: a trailing tiny parent with no following sibling folds back.
    for i in range(len(merged) - 1, 0, -1):
        p = merged[i]
        prev = merged[i - 1]
        if (
            p.char_count < app_config.PC_PARENT_MIN_CHARS
            and not p.is_atomic
            and not prev.is_atomic
            and prev.anchor == p.anchor
            and prev.char_count + p.char_count <= app_config.PC_PARENT_MAX_CHARS
        ):
            _absorb(prev, p)
            merged.pop(i)
    return merged


def _absorb(into: Parent, other: Parent) -> None:
    into.blocks += other.blocks
    into.text = _join(into.blocks)
    into.char_count = char_len(into.text)


def _prepend(into: Parent, other: Parent) -> None:
    """Fold ``other`` in FRONT of ``into`` (keeps ``into``'s title/anchor)."""
    into.blocks = other.blocks + into.blocks
    into.text = _join(into.blocks)
    into.char_count = char_len(into.text)


# ---------------------------------------------------------------- stage 2b
def _floor_fold(parents: list[Parent]) -> list[Parent]:
    """Hard floor (PC_PARENT_MIN_CHARS): fold every non-atomic parent still
    under the floor into an adjacent non-atomic parent EVEN ACROSS sections —
    so no under-MIN chunk (lone signature block, one-line section, runt minted
    by the size-cap) reaches the index. Prefers folding backward into the
    previous parent; falls forward otherwise.

    Runs each single pass to a FIXPOINT: a fold that grows a neighbour can leave
    *it* still under MIN (e.g. a backward-absorbed runt that itself was stuck),
    so we re-scan until a whole pass folds nothing — guaranteeing the result is
    fully merged toward MIN. Content is never dropped and MAX is never crossed:
    a runt whose only neighbours are atomic or would overflow PARENT_MAX stays
    (genuinely unavoidable — e.g. the whole doc is under MIN, or it is sandwiched
    between atomic tables)."""
    if app_config.PC_PARENT_MIN_CHARS <= 0:
        return parents
    out = list(parents)
    while True:
        folded = _floor_fold_pass(out)
        if len(folded) == len(out):       # a whole pass changed nothing → fixpoint
            return folded
        out = folded


def _floor_fold_pass(parents: list[Parent]) -> list[Parent]:
    """One left-to-right fold pass — see ``_floor_fold`` for the full contract."""
    out = list(parents)
    i = 0
    while i < len(out):
        p = out[i]
        if p.is_atomic or p.char_count >= app_config.PC_PARENT_MIN_CHARS:
            i += 1
            continue
        prev = out[i - 1] if i > 0 else None
        nxt = out[i + 1] if i + 1 < len(out) else None
        if prev is not None and not prev.is_atomic \
                and prev.char_count + p.char_count <= app_config.PC_PARENT_MAX_CHARS:
            _absorb(prev, p)
            out.pop(i)            # re-examine the element now at i
            continue
        if nxt is not None and not nxt.is_atomic \
                and nxt.char_count + p.char_count <= app_config.PC_PARENT_MAX_CHARS:
            _prepend(nxt, p)
            out.pop(i)            # combined parent now sits at i — re-examine
            continue
        i += 1                    # nowhere to fold — keep the runt
    return out


# ---------------------------------------------------------------- stage 3
def _size_cap(parents: list[Parent]) -> list[Parent]:
    """Split over-long parents; each fragment becomes its own parent."""
    out: list[Parent] = []
    for p in parents:
        if p.is_atomic:
            out += _cap_table(p) if p.char_count > app_config.PC_PARENT_MAX_CHARS else [p]
        else:
            out += _cap_text(p) if p.char_count > app_config.PC_PARENT_MAX_CHARS else [p]
    return out


# ---------------------------------------------------------------- stage 3b
def _has_table(p: Parent) -> bool:
    return any(b.kind == "table" for b in p.blocks)


def _rescue_runts(parents: list[Parent]) -> list[Parent]:
    """Resolve the MIN-vs-MAX conflict ``_floor_fold`` cannot: a sub-MIN parent
    whose BOTH non-atomic neighbours are already near PARENT_MAX, so folding
    either way would overflow MAX. Instead of leaving the runt, MERGE it with an
    adjacent neighbour (next preferred, else prev) and REDRAW the boundary by
    sentence-packing the union toward a BALANCED target — so the resulting pieces
    each clear MIN and stay ≤ MAX. Sentence boundaries keep meaning intact (never
    mid-sentence) and heading lines stay whole; block/heading cuts are preferred
    implicitly because whole lines are the packing units.

    Commits only when EVERY piece lands in [MIN, MAX]; if the union cannot be
    balanced that way (e.g. one indivisible giant sentence) the runt is left
    untouched — content is never dropped, MAX is never exceeded by a committed
    piece. Tables are left alone (a table-bearing parent is skipped) so row
    structure is never shredded."""
    if app_config.PC_PARENT_MIN_CHARS <= 0:
        return parents
    out = list(parents)
    i = 0
    while i < len(out):
        p = out[i]
        if p.is_atomic or p.char_count >= app_config.PC_PARENT_MIN_CHARS or _has_table(p):
            i += 1
            continue
        for j in (i + 1, i - 1):                      # extend forward first, then backward
            if not (0 <= j < len(out)):
                continue
            n = out[j]
            if n.is_atomic or _has_table(n):
                continue
            left, right = (p, n) if j > i else (n, p)
            union = (left.text + "\n\n" + right.text).strip()
            target = _balanced_target(char_len(union), app_config.PC_PARENT_MAX_CHARS)
            pieces = [_clone(left, t) for t in _pack(_sentences(union), target)]
            if pieces and all(
                app_config.PC_PARENT_MIN_CHARS <= pc.char_count <= app_config.PC_PARENT_MAX_CHARS
                for pc in pieces
            ):
                lo = min(i, j)
                out[lo:max(i, j) + 1] = pieces
                i = lo                                # re-examine from the first new piece
                break
        else:
            i += 1                                    # no usable neighbour — keep the runt
    return out


# ---------------------------------------------------------------- stage 3c
def _runt_blocks(p: Parent) -> list[Block]:
    """The parent's blocks, reconstructing one if a size-cap clone dropped them
    (``_cap_table``/``_cap_text`` clones carry text only)."""
    if p.blocks:
        return list(p.blocks)
    kind = "table" if p.is_atomic else "para"
    return [Block(kind, 0, p.text, p.text, atomic=p.is_atomic)]


def _merge_blocks(left: Parent, right: Parent) -> Parent:
    """Concatenate two parents (any kind) into one — ``is_atomic`` is recomputed,
    so folding a prose runt into a table parent yields a MIXED parent whose table
    rows still drive row-children in stage 4."""
    return _clone_blocks(left, _runt_blocks(left) + _runt_blocks(right))


def _contains_table(p: Parent) -> bool:
    """A pure-table (atomic) parent, or a mixed parent holding a table block."""
    return p.is_atomic or _has_table(p)


def _merge_ceiling(a: Parent, b: Parent) -> int:
    """Ceiling for a fold/merge whose result would CONTAIN A TABLE: such a union
    may exceed PARENT_MAX up to ``PARENT_MAX + PARENT_MIN`` (tables resist clean
    splitting, so we tolerate a larger merged parent rather than break rows). A
    text-only merge stays bounded by PARENT_MAX — the initial size-cap likewise
    only ever uses PARENT_MAX."""
    if _contains_table(a) or _contains_table(b):
        return app_config.PC_PARENT_MAX_CHARS + app_config.PC_PARENT_MIN_CHARS
    return app_config.PC_PARENT_MAX_CHARS


def _consolidate_runts(parents: list[Parent]) -> list[Parent]:
    """Final consolidation: EVERY parent still under MIN — text OR atomic table,
    ANYWHERE incl. the first/last — folds into its smaller adjacent neighbour,
    repeated to a fixpoint, so no small fragment is left stranded. In particular a
    small atomic table wedged between sections (which ``_floor_fold`` and
    ``_rescue_runts`` both skip) is absorbed into a neighbour as a MIXED parent
    (its table rows still drive row-children in stage 4).

    The ceiling is table-conditional (``_merge_ceiling``): a union CONTAINING A
    TABLE may exceed PARENT_MAX up to ``PARENT_MAX + PARENT_MIN``; a text-only
    union stays ≤ PARENT_MAX. A runt is kept ONLY when no neighbour fits even that
    relaxed ceiling — the whole doc is one short parent (a bill), or both
    neighbours are too big. Content is never dropped; no committed parent exceeds
    the ceiling."""
    if app_config.PC_PARENT_MIN_CHARS <= 0:
        return parents
    out = list(parents)
    i = 0
    while i < len(out):
        p = out[i]
        if p.char_count >= app_config.PC_PARENT_MIN_CHARS:
            i += 1
            continue
        cands = [j for j in (i - 1, i + 1)
                 if 0 <= j < len(out)
                 and out[j].char_count + p.char_count <= _merge_ceiling(p, out[j])]
        if not cands:
            i += 1                       # no neighbour fits even the relaxed ceiling — keep
            continue
        j = min(cands, key=lambda k: out[k].char_count)   # cluster smallest first
        left, right = (p, out[j]) if j > i else (out[j], p)
        lo = min(i, j)
        out[lo:max(i, j) + 1] = [_merge_blocks(left, right)]
        i = lo                           # re-examine the merged parent
    return out


def _clone(src: Parent, text: str) -> Parent:
    return Parent(blocks=[], title=src.title, parent_stack=list(src.parent_stack),
                  anchor=src.anchor, is_atomic=src.is_atomic, text=text,
                  char_count=char_len(text))


def _clone_blocks(src: Parent, blocks: list[Block]) -> Parent:
    """Fragment parent that KEEPS its blocks, so stage 4 still sees table
    boundaries (mixed fragments route tables to row-children, not sentences)."""
    text = _join(blocks)
    atomic = any(b.kind == "table" for b in blocks) and not any(
        b.kind == "para" for b in blocks
    )
    return Parent(blocks=list(blocks), title=src.title,
                  parent_stack=list(src.parent_stack), anchor=src.anchor,
                  is_atomic=atomic, text=text, char_count=char_len(text))


def _balanced_target(total: int, limit: int) -> int:
    """Even fragment size for the size-cap: split into ``ceil(total/limit)``
    pieces of ~equal size rather than greedily filling each to ``limit``.

    Greedy packing strands the remainder — a 21k section under a 10k ceiling
    becomes [10k, 10k, 1k], and that 1k tail is a runt whose 10k sibling has no
    room to re-absorb it. Targeting ``total/k`` instead yields [7k, 7k, 7k]. The
    target stays in ``(limit/2, limit]`` (for total > limit), hence well above
    PC_PARENT_MIN_CHARS, so a balanced cap never mints a sub-MIN tail."""
    k = max(1, (total + limit - 1) // limit)
    return (total + k - 1) // k


def _cap_text(p: Parent) -> list[Parent]:
    """Block-aware cap: prefer H3 boundaries, then pack whole BLOCKS toward a
    BALANCED target (``_balanced_target`` — even pieces ≤ PARENT_MAX, so capping
    never strands a sub-MIN remainder). Adjacent H3 groups are PACKED together so
    a run of tiny heading-only sections (cover pages, repeated page headers/
    footers) never each become their own runt parent. Only prose blocks are
    ever sentence-split; a table inside a mixed parent moves whole (row-split
    above PARENT_MAX) — never mid-row."""
    target = _balanced_target(p.char_count, app_config.PC_PARENT_MAX_CHARS)
    frags: list[list[Block]] = []
    cur: list[Block] = []
    cur_len = 0
    for grp in _split_at_h3(p.blocks):
        glen = char_len(_join(grp))
        if glen > app_config.PC_PARENT_MAX_CHARS:   # oversized group: flush, then split alone
            if cur:
                frags.append(cur)
                cur, cur_len = [], 0
            frags += _pack_blocks(grp, target)
            continue
        if cur and cur_len + glen > target:         # flush at the BALANCED target (≤ PARENT_MAX)
            frags.append(cur)
            cur, cur_len = [], 0
        cur += grp
        cur_len += glen
    if cur:
        frags.append(cur)
    return [_clone_blocks(p, f) for f in frags if f] or [p]


def _split_at_h3(blocks: list[Block]) -> list[list[Block]]:
    """Group blocks into runs that each start at an H3+ heading."""
    groups: list[list[Block]] = []
    for b in blocks:
        if b.kind == "heading" and b.level >= 3 and groups and groups[-1]:
            groups.append([b])
        elif groups:
            groups[-1].append(b)
        else:
            groups.append([b])
    return [g for g in groups if g]


def _pack_blocks(blocks: list[Block], target: int) -> list[list[Block]]:
    """Pack whole blocks into fragments toward ``target`` (≤ PARENT_MAX, balanced
    by the caller so no sub-MIN tail is stranded).

    ``target`` is the PACKING threshold; the absolute limit PARENT_MAX gates
    whether a SINGLE block (prose or table) must be split.
    Tables are ATOMIC: one packs with neighbours only if the total fits,
    stands alone up to PARENT_MAX, and row-splits (header repeated) above —
    sentence-splitting never touches a table. Oversized prose blocks fall
    back to sentence packing (also toward ``target``).
    """
    frags: list[list[Block]] = []
    cur: list[Block] = []
    cur_len = 0

    def close():
        nonlocal cur, cur_len
        if cur:
            frags.append(cur)
        cur, cur_len = [], 0

    for b in blocks:
        size = char_len(b.md)
        if b.kind == "table":
            if size > app_config.PC_PARENT_MAX_CHARS:              # row-split, header repeated
                close()
                for piece in _table_pieces(b.md, app_config.PC_PARENT_MAX_CHARS):
                    frags.append([Block("table", 0, piece, piece, atomic=True)])
                continue
            if cur and cur_len + size > target:
                close()
            cur.append(b)
            cur_len += size
            if cur_len > target:                                   # a lone table may exceed the
                close()                        # balanced target, up to PARENT_MAX
            continue
        if size > app_config.PC_PARENT_MAX_CHARS:                  # oversized prose block
            close()
            for piece in _pack(_sentences(b.md), target):
                frags.append([Block("para", 0, piece, piece)])
            continue
        if cur and cur_len + size > target:
            close()
        cur.append(b)
        cur_len += size
    close()
    return frags


def _table_pieces(table_md: str, limit: int) -> list[str]:
    """Split a GFM table into ≤``limit`` pieces by ROW, repeating header+rule."""
    header, data = table_header(table_md)
    if not header or not data:
        return [table_md]
    pieces, cur = [], []
    for row in data:
        cand = "\n".join([header, *cur, row])
        if cur and char_len(cand) > limit:
            pieces.append("\n".join([header, *cur]))
            cur = [row]
        else:
            cur.append(row)
    if cur:
        pieces.append("\n".join([header, *cur]))
    return pieces


def _cap_table(p: Parent) -> list[Parent]:
    """Split a long table by rows, repeating header+rule in each fragment."""
    pieces = _table_pieces(p.text, app_config.PC_PARENT_MAX_CHARS)
    if len(pieces) == 1:
        return [p]
    return [_clone(p, piece) for piece in pieces]


def _pack(units: list[str], limit: int) -> list[str]:
    """Greedily pack units (sentences/rows) into fragments under ``limit``.

    ``cur_tok`` counts the JOINED length (units + one separator each) so a
    fragment never exceeds ``limit`` by the join spaces.
    """
    out, cur, cur_tok = [], [], 0
    for u in units:
        ut = char_len(u) + (1 if cur else 0)   # +1 for the " " join separator
        if cur and cur_tok + ut > limit:
            out.append(" ".join(cur))
            cur, cur_tok = [], 0
            ut = char_len(u)
        cur.append(u)
        cur_tok += ut
    if cur:
        out.append(" ".join(cur))
    return out


# ---------------------------------------------------------------- stage 4
def _make_children(parent: Parent) -> list[Child]:
    """Children for one parent. Tables (even inside a mixed parent) split by row
    with the header repeated; prose splits into sentence windows. Overlap stays
    within a text segment — it never bleeds across a table or out of the parent."""
    if parent.blocks:
        children: list[Child] = []
        pos = 0
        run: list[str] = []

        def flush_text():
            nonlocal pos, run
            if run:
                ch, pos = _text_children("\n\n".join(run), parent.parent_id, pos)
                children.extend(ch)
                run = []

        for b in parent.blocks:
            if b.kind == "table":
                flush_text()
                ch, pos = _table_children(b.md, parent.parent_id, pos)
                children.extend(ch)
            else:
                run.append(b.md)
        flush_text()
        return children

    # Cloned parent (from size cap): homogeneous — a table or pure prose.
    seg = _table_children if parent.is_atomic else _text_children
    return seg(parent.text, parent.parent_id, 0)[0]


def _text_children(text: str, parent_id: str, start_pos: int) -> tuple[list[Child], int]:
    """Sentence-window children over a prose segment; overlap by CHILD_OVERLAP."""
    sents = _sentences(text)
    if not sents:
        return [], start_pos
    children: list[Child] = []
    cur: list[str] = []
    cur_tok = 0
    pos = start_pos
    for s in sents:
        st = char_len(s)
        if cur and cur_tok + st > app_config.PC_CHILD_MAX_CHARS:
            body = " ".join(cur)
            children.append(Child(parent_id, pos, body, char_len(body)))
            pos += 1
            cur, cur_tok = _overlap_tail(cur)
        cur.append(s)
        cur_tok += st
    if cur:
        body = " ".join(cur)
        # Fold a tiny trailing remnant back into the previous child.
        if children and char_len(body) < app_config.PC_CHILD_MIN_CHARS:
            prev = children[-1]
            merged = prev.text + " " + body
            children[-1] = Child(parent_id, prev.position, merged, char_len(merged))
        else:
            children.append(Child(parent_id, pos, body, char_len(body)))
            pos += 1
    return children, pos


def _overlap_tail(sents: list[str]) -> tuple[list[str], int]:
    """Trailing sentences worth up to CHILD_OVERLAP tokens, to seed the next child."""
    tail: list[str] = []
    tok = 0
    for s in reversed(sents):
        st = char_len(s)
        if tok + st > app_config.PC_CHILD_OVERLAP:
            break
        tail.insert(0, s)
        tok += st
    return tail, tok


def _table_children(table_md: str, parent_id: str, start_pos: int) -> tuple[list[Child], int]:
    """One child per group of table rows, header repeated, under CHILD_MAX."""
    header, data = table_header(table_md)
    if not header:
        return [Child(parent_id, start_pos, table_md, char_len(table_md))], start_pos + 1
    children, cur, pos = [], [], start_pos
    for row in data:
        cand = "\n".join([header, *cur, row])
        if cur and char_len(cand) > app_config.PC_CHILD_MAX_CHARS:
            body = "\n".join([header, *cur])
            children.append(Child(parent_id, pos, body, char_len(body)))
            pos += 1
            cur = [row]
        else:
            cur.append(row)
    if cur:
        body = "\n".join([header, *cur])
        children.append(Child(parent_id, pos, body, char_len(body)))
        pos += 1
    return children, pos


# ---------------------------------------------------------------- driver
def split_document(markdown: str) -> tuple[list[Parent], list[Child]]:
    """Clean, then run stages 1-4 (incl. 2b floor fold, 3b runt rescue, 3c
    overflow fold). Returns (parents, children) with ids, positions and
    per-chunk ``doc_hash``."""
    parents = _structural_split(parse_blocks(clean_markdown(markdown)))
    parents = _floor_fold(_merge_small(parents))   # stage 2 (sibling merge) + 2b (floor)
    # Stage 3 size-cap can mint fresh runts when it fragments a giant parent at
    # H3 boundaries (e.g. a lone heading-only group), so re-sweep the floor
    # AFTER the cap — the first 2b pass ran before any of these existed.
    parents = _floor_fold(_size_cap(parents))
    # Stage 3b: a sub-MIN parent stuck between two near-MAX neighbours can't fold
    # (MAX would overflow) — merge with a neighbour and re-cap balanced so the
    # redrawn pieces each clear MIN.
    parents = _rescue_runts(parents)
    # Stage 3c: consolidate EVERY remaining sub-MIN parent (incl. small atomic
    # tables and first/last) into its smaller neighbour, table-conditional ceiling
    # PARENT_MAX(+PARENT_MIN if union has a table); kept only if nothing fits.
    parents = _consolidate_runts(parents)

    children: list[Child] = []
    child_seq = 0
    for idx, p in enumerate(parents):
        p.position = idx
        p.parent_id = f"parent_{idx + 1:010d}"
        p.doc_hash = text_hash(p.text)
        for ch in _make_children(p):
            child_seq += 1
            ch.child_id = f"child_{child_seq:010d}"
            ch.doc_hash = text_hash(ch.text)
            p.child_ids.append(ch.child_id)
            children.append(ch)
    return parents, children
