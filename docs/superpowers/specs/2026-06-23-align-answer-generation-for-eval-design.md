# Align answer-generation (general-agent) for EnterpriseRAG-Bench eval

**Date:** 2026-06-23
**Scope:** `general-agent/` answer-generation path only. Scoring/extraction on the
benchmark side lives in `src/` and is OUT of scope.

## Problem

`general-agent/` produces `{question_id, answer, document_ids}` for the benchmark
scorer (`src/scripts/answer_evaluation/metrics_based_eval.py`). The format already
matches. The risk is in HOW the answer is generated, specifically the path that
yields `document_ids`:

`document_ids` is derived from citations the agent writes in its answer. The
orchestrator parses them with a strict regex and resolves them to full doc_ids.
The chain only works if the agent emits citations in the EXACT shape the regex
expects — and today the system prompt describes that shape inaccurately, inviting
the model to deviate. There is also a domain-filter parameter that can only hurt
in this corpus, and an eval-side over-listing fallback that conflicts with honest
abstention.

### Verified facts (do not re-litigate)

- doc_id IS aligned with the benchmark. Ingest sets `doc_id = uuid32hex` from the
  filename (`general-agent/eval/ingest_gold_docs.py:30-34,71`); search returns the
  full `doc_id` (`search_kb.py:215-216`); `Source` keeps it
  (`schemas/retrieval.py:18`); export maps `"dsid_"+doc_id`
  (`run_agent_eval.py:40-42`). No ingestion/schema change needed.
- The citation token shown to the agent is `doc_id[:8]#p<position>`, e.g.
  `[a1b2c3d4#p3]`, lowercase (`search_kb.py:101,211`).
- The orchestrator parses citations with
  `CITATION_RE = \[([0-9a-fA-F]{6,}#p\d+)\]`, lowercases, and resolves by EXACT
  match against `cite_index` (`orchestrator.py:34,226-227`).
- `kb_get_document` expects its `doc_id` argument to be the prefix BEFORE `#p`
  (`get_document.py:85-87`).
- All gold-docs are ingested under one domain (`general_text`,
  `ingest_gold_docs.py:42-43`), so any value of the `domains` filter silently
  excludes valid results.

## Decisions (locked with user)

1. **Citation reliability = prompt-only.** Tighten the system prompt so the agent
   emits citations matching the existing regex exactly. Do NOT add tolerant/fuzzy
   parsing in code; keep the orchestrator's exact parser as-is.
2. **Remove domain logic entirely** from the answer-generation path (not a toggle).
3. **Harness stays thin.** `run_agent_eval` writes `document_ids` straight from
   `done["sources"]`; no eval-side policy (no over-listing fallback, no
   precision-first layer). Scoring is the benchmark's job (`src/`).
4. **Chunking unchanged.** Keep the current parent-child splitter
   (`ingestion/splitters/`). No change.
5. **Validation before any paid run** via a notebook that answers ONE question and
   checks the output format.

## Design

### Component 1 — `general-agent/eval/prompts_eval.py` (the core fix)

Rewrite the citation block of `EVAL_SYSTEM_PROMPT` so it aligns with
`CITATION_RE` and the tool's actual token shape:

- Describe the token by what it IS: "copy the bracketed reference EXACTLY as the
  tool prints it, e.g. if the tool shows `[a1b2c3d4#p3]`, write `[a1b2c3d4#p3]`."
  Remove the misleading "doc_id / written in full" wording that implies expanding
  to a 32-hex id the model never saw.
- State the regex-critical constraints plainly: no whitespace inside the brackets
  (the hex must come immediately after `[`); keep `#p`; lowercase; one reference
  per bracket pair; for multiple sources put each in its own brackets back-to-back
  (`[a1b2c3d4#p0][e5f6a7b8#p2]`); never a comma or two refs in one pair.
- Clarify `kb_get_document`: pass the part BEFORE `#p` (e.g. `a1b2c3d4`) as
  `doc_id`, plus the positions.
- Keep the already-added abstain rule (not found ⇒ say so, cite NOTHING) and the
  completeness-over-brevity rule — both shape answer generation.
- Remove the sentence mentioning the `domains` filter (the param will be gone).

Update `NEUTRAL_KB_DESCRIPTION` to drop the `domains` sentence and align the
`kb_get_document` phrasing.

### Component 2 — `general-agent/retrieval/tools/search_kb.py` (remove domain logic)

- Remove the `domains` property from `spec().input_schema` and the `domains`
  parameter from `kb_search(...)`.
- Delete `_domain_enum()` and `_domain_gloss()` (only used to build the domains
  enum/description).
- Call `_store.hybrid_search_children(..., domains=None)` and drop domain-related
  branches in the no-results guidance text.
- `_store`/Weaviate keep accepting `domains=None`, so lower layers are untouched.

### Component 3 — `general-agent/eval/run_agent_eval.py` (revert to thin)

- Remove `_doc_ids_from_answer`, `_CITE_TOKEN_RE`, the precision-first commentary,
  and `n_recovered`.
- `document_ids = _doc_ids_from_sources(done["sources"])` (cited docs only). No
  over-listing fallback to `surfaced_sources`.
- Keep `_meta` observability fields (`question_type`, `rounds`, `latency_ms`,
  `flags`, `n_cited`, `n_surfaced`).

### Component 4 — chunking

No change. Current parent-child splitter stays.

### Component 5 — validation

`notebooks/validate_one_question.ipynb`:

- Run `answer_one` for ONE real question end-to-end (needs Weaviate + OpenAI key).
- Print `answer`, `document_ids`, `_meta`; assert the row has exactly
  `{question_id, answer, document_ids}` and that `document_ids` are `dsid_`-prefixed
  and (for a gold-bearing question) intersect `expected_doc_ids`.
- Re-run the existing orchestrator tests for the copy; update any test that
  references the removed `domains` parameter.

## Data flow (after changes)

```
agent answer (cites [8hex#pN], exactly as prompted)
  └─ orchestrator CITATION_RE (exact, unchanged) → done["sources"] (full doc_id)
  └─ run_agent_eval: doc_id → "dsid_"+doc_id, dedupe → document_ids
  └─ answers_general_agent.jsonl {question_id, answer, document_ids}
        → scored by src/ (out of scope)
```

## Error handling / edge cases

- **No citation** (agent abstains or forgets): `done["sources"]` empty ⇒
  `document_ids = []`. Correct for `info_not_found`/`high_level`; for gold-bearing
  questions it is a recall miss, surfaced via `flags`/`n_surfaced` for analysis.
- **Malformed citation** the regex rejects: dropped by the existing parser
  (`unknown_citation` flag). The prompt fix is what minimizes this.
- **8-hex prefix collision** (`resolve_doc_id_prefix`, `kb_get_document` cold
  path): probability ~6e-5 over 722 docs. Accepted residual risk; not addressed.

## Out of scope

- Any eval/scoring logic (handled in `src/`).
- Tolerant/fuzzy citation parsing.
- Ingestion, chunking, embedding, Weaviate schema.
- The original `general-agent` project at `/home/boltbolt/Desktop/general-agent`.

## Success criteria

- The notebook answers one question, output is exactly the 3-key format, and
  `document_ids` for a gold-bearing question match `expected_doc_ids`.
- Orchestrator tests pass after the domain removal.
