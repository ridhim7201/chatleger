# AGENTS.md — ChatLedger

This file is for AI coding assistants (Claude, Copilot, Cursor, etc.) working
on this codebase. Read it before making any changes. It tells you how this
project is structured, what the rules are, and where things can go wrong.

---

## What this project is

ChatLedger is a local-first, offline-capable web app that parses WhatsApp chat
exports and extracts structured data (action items, decisions, open questions)
using a locally-running Ollama model. Nothing phones home. No external APIs.
All inference runs at `localhost:11434`.

---

## Non-negotiable rules

**1. Never add external HTTP calls.**
The entire value proposition of this project is offline resiliency. If you add
any `requests.get(...)` or `fetch(...)` pointing outside `localhost`, you break
the core constraint. This includes CDN links in the frontend — no external JS,
CSS, or font imports.

**2. Never weaken Pydantic validation in schema.py.**
The strict schema is what protects the SQLite database from malformed LLM
output. If you encounter a validation error, fix the prompt or the retry logic
in `extractor.py` — do not loosen the schema to make errors disappear.

**3. Never skip the retry/fallback in extractor.py.**
The two-attempt retry with graceful fallback to empty `ExtractionResult` is
required behavior, not optional error handling. A crashed pipeline from one
bad chunk is a much worse outcome than a few missed items.

**4. Never alter message_ids after parser.py assigns them.**
`msg_0`, `msg_1`, ... are the foreign keys used across the entire system. If
you renumber, rename, or reassign them anywhere downstream, you break all
traceability and the SQLite FK relationships.

**5. Never mock Ollama in main.py or db.py.**
Mocking belongs in tests only. Production code must always make real calls
to `localhost:11434`. If Ollama isn't running, the `/health` endpoint should
return a clear error, not silently succeed.

---

## Where to put things

| What | Where |
|---|---|
| New extraction fields | `schema.py` (add to model) + `db.py` (add column + migration) |
| Prompt changes | `extractor.py` — the `EXTRACTION_PROMPT` constant at the top |
| New API endpoint | `main.py` only |
| Parsing edge cases | `parser.py` + corresponding test in `test_parser.py` |
| Chunking strategy changes | `chunker.py` + update `SKILL.md` defaults section |
| Dedup threshold changes | `merge.py` + update `SKILL.md` merge section |
| New UI features | `frontend/index.html` only — no new files, keep it single-file |

---

## What each file does (one line each)

- `schema.py` — Pydantic models; the source of truth for what valid extraction output looks like
- `parser.py` — turns raw .txt into List[dict] message rows; pure string ops, no AI
- `chunker.py` — splits message rows into overlapping windows for LLM context management
- `extractor.py` — calls Ollama, retries on failure, caches results, returns ExtractionResult
- `merge.py` — combines per-chunk results, deduplicates, resolves answered questions
- `db.py` — SQLite init, insert, and query; FK traceability between items and source messages
- `main.py` — FastAPI app wiring all of the above; exposes upload/results/export/health
- `frontend/index.html` — single-file vanilla JS UI; fetch-only, no external dependencies

---

## How the pipeline runs

```
POST /upload (main.py)
    → parse_chat()           (parser.py)
    → chunk_messages()       (chunker.py)
    → [for each chunk]
        extract_chunk()      (extractor.py)
            → cache check
            → Ollama call
            → Pydantic validation
            → retry if needed
            → fallback to empty on double failure
    → merge_results()        (merge.py)
    → insert_messages()      (db.py)
    → insert_extraction_result() (db.py)
    → return summary JSON
```

---

## How to add a new extraction category

1. Add a new Pydantic model in `schema.py`
2. Add it as a field on `ExtractionResult` with `default=[]`
3. Add the new category's instructions and schema to `EXTRACTION_PROMPT` in `extractor.py`
4. Add the SQLite table in `db.py` `init_db()` and a new `insert_*` function
5. Add a GET endpoint or extend `/results` in `main.py`
6. Add a new tab/section in `frontend/index.html`
7. Add test cases in `tests/`

Do all seven steps. A partial implementation (e.g. schema added but no table)
will cause a silent data loss bug.

---

## Testing expectations

- All tests in `/tests` must pass with `pytest tests/ -v` before any PR
- `test_extractor.py` must mock `requests.post` — never hit real Ollama in CI
- The retry path and double-failure path must both have explicit test coverage
- Adding a new module means adding a corresponding `test_{module}.py`

---

## CI checks (all must pass)

```
black --check .
ruff check .
mypy backend/
bandit -r backend/
pytest tests/ -v
```

If any of these fails, do not merge. Fix the issue, don't skip the check.

---

## Things that look wrong but aren't

- `format: "json"` in the Ollama call — this is intentional, it constrains
  sampling-level output to valid JSON syntax
- Empty `ExtractionResult()` returned on extraction failure — this is the
  graceful fallback, not a bug
- `PRAGMA foreign_keys = ON` in `db.py` — SQLite doesn't enforce FKs by
  default; this line is required
- `temperature: 0.1` instead of `0` — some models behave oddly at exactly 0;
  0.1 gives slightly more reliable format adherence

---

## Things that are actually wrong if you see them

- Any `requests.get(...)` pointing outside localhost
- `ExtractionResult` fields set to `None` instead of `[]`
- `message_id` values that aren't in the format `msg_{integer}`
- Any test that imports and calls `extract_chunk()` without mocking `requests.post`
- Any `index.html` that loads a script from a CDN URL