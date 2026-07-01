# SKILL.md — ChatLedger

This file documents the technical patterns, conventions, and known behaviors
specific to this project. Read this before touching any module.

---

## Core pipeline mental model

```
.txt file
    ↓
parser.py       → List[MessageRow]        (pure string ops, no AI)
    ↓
chunker.py      → List[List[MessageRow]]  (sliding window, overlapping)
    ↓
extractor.py    → List[ExtractionResult]  (one Ollama call per chunk)
    ↓
merge.py        → ExtractionResult        (dedup + answered-question resolution)
    ↓
db.py           → SQLite                  (insert + FK traceability)
```

Each stage is independently testable. Never skip stages or shortcut between them.

---

## parser.py

### WhatsApp export format

```
[DD/MM/YY, HH:MM:SS] Sender Name: message text
```

Multi-line messages have no timestamp on continuation lines. They must be
appended to the previous message's text, not treated as new rows.

System messages (e.g. "Messages and calls are end-to-end encrypted") have no
sender field. These should be parsed as `sender: "SYSTEM"` and excluded from
chunking/extraction.

### Known edge cases

- Messages containing colons in the body (e.g. "Meeting at 3:00 PM today") —
  split on the FIRST colon after the sender name only, not all colons
- Messages with emoji, Unicode, and line breaks — all valid, don't sanitize
- Empty lines — skip silently
- Group notifications (e.g. "You added Anjali") — treat as SYSTEM

### message_id convention

Sequential: `msg_0`, `msg_1`, `msg_2`, ... assigned during parsing. These are
the FK references used through the entire pipeline. Never reassign or re-index
after this point.

---

## chunker.py

### Default parameters

- `window_size = 15` messages per chunk
- `overlap = 3` messages shared between consecutive chunks

The overlap exists specifically so that a question asked at the end of chunk N
and answered at the start of chunk N+1 can be detected as answered during merge.

Don't change these defaults without updating the merge heuristics.

---

## extractor.py

### Ollama call contract

```python
POST http://localhost:11434/api/generate
{
  "model": "phi3:mini",       # or llama3.2:3b
  "prompt": "...",
  "stream": false,
  "format": "json",           # CRITICAL — enforces JSON-mode at sampling level
  "options": {
    "temperature": 0.1,       # low = deterministic extraction
    "num_ctx": 4096
  }
}
```

`format: "json"` constrains token sampling to valid JSON syntax. It does NOT
guarantee your schema — validate with Pydantic afterward regardless.

### Retry logic

1. First attempt: full extraction prompt
2. On `JSONDecodeError` or `ValidationError`: retry once with corrective prompt
   appending the raw error and asking the model to fix only the format
3. On second failure: log `WARNING: chunk {hash} failed extraction` and return
   empty `ExtractionResult()` — never raise, never crash the pipeline

This is intentional. A failed chunk means some items are missed, not that the
whole run is invalid. The UI shows failed chunk counts in the pipeline summary.

### Caching

Cache key = `sha256(json.dumps(chunk, sort_keys=True))`.
Cache store = `/.cache/chatledger/{hash}.json` (relative to project root).
If a cache hit exists, skip the Ollama call entirely and return the cached
`ExtractionResult`. This is the "caching under input latency" behavior.

To bust the cache for a specific chunk during debugging:
```bash
rm .cache/chatledger/{hash}.json
```

---

## schema.py — Pydantic models

```python
class ActionItem(BaseModel):
    owner: str
    task: str
    deadline: Optional[str] = None
    source_message_id: str

class Decision(BaseModel):
    decision: str
    made_by: Optional[str] = None
    source_message_id: str

class OpenQuestion(BaseModel):
    asker: str
    question: str
    answered: bool = False
    source_message_id: str

class ExtractionResult(BaseModel):
    action_items: List[ActionItem] = []
    decisions: List[Decision] = []
    open_questions: List[OpenQuestion] = []
```

Never strip or weaken these validators. The strict schema is what protects the
pipeline from malformed LLM output reaching the database.

---

## merge.py

### Dedup strategy

Use `difflib.SequenceMatcher` on the `task` field (action items) and `decision`
field (decisions). Similarity threshold: `0.85`. If two items from overlapping
chunks exceed this threshold, keep the one with a non-null `deadline`/`made_by`,
or the first one if both are equal.

### Open question resolution

A question is marked `answered=True` during merge if a later chunk contains an
action item or decision whose text shares significant keyword overlap with the
question text (stopword-filtered). This is a heuristic — it reduces false
positives in the results UI but is not guaranteed correct. The source message
trace always lets a human verify.

---

## db.py — SQLite schema

```sql
messages        (message_id TEXT PK, timestamp TEXT, sender TEXT, text TEXT)
action_items    (id INTEGER PK, owner TEXT, task TEXT, deadline TEXT,
                 source_message_id TEXT FK → messages.message_id)
decisions       (id INTEGER PK, decision TEXT, made_by TEXT,
                 source_message_id TEXT FK → messages.message_id)
open_questions  (id INTEGER PK, asker TEXT, question TEXT, answered INTEGER,
                 source_message_id TEXT FK → messages.message_id)
```

FK enforcement requires `PRAGMA foreign_keys = ON` — this is set in `init_db()`.
Do not disable it.

---

## Testing conventions

- Never hit the real Ollama instance in tests — mock `requests.post` in
  `test_extractor.py` with pre-canned valid and invalid JSON responses
- Test the retry path explicitly by returning invalid JSON on mock call 1 and
  valid JSON on mock call 2
- Test the double-failure path by returning invalid JSON on both mock calls and
  asserting the result is an empty `ExtractionResult`
- Parser tests use the sample `.txt` string defined in `parser.py`'s own
  `__main__` block — keep that sample string in sync with test expectations

---

## Ollama model notes

| Model | Speed | Extraction quality | Recommended for |
|---|---|---|---|
| phi3:mini | Fast | Good | Default, most hardware |
| llama3.2:1b | Very fast | Weaker on schema adherence | Low-RAM machines |
| llama3.2:3b | Moderate | Better than 1b | If you have 8GB+ RAM |

Switch model via the `CHATLEDGER_MODEL` environment variable:
```bash
CHATLEDGER_MODEL=llama3.2:3b uvicorn backend.main:app
```

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| All chunks return empty ExtractionResult | Ollama not running | `ollama serve` |
| extraction_failed on every chunk | Model not pulled | `ollama pull phi3:mini` |
| Sender names duplicated with slight variations | WhatsApp name changes mid-chat | Known limitation, no fix currently |
| Action items missing deadlines | Model missed implicit dates | Prompt iteration needed |
| Open questions all marked unanswered | merge.py keyword overlap too strict | Lower threshold in `merge.py` |