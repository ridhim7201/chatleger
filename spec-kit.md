# SPEC-KIT.md — ChatLedger

The full technical specification: what this system does, the exact contracts
between every component, the data schemas, the API surface, and the acceptance
criteria for each feature. Use this as the ground truth when building,
reviewing, or evaluating any part of the project.

---

## System overview

ChatLedger is a local-first web application that takes a WhatsApp-exported
`.txt` chat log as input and produces a structured extraction of three
information types: action items, decisions, and open questions. All processing
runs offline using a locally-served Ollama model. Results are persisted in
SQLite and served through a FastAPI backend to a single-page HTML frontend.

---

## Functional requirements

### FR-1: File ingestion
- Accept WhatsApp `.txt` export files via HTTP file upload
- Parse into structured message rows: `(message_id, timestamp, sender, text)`
- Assign sequential message IDs: `msg_0`, `msg_1`, ...
- Preserve multi-line messages as a single row (continuation lines appended to parent)
- Identify and tag system messages as `sender: "SYSTEM"`
- Support files up to 50,000 messages without crashing or timeout

### FR-2: Chunking
- Split message rows into overlapping windows
- Default: 15 messages per chunk, 3-message overlap
- Overlap must ensure questions at chunk N boundary can be resolved by chunk N+1
- Chunk boundaries logged and returned in the pipeline summary

### FR-3: LLM extraction
- Each chunk sent to Ollama as a single prompt
- Model must be configurable via `CHATLEDGER_MODEL` env var (default: `phi3:mini`)
- Output validated against `ExtractionResult` Pydantic schema before use
- Invalid output triggers one retry with corrective prompt
- Double failure returns empty `ExtractionResult` and logs a warning
- Successful results cached by chunk content hash to avoid redundant inference

### FR-4: Merging
- Combine per-chunk `ExtractionResult` objects into one final result
- Deduplicate action items and decisions appearing in overlapping chunks (similarity ≥ 0.85)
- Resolve open questions as `answered=True` if later chunk implies a reply
- Dedup and resolution logic must be deterministic given the same inputs

### FR-5: Storage
- Store all parsed messages and extracted items in SQLite
- Foreign keys from extracted items to source `message_id` enforced at DB level
- DB initialized on first run; subsequent runs append to existing DB per chat session

### FR-6: Results API
- Return all extracted items as JSON via `GET /results`
- Support filtering by `sender`/`owner` via query params
- Support export as CSV or JSON via `GET /results/export?format=csv|json`

### FR-7: Offline resiliency
- Zero external HTTP calls during any operation
- Frontend makes no CDN requests
- `GET /health` reports Ollama reachability at `localhost:11434`
- UI displays a visible offline-mode indicator at all times

---

## Data schemas

### Message row (in-memory, pre-DB)

```python
{
  "message_id": str,     # "msg_0", "msg_1", ...
  "timestamp": str,      # original timestamp string from export
  "sender": str,         # sender name as-is from export, or "SYSTEM"
  "text": str            # full message text, multi-line concatenated
}
```

### ExtractionResult (Pydantic)

```python
class ActionItem(BaseModel):
    owner: str                        # required
    task: str                         # required
    deadline: Optional[str] = None    # null if no date mentioned
    source_message_id: str            # required, must match a msg_id

class Decision(BaseModel):
    decision: str                     # required
    made_by: Optional[str] = None     # null if group decision / unclear
    source_message_id: str            # required

class OpenQuestion(BaseModel):
    asker: str                        # required
    question: str                     # required
    answered: bool = False            # resolved during merge phase
    source_message_id: str            # required

class ExtractionResult(BaseModel):
    action_items: List[ActionItem] = []
    decisions: List[Decision] = []
    open_questions: List[OpenQuestion] = []
```

### SQLite tables

```sql
CREATE TABLE messages (
    message_id   TEXT PRIMARY KEY,
    timestamp    TEXT NOT NULL,
    sender       TEXT NOT NULL,
    text         TEXT NOT NULL
);

CREATE TABLE action_items (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    owner              TEXT NOT NULL,
    task               TEXT NOT NULL,
    deadline           TEXT,
    source_message_id  TEXT NOT NULL,
    FOREIGN KEY (source_message_id) REFERENCES messages(message_id)
);

CREATE TABLE decisions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    decision           TEXT NOT NULL,
    made_by            TEXT,
    source_message_id  TEXT NOT NULL,
    FOREIGN KEY (source_message_id) REFERENCES messages(message_id)
);

CREATE TABLE open_questions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    asker              TEXT NOT NULL,
    question           TEXT NOT NULL,
    answered           INTEGER NOT NULL DEFAULT 0,   -- 0 = false, 1 = true
    source_message_id  TEXT NOT NULL,
    FOREIGN KEY (source_message_id) REFERENCES messages(message_id)
);
```

---

## API specification

### POST /upload

**Request:** `multipart/form-data`, field `file`, `.txt` only

**Response 200:**
```json
{
  "status": "ok",
  "messages_parsed": 312,
  "chunks_processed": 22,
  "chunks_failed": 1,
  "action_items_found": 14,
  "decisions_found": 6,
  "open_questions_found": 9
}
```

**Response 422:** file is not `.txt` or is empty
**Response 500:** pipeline crash with error detail

### GET /results

**Query params:**
- `owner` (optional) — filter action_items by owner name
- `sender` (optional) — filter all results by original sender name

**Response 200:**
```json
{
  "action_items": [...],
  "decisions": [...],
  "open_questions": [...]
}
```

### GET /results/export

**Query params:**
- `format` — `"csv"` or `"json"` (required)

**Response 200:** file download, Content-Disposition attachment

**Response 400:** invalid or missing `format` param

### GET /health

**Response 200:**
```json
{
  "status": "ok",
  "ollama_reachable": true,
  "model": "phi3:mini"
}
```

**Response 200 (degraded):**
```json
{
  "status": "degraded",
  "ollama_reachable": false,
  "model": "phi3:mini"
}
```
(HTTP 200 either way — health is informational, not a hard failure gate)

---

## Extraction prompt contract

The prompt sent to Ollama for each chunk must:

1. Instruct the model to output raw JSON only (no preamble, no markdown fences)
2. Define all three categories with field-level descriptions
3. Include the `source_message_id` requirement explicitly
4. State precision > recall explicitly (uncertain items excluded, not guessed)
5. Define what qualifies as an action item vs a vague intention
6. Define what qualifies as a decision vs a casual opinion
7. Define the open question scoping rule (unanswered within this chunk only)

The prompt must not:
- Ask for any output other than a single JSON object
- Ask the model to explain its reasoning
- Use few-shot examples longer than 3 lines (increases token cost per chunk significantly)

---

## Ollama call contract

```
POST http://localhost:11434/api/generate
Content-Type: application/json

{
  "model": "<CHATLEDGER_MODEL>",
  "prompt": "<assembled prompt with chunk text>",
  "stream": false,
  "format": "json",
  "options": {
    "temperature": 0.1,
    "num_ctx": 4096
  }
}
```

Response field used: `response.json()["response"]` — a string parsed as JSON.

---

## Caching contract

- Cache location: `.cache/chatledger/` (relative to project root)
- Cache key: `sha256(json.dumps(chunk_messages, sort_keys=True))`
- Cache file: `{hash}.json` containing a serialized `ExtractionResult`
- Cache is never invalidated automatically — delete files manually to bust
- Cache is read before the Ollama call, written only on successful validation

---

## CI pipeline specification

All jobs must genuinely pass or fail based on code state. No stubs.

| Job | Tool | Failure condition |
|---|---|---|
| format | `black --check .` | Any file not conforming to black formatting |
| lint | `ruff check .` | Any lint violation not suppressed with justification |
| typecheck | `mypy backend/` | Any type error |
| security | `bandit -r backend/` | Any HIGH severity finding |
| test | `pytest tests/ -v` | Any test failure or uncaught exception |
| commit-lint | conventional-commits action | Any commit message not matching format |

---

## Acceptance criteria

### AC-1: Parser
- Correctly parses the sample 15-message export provided in `parser.py`
- Multi-line messages produce one row, not two
- System messages produce `sender: "SYSTEM"` rows
- Messages with colons in body text are not incorrectly split

### AC-2: Chunker
- 20-message input with `window=15, overlap=3` produces exactly 2 chunks
- Chunk 1 has messages 0–14, Chunk 2 has messages 12–19 (3-message overlap)

### AC-3: Extractor
- Valid Ollama response produces a valid `ExtractionResult`
- Invalid JSON on first call → retry fired, second valid response accepted
- Invalid JSON on both calls → empty `ExtractionResult` returned, no exception raised
- Repeated call on same chunk → cache hit, no Ollama call made

### AC-4: Merge
- Two chunks with 3 overlapping identical action items produce 1, not 2
- Question in chunk N with decision in chunk N+1 referencing same topic → `answered=True`

### AC-5: API
- `POST /upload` with a valid .txt returns 200 with counts > 0
- `POST /upload` with a non-.txt file returns 422
- `GET /results` returns all three categories
- `GET /health` returns `ollama_reachable: false` when Ollama is not running

### AC-6: Offline
- All of the above work with no internet connection
- `index.html` loads and functions with network disabled
- No requests to any host other than `localhost`

---

## Out of scope (explicitly not built)

- Multi-file / batch upload
- WhatsApp media attachment parsing
- Slack JSON export support (referenced in concept but not implemented)
- User authentication
- Cloud deployment
- Real-time / streaming chat ingestion
- Mobile app