# ChatLedger

> Turn raw WhatsApp chat exports into a structured ledger of action items, decisions, and open questions — fully offline, powered by a local Ollama model.

---

## What it does

ChatLedger parses a WhatsApp-exported `.txt` chat log and extracts three things from it:

- **Action items** — who committed to doing what, and by when
- **Decisions** — what was resolved or agreed on
- **Open questions** — things asked but never answered

Results are stored in a local SQLite database and browsable through a single-page web UI. No data leaves your machine. No internet connection required after initial setup.

---

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, FastAPI |
| LLM | Ollama (phi3:mini or llama3.2:3b, local) |
| Database | SQLite |
| Frontend | Vanilla HTML/JS, no build step |
| Tests | pytest, mypy, ruff, black, bandit |
| CI | GitHub Actions |

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Git

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourname/chatledger.git
cd chatledger
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Pull the model and start Ollama

```bash
ollama pull phi3:mini
ollama serve                    # runs at http://localhost:11434
```

Verify it's working:

```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"phi3:mini","prompt":"say hello","stream":false}'
```

### 4. Start the backend

```bash
uvicorn backend.main:app --reload
```

API available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

### 5. Open the frontend

```bash
open frontend/index.html
# or: python -m http.server 3000 --directory frontend
```

---

## Project structure

```
/chatledger
  /backend
    main.py         # FastAPI app, endpoints
    parser.py       # .txt ingestion and message-row extraction
    chunker.py      # sliding window chunking with overlap
    extractor.py    # Ollama calls, retry logic, result caching
    merge.py        # chunk-level dedup and merge
    db.py           # SQLite schema and insert/query functions
    schema.py       # Pydantic models for extraction output
  /frontend
    index.html      # single-file UI, no build step
  /tests
    test_parser.py
    test_extractor.py
    test_merge.py
  .github/
    workflows/
      ci.yml        # lint, type-check, security scan, tests
  requirements.txt
  README.md
  CONTRIBUTING.md
  AGENTS.md
  SKILL.md
  SPEC-KIT.md
```

---

## Usage

1. Export a WhatsApp chat: **Chat → ⋮ → More → Export chat → Without media**
2. Upload the `.txt` file via the web UI
3. Wait for the pipeline to run (progress shown per stage)
4. Browse Action Items, Decisions, and Open Questions in the results tabs
5. Filter by sender/owner, or export as CSV or JSON

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload .txt file, run full pipeline |
| GET | `/results` | Return all extracted results as JSON |
| GET | `/results/export?format=csv\|json` | Download results |
| GET | `/health` | Backend status + Ollama reachability check |

---

## Running tests

```bash
pytest tests/ -v
```

---

## Offline resiliency

ChatLedger makes zero external network calls during operation. All inference runs through `localhost:11434` (Ollama). The frontend makes no CDN requests. The only time you need internet is for the one-time `ollama pull phi3:mini` during setup.

The UI displays a **"Network: OFF — running fully local"** badge to confirm this.

---

## License

MIT