"""main.py — FastAPI application for ChatLedger.

Wires together the full pipeline:
    parser  → chunker  → extractor  → merge  → db

Start the server
----------------
    # from the /backend directory:
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload

Endpoints
---------
    POST /upload                        run the full pipeline on a .txt file
    GET  /results                       all extracted items from the db
    GET  /results/export?format=csv|json  download results
    GET  /health                        liveness + Ollama reachability check
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from chunker import chunk_messages
from db import (
    fetch_action_items,
    fetch_decisions,
    fetch_open_questions,
    fetch_source_message,
    init_db,
    insert_extraction_result,
    insert_messages,
)
from extractor import OLLAMA_URL, extract_chunk
from merge import merge_results
from parser import _parse_raw
from schema import ExtractionResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("chatledger.main")

# ---------------------------------------------------------------------------
# Database path  (sits next to this file, created on first startup)
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent / "chatledger.db"

# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    init_db(DB_PATH)
    log.info("Database ready at %s", DB_PATH)
    yield


app = FastAPI(
    title="ChatLedger",
    description="Offline chat-log analyser powered by a local Ollama model.",
    version="1.0.0",
    lifespan=_lifespan,
)

# Allow any localhost origin so the plain-HTML frontend can call the API
# from file://, http://localhost:*, or http://127.0.0.1:* during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ollama_reachable() -> bool:
    """Return True if the local Ollama server responds within 2 seconds."""
    try:
        # Hit the tags endpoint (light, no model needed) rather than /generate
        base = OLLAMA_URL.rsplit("/api/", 1)[0]
        httpx.get(f"{base}/api/tags", timeout=2.0)
        return True
    except httpx.HTTPError:
        return False


def _rows_to_csv(rows: list[dict]) -> str:
    """Serialise a list of dicts to a CSV string."""
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------


@app.post("/upload", status_code=status.HTTP_200_OK)
async def upload(
    file: Annotated[UploadFile, File(description="WhatsApp-exported .txt chat log")],
) -> dict:
    """Run the full extraction pipeline on an uploaded .txt file.

    Pipeline stages
    ---------------
    1. Parse   — regex-split the WhatsApp export into message rows
    2. Chunk   — split into overlapping 15-message windows
    3. Extract — call the local Ollama model once per chunk
    4. Merge   — dedup action items / decisions / open questions
    5. Store   — persist to SQLite

    Returns a summary dict with item counts and any failed chunk IDs.
    """
    # ── Validate file type ─────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only WhatsApp .txt exports are accepted.",
        )

    raw_bytes = await file.read()
    try:
        raw_text = raw_bytes.decode("utf-8-sig")  # strip BOM if present
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin-1")  # last-resort fallback

    t_start = time.perf_counter()

    # ── Stage 1: Parse ────────────────────────────────────────────────
    messages = _parse_raw(raw_text)
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No parseable messages found. Is this a WhatsApp .txt export?",
        )
    log.info("Parsed %d messages from %s", len(messages), file.filename)

    # ── Stage 2: Chunk ────────────────────────────────────────────────
    chunks = chunk_messages(messages)
    log.info("Created %d chunks", len(chunks))

    # ── Stage 3: Extract (one Ollama call per chunk) ──────────────────
    chunk_results: list[ExtractionResult] = []
    failed_chunks: list[str] = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"
        result = extract_chunk(chunk)
        if result is None or result == ExtractionResult():
            # extract_chunk already logged the failure internally
            failed_chunks.append(chunk_id)
            log.warning("Extraction yielded nothing for %s", chunk_id)
        else:
            chunk_results.append(result)

    log.info(
        "Extraction complete: %d/%d chunks succeeded",
        len(chunk_results),
        len(chunks),
    )

    # ── Stage 4: Merge ────────────────────────────────────────────────
    merged = merge_results(chunk_results) if chunk_results else ExtractionResult()

    # ── Stage 5: Store ────────────────────────────────────────────────
    insert_messages(DB_PATH, messages)
    insert_extraction_result(DB_PATH, merged)

    elapsed = round(time.perf_counter() - t_start, 2)
    log.info("Pipeline complete in %.2fs", elapsed)

    return {
        "status": "ok",
        "filename": file.filename,
        "elapsed_seconds": elapsed,
        "pipeline": {
            "messages_parsed": len(messages),
            "chunks_total": len(chunks),
            "chunks_succeeded": len(chunk_results),
            "chunks_failed": len(failed_chunks),
            "failed_chunk_ids": failed_chunks,
        },
        "extracted": {
            "action_items": len(merged.action_items),
            "decisions": len(merged.decisions),
            "open_questions": len(merged.open_questions),
        },
    }


# ---------------------------------------------------------------------------
# GET /results
# ---------------------------------------------------------------------------


@app.get("/results")
def get_results(
    owner: Annotated[str | None, Query(description="Filter action_items by owner")] = None,
    made_by: Annotated[str | None, Query(description="Filter decisions by made_by")] = None,
    asker: Annotated[str | None, Query(description="Filter open_questions by asker")] = None,
    source_message_id: Annotated[
        str | None,
        Query(description="Return the source message for this id instead of all results"),
    ] = None,
) -> dict:
    """Return all extracted items from the database.

    Optional query params filter each category independently.
    ``source_message_id`` returns the raw message row for click-to-reveal.
    """
    if source_message_id is not None:
        msg = fetch_source_message(DB_PATH, source_message_id)
        if msg is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No message with id {source_message_id!r}",
            )
        return {"source_message": msg}

    return {
        "action_items": fetch_action_items(DB_PATH, owner=owner),
        "decisions": fetch_decisions(DB_PATH, made_by=made_by),
        "open_questions": fetch_open_questions(DB_PATH, asker=asker),
    }


# ---------------------------------------------------------------------------
# GET /results/export
# ---------------------------------------------------------------------------


_EXPORT_TABLES = {
    "action_items": fetch_action_items,
    "decisions": fetch_decisions,
    "open_questions": fetch_open_questions,
}


@app.get("/results/export")
def export_results(
    format: Annotated[  # noqa: A002  (shadows built-in, but it's a query param name)
        str,
        Query(description="Export format: 'json' or 'csv'"),
    ] = "json",
    table: Annotated[
        str,
        Query(description="Which table to export: action_items | decisions | open_questions"),
    ] = "action_items",
) -> StreamingResponse:
    """Download extracted items as JSON or CSV.

    Examples
    --------
        GET /results/export?format=csv&table=action_items
        GET /results/export?format=json&table=decisions
    """
    fmt = format.lower().strip()
    tbl = table.lower().strip()

    if fmt not in {"json", "csv"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported format {fmt!r}. Use 'json' or 'csv'.",
        )
    if tbl not in _EXPORT_TABLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown table {tbl!r}. Choose from: {', '.join(_EXPORT_TABLES)}.",
        )

    rows = _EXPORT_TABLES[tbl](DB_PATH)
    filename = f"chatledger_{tbl}.{fmt}"

    if fmt == "json":
        body = json.dumps(rows, indent=2, ensure_ascii=False)
        return StreamingResponse(
            iter([body]),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # CSV
    body = _rows_to_csv(rows)
    return StreamingResponse(
        iter([body]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Liveness check.  Also probes Ollama so the UI can show a warning
    if the model server isn't running before the user uploads a file."""
    ollama_ok = _ollama_reachable()
    return {
        "status": "ok",
        "database": str(DB_PATH),
        "ollama": {
            "url": OLLAMA_URL,
            "reachable": ollama_ok,
            "note": (
                "Ready"
                if ollama_ok
                else "Not reachable — start Ollama and run `ollama pull phi3:mini`"
            ),
        },
        "offline_mode": True,
        "network_note": "Only localhost:11434 (Ollama) is ever contacted — no cloud calls.",
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Start the server:
    #   python main.py
    # or, with auto-reload for development:
    #   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)