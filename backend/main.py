"""ChatLedger FastAPI application.

Runs fully offline once the Ollama model is pulled: the only outbound
network call this process ever makes is to http://localhost:11434
(local Ollama). No third-party API calls are made anywhere in this app.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import asdict

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import db
from chunker import chunk_messages
from extractor import extract_chunk
from merge import merge_chunk_results
from parser import parse_whatsapp_export
from schema import ExtractionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatledger.main")

app = FastAPI(title="ChatLedger", version="1.0.0")

# CORS is local-only; the frontend is served from the same origin in
# production use, this just eases local dev across ports.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    db.init_db()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "network": "OFF — running fully local", "offline_mode": True}


@app.post("/api/upload")
async def upload_chat(file: UploadFile = File(...)) -> dict:
    """Ingest -> chunk -> extract -> merge -> store a WhatsApp export."""
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt WhatsApp exports are supported")

    raw_bytes = await file.read()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("utf-8", errors="replace")

    stages = {
        "ingestion": "pending",
        "chunking": "pending",
        "extraction": "pending",
        "storage": "pending",
    }

    # --- Ingestion ---
    messages = parse_whatsapp_export(raw_text)
    if not messages:
        raise HTTPException(status_code=400, detail="No parseable messages found in file")
    stages["ingestion"] = "done"

    # --- Chunking ---
    chunks = chunk_messages(messages)
    stages["chunking"] = "done"

    # --- Extraction ---
    chunk_results = []
    failed_chunks = 0
    for chunk in chunks:
        result = extract_chunk(chunk)
        if result is None:
            failed_chunks += 1
            continue
        chunk_results.append(result)
    stages["extraction"] = "done"

    # --- Merge ---
    merged: ExtractionResult = (
        merge_chunk_results(chunk_results) if chunk_results else ExtractionResult()
    )

    # --- Storage ---
    db.reset_db()
    db.insert_messages([asdict(m) for m in messages])
    db.insert_action_items(merged.action_items)
    db.insert_decisions(merged.decisions)
    db.insert_open_questions(merged.open_questions)
    stages["storage"] = "done"

    return {
        "stages": stages,
        "message_count": len(messages),
        "chunk_count": len(chunks),
        "failed_chunks": failed_chunks,
        "action_items": len(merged.action_items),
        "decisions": len(merged.decisions),
        "open_questions": len(merged.open_questions),
        "offline_mode": True,
    }


@app.get("/api/messages")
def list_messages() -> list[dict]:
    return db.fetch_all("messages")


@app.get("/api/action_items")
def list_action_items(owner: str | None = Query(default=None)) -> list[dict]:
    rows = db.fetch_all("action_items")
    if owner:
        rows = [r for r in rows if r["owner"].lower() == owner.lower()]
    return rows


@app.get("/api/decisions")
def list_decisions(made_by: str | None = Query(default=None)) -> list[dict]:
    rows = db.fetch_all("decisions")
    if made_by:
        rows = [r for r in rows if r["made_by"].lower() == made_by.lower()]
    return rows


@app.get("/api/open_questions")
def list_open_questions(asker: str | None = Query(default=None)) -> list[dict]:
    rows = db.fetch_all("open_questions")
    if asker:
        rows = [r for r in rows if r["asker"].lower() == asker.lower()]
    return rows


@app.get("/api/source/{message_id}")
def get_source_message(message_id: str) -> dict:
    rows = db.fetch_all("messages")
    for row in rows:
        if row["message_id"] == message_id:
            return row
    raise HTTPException(status_code=404, detail="message not found")


@app.get("/api/export")
def export_results(
    fmt: str = Query(default="json", pattern="^(json|csv)$"),
    table: str = Query(default="action_items"),
) -> StreamingResponse:
    if table not in {"action_items", "decisions", "open_questions"}:
        raise HTTPException(status_code=400, detail="invalid table")
    rows = db.fetch_all(table)

    if fmt == "json":
        buf = io.StringIO(json.dumps(rows, indent=2))
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={table}.json"},
        )

    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={table}.csv"},
    )


# Serve the single-page frontend (mounted last so /api routes take priority).
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
