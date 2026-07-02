"""Integration tests for main.py FastAPI routes.

Uses FastAPI's TestClient (synchronous httpx wrapper).  The tests patch:
  - extract_chunk    → returns a canned ExtractionResult (no Ollama needed)
  - DB_PATH in main  → tmp_path per test (no shared state)

This exercises the full HTTP layer and pipeline wiring without any I/O to
a real model or a shared database file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from schema import ActionItem, Decision, ExtractionResult, OpenQuestion  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal valid WhatsApp export covering 3 messages
SAMPLE_TXT = (
    "[01/06/24, 09:00:00] Alice: Can we confirm the release date?\n"
    "[01/06/24, 09:01:00] Bob: June 15th — that was the decision.\n"
    "[01/06/24, 09:02:00] Carol: Who is writing the blog post?\n"
)

CANNED_RESULT = ExtractionResult(
    action_items=[
        ActionItem(owner="Bob", task="Confirm June 15th release", source_message_id="msg_1"),
    ],
    decisions=[
        Decision(decision="Release date is June 15th", made_by="Bob", source_message_id="msg_1"),
    ],
    open_questions=[
        OpenQuestion(
            asker="Carol",
            question="Who is writing the blog post?",
            answered=False,
            source_message_id="msg_2",
        ),
    ],
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Fresh TestClient with isolated tmp DB and mocked extractor per test."""
    import main as main_module

    # Redirect the database to a temp file so tests don't share state
    monkeypatch.setattr(main_module, "DB_PATH", tmp_path / "test.db")

    # Replace extract_chunk with a function that always succeeds instantly
    monkeypatch.setattr(main_module, "extract_chunk", lambda chunk, **kw: CANNED_RESULT)

    # Re-initialise the DB at the new path (simulates startup event)
    from db import init_db

    init_db(tmp_path / "test.db")

    return TestClient(main_module.app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["offline_mode"] is True
    assert "ollama" in body


def test_health_ollama_key_present(client):
    r = client.get("/health")
    ollama = r.json()["ollama"]
    assert "reachable" in ollama
    assert "url" in ollama


# ---------------------------------------------------------------------------
# /upload — success paths
# ---------------------------------------------------------------------------


def _upload(client, content: str = SAMPLE_TXT, filename: str = "chat.txt"):
    return client.post(
        "/upload",
        files={"file": (filename, content.encode(), "text/plain")},
    )


def test_upload_returns_200_with_summary(client):
    r = _upload(client)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["pipeline"]["messages_parsed"] == 3
    assert body["extracted"]["action_items"] == 1
    assert body["extracted"]["decisions"] == 1
    assert body["extracted"]["open_questions"] == 1


def test_upload_summary_contains_pipeline_stats(client):
    r = _upload(client)
    p = r.json()["pipeline"]
    assert "chunks_total" in p
    assert "chunks_succeeded" in p
    assert "chunks_failed" in p
    assert "failed_chunk_ids" in p
    assert isinstance(p["failed_chunk_ids"], list)


def test_upload_elapsed_seconds_present(client):
    r = _upload(client)
    assert "elapsed_seconds" in r.json()


# ---------------------------------------------------------------------------
# /upload — failure paths
# ---------------------------------------------------------------------------


def test_upload_rejects_non_txt_extension(client):
    r = client.post(
        "/upload",
        files={"file": ("chat.pdf", b"not a txt", "application/pdf")},
    )
    assert r.status_code == 422


def test_upload_rejects_unparseable_content(client):
    r = _upload(client, content="\n\n\n\n")
    assert r.status_code == 422


def test_upload_handles_utf8_bom(client):
    bom_content = "\ufeff" + SAMPLE_TXT
    r = _upload(client, content=bom_content)
    assert r.status_code == 200
    assert r.json()["pipeline"]["messages_parsed"] == 3


# ---------------------------------------------------------------------------
# /results — after a successful upload
# ---------------------------------------------------------------------------


@pytest.fixture()
def populated_client(client):
    _upload(client)
    return client


def test_results_returns_all_three_categories(populated_client):
    r = populated_client.get("/results")
    assert r.status_code == 200
    body = r.json()
    assert "action_items" in body
    assert "decisions" in body
    assert "open_questions" in body


def test_results_action_items_correct(populated_client):
    items = populated_client.get("/results").json()["action_items"]
    assert len(items) == 1
    assert items[0]["owner"] == "Bob"
    assert items[0]["task"] == "Confirm June 15th release"


def test_results_decisions_correct(populated_client):
    decs = populated_client.get("/results").json()["decisions"]
    assert len(decs) == 1
    assert "June 15th" in decs[0]["decision"]


def test_results_open_questions_correct(populated_client):
    qs = populated_client.get("/results").json()["open_questions"]
    assert len(qs) == 1
    assert qs[0]["asker"] == "Carol"
    assert qs[0]["answered"] == 0  # stored as int in sqlite


def test_results_filter_by_owner(populated_client):
    r = populated_client.get("/results?owner=Bob")
    assert r.status_code == 200
    items = r.json()["action_items"]
    assert all(i["owner"] == "Bob" for i in items)


def test_results_filter_by_owner_no_match(populated_client):
    r = populated_client.get("/results?owner=Nobody")
    assert r.status_code == 200
    assert r.json()["action_items"] == []


def test_results_source_message_lookup(populated_client):
    r = populated_client.get("/results?source_message_id=msg_1")
    assert r.status_code == 200
    msg = r.json()["source_message"]
    assert msg["message_id"] == "msg_1"
    assert msg["sender"] == "Bob"


def test_results_source_message_missing_returns_404(populated_client):
    r = populated_client.get("/results?source_message_id=msg_999")
    assert r.status_code == 404


def test_results_empty_db_returns_empty_lists(client):
    r = client.get("/results")
    assert r.status_code == 200
    body = r.json()
    assert body["action_items"] == []
    assert body["decisions"] == []
    assert body["open_questions"] == []


# ---------------------------------------------------------------------------
# /results/export
# ---------------------------------------------------------------------------


def test_export_json_action_items(populated_client):
    r = populated_client.get("/results/export?format=json&table=action_items")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert "attachment" in r.headers["content-disposition"]
    data = json.loads(r.content)
    assert isinstance(data, list)
    assert data[0]["owner"] == "Bob"


def test_export_csv_action_items(populated_client):
    r = populated_client.get("/results/export?format=csv&table=action_items")
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert "attachment" in r.headers["content-disposition"]
    text = r.text
    assert "owner" in text  # header row
    assert "Bob" in text  # data row


def test_export_json_decisions(populated_client):
    r = populated_client.get("/results/export?format=json&table=decisions")
    assert r.status_code == 200
    data = json.loads(r.content)
    assert data[0]["made_by"] == "Bob"


def test_export_csv_open_questions(populated_client):
    r = populated_client.get("/results/export?format=csv&table=open_questions")
    assert r.status_code == 200
    assert "Carol" in r.text


def test_export_defaults_to_json_action_items(populated_client):
    r = populated_client.get("/results/export")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")


def test_export_invalid_format_returns_422(populated_client):
    r = populated_client.get("/results/export?format=xml")
    assert r.status_code == 422


def test_export_invalid_table_returns_422(populated_client):
    r = populated_client.get("/results/export?table=messages")
    assert r.status_code == 422


def test_export_empty_db_returns_empty_json(client):
    r = client.get("/results/export?format=json&table=action_items")
    assert r.status_code == 200
    assert json.loads(r.content) == []


def test_export_filename_in_content_disposition(populated_client):
    r = populated_client.get("/results/export?format=csv&table=decisions")
    assert "chatledger_decisions.csv" in r.headers["content-disposition"]