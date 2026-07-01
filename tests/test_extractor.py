import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import extractor  # noqa: E402
from chunker import Chunk  # noqa: E402
from parser import ParsedMessage  # noqa: E402
from schema import ExtractionResult, ValidationError  # noqa: E402


def make_chunk() -> Chunk:
    msgs = [
        ParsedMessage("msg_0", "01/02/24 10:00:00", "Alice", "Can someone review the PR?"),
        ParsedMessage("msg_1", "01/02/24 10:01:00", "Bob", "I'll do it by Friday"),
    ]
    return Chunk(chunk_id="chunk_0", messages=msgs, chunk_hash="testhash123")


VALID_RESPONSE = json.dumps(
    {
        "action_items": [
            {
                "owner": "Bob",
                "task": "Review the PR",
                "deadline": "Friday",
                "source_message_id": "msg_1",
            }
        ],
        "decisions": [],
        "open_questions": [],
    }
)

INVALID_JSON_RESPONSE = "this is not json at all {{"

SCHEMA_INVALID_RESPONSE = json.dumps(
    {
        "action_items": [{"owner": "", "task": "Review the PR", "source_message_id": "msg_1"}],
        "decisions": [],
        "open_questions": [],
    }
)


class FakeOllamaTransport(httpx.BaseTransport):
    """A fake transport returning a scripted sequence of responses."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        idx = min(self.calls, len(self.responses) - 1)
        body = self.responses[idx]
        self.calls += 1
        return httpx.Response(200, json={"response": body})


class FailingTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)


def test_parse_and_validate_accepts_valid_response():
    result = extractor._parse_and_validate(VALID_RESPONSE)
    assert isinstance(result, ExtractionResult)
    assert len(result.action_items) == 1
    assert result.action_items[0].owner == "Bob"


def test_parse_and_validate_rejects_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        extractor._parse_and_validate(INVALID_JSON_RESPONSE)


def test_parse_and_validate_rejects_schema_violation():
    with pytest.raises(ValidationError):
        extractor._parse_and_validate(SCHEMA_INVALID_RESPONSE)


def test_extract_chunk_succeeds_on_first_try(monkeypatch):
    chunk = make_chunk()
    transport = FakeOllamaTransport([VALID_RESPONSE])
    client = httpx.Client(transport=transport)

    monkeypatch.setattr(extractor, "get_cached_result", lambda h: None)
    monkeypatch.setattr(extractor, "set_cached_result", lambda h, r: None)

    result = extractor.extract_chunk(chunk, client=client, use_cache=True)
    assert result is not None
    assert transport.calls == 1
    assert result.action_items[0].owner == "Bob"


def test_extract_chunk_retries_once_then_succeeds(monkeypatch):
    chunk = make_chunk()
    transport = FakeOllamaTransport([INVALID_JSON_RESPONSE, VALID_RESPONSE])
    client = httpx.Client(transport=transport)

    monkeypatch.setattr(extractor, "get_cached_result", lambda h: None)
    monkeypatch.setattr(extractor, "set_cached_result", lambda h, r: None)

    result = extractor.extract_chunk(chunk, client=client, use_cache=True)
    assert result is not None
    assert transport.calls == 2


def test_extract_chunk_logs_failure_and_returns_none_after_two_failures(monkeypatch):
    chunk = make_chunk()
    transport = FakeOllamaTransport([INVALID_JSON_RESPONSE, INVALID_JSON_RESPONSE])
    client = httpx.Client(transport=transport)

    logged: dict[str, Any] = {}

    def fake_log_failure(chunk_id, chunk_hash, error):
        logged["chunk_id"] = chunk_id
        logged["chunk_hash"] = chunk_hash
        logged["error"] = error

    monkeypatch.setattr(extractor, "get_cached_result", lambda h: None)
    monkeypatch.setattr(extractor, "set_cached_result", lambda h, r: None)
    monkeypatch.setattr(extractor, "log_extraction_failure", fake_log_failure)

    result = extractor.extract_chunk(chunk, client=client, use_cache=True)
    assert result is None
    assert transport.calls == 2
    assert logged["chunk_id"] == "chunk_0"
    assert logged["chunk_hash"] == "testhash123"


def test_extract_chunk_handles_ollama_connection_failure_gracefully(monkeypatch):
    """Pipeline must not crash if Ollama is unreachable; it should log and
    return None so the caller can continue processing other chunks."""
    chunk = make_chunk()
    client = httpx.Client(transport=FailingTransport())

    logged: dict[str, Any] = {}
    monkeypatch.setattr(extractor, "get_cached_result", lambda h: None)
    monkeypatch.setattr(extractor, "set_cached_result", lambda h, r: None)
    monkeypatch.setattr(
        extractor,
        "log_extraction_failure",
        lambda chunk_id, chunk_hash, error: logged.update(chunk_id=chunk_id, error=error),
    )

    result = extractor.extract_chunk(chunk, client=client, use_cache=True)
    assert result is None
    assert logged["chunk_id"] == "chunk_0"


def test_extract_chunk_uses_cache_and_skips_ollama_call(monkeypatch):
    chunk = make_chunk()
    transport = FakeOllamaTransport([VALID_RESPONSE])
    client = httpx.Client(transport=transport)

    cached_payload = json.loads(VALID_RESPONSE)
    monkeypatch.setattr(extractor, "get_cached_result", lambda h: cached_payload)

    result = extractor.extract_chunk(chunk, client=client, use_cache=True)
    assert result is not None
    assert transport.calls == 0  # never called Ollama, served from cache
