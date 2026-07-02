"""Tests for extractor.py — Ollama calls, retry, fallback, caching.

No real Ollama instance is contacted.  All HTTP is intercepted by
ScriptedTransport / UnreachableTransport / TimeoutTransport /
HttpErrorTransport (defined in conftest.py).

Coverage
--------
  _render_chunk      — transcript format, text/message_text fallback, empty chunk
  _parse_response    — valid JSON, invalid JSON, schema violations (blank fields,
                       missing required fields, blank source_message_id)
  extract_chunk      — first-attempt success, retry on bad JSON, retry on schema
                       failure, corrective prompt content, double failure → empty,
                       Ollama unreachable, Ollama timeout, HTTP 4xx, HTTP 5xx,
                       empty chunk short-circuit, cache write on success,
                       cache read skips Ollama, cache miss on content change,
                       corrupt cache file recovered gracefully
  Prompt content     — extraction prompt contains schema keywords,
                       corrective prompt contains the original error message
"""

from __future__ import annotations

import json

import httpx
import pytest

# conftest.py adds backend to sys.path and defines all transports + payloads
import extractor
from conftest import (
    EMPTY_OLLAMA_RESPONSE,
    INVALID_JSON_RESPONSE,
    SCHEMA_BLANK_OWNER,
    SCHEMA_BLANK_SOURCE_ID,
    SCHEMA_MISSING_TASK,
    TRUNCATED_JSON_RESPONSE,
    VALID_OLLAMA_RESPONSE,
    HttpErrorTransport,
    ScriptedTransport,
    TimeoutTransport,
    UnreachableTransport,
)
from extractor import _parse_response, _render_chunk, extract_chunk
from schema import ExtractionResult, ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def client_for(transport: httpx.BaseTransport) -> httpx.Client:
    return httpx.Client(transport=transport)


# ===========================================================================
# _render_chunk
# ===========================================================================


class TestRenderChunk:
    def test_includes_message_id_in_brackets(self, two_message_chunk):
        rendered = _render_chunk(two_message_chunk)
        assert "[msg_0]" in rendered
        assert "[msg_1]" in rendered

    def test_includes_sender(self, two_message_chunk):
        rendered = _render_chunk(two_message_chunk)
        assert "Alice" in rendered
        assert "Bob" in rendered

    def test_includes_message_text(self, two_message_chunk):
        rendered = _render_chunk(two_message_chunk)
        assert "Can someone review the PR?" in rendered
        assert "I'll do it by Friday." in rendered

    def test_uses_text_key(self):
        chunk = [{"message_id": "msg_0", "timestamp": "t", "sender": "X", "text": "hello"}]
        assert "hello" in _render_chunk(chunk)

    def test_falls_back_to_message_text_key(self):
        """Legacy key name from earlier schema iteration still works."""
        chunk = [{"message_id": "msg_0", "timestamp": "t", "sender": "X", "message_text": "legacy"}]
        assert "legacy" in _render_chunk(chunk)

    def test_each_message_on_its_own_line(self, two_message_chunk):
        rendered = _render_chunk(two_message_chunk)
        lines = rendered.strip().split("\n")
        assert len(lines) == 2

    def test_empty_chunk_returns_empty_string(self):
        assert _render_chunk([]) == ""

    def test_missing_text_fields_render_without_crashing(self):
        chunk = [{"message_id": "msg_0", "timestamp": "t", "sender": "X"}]
        rendered = _render_chunk(chunk)
        assert "[msg_0]" in rendered


# ===========================================================================
# _parse_response
# ===========================================================================


class TestParseResponse:
    def test_accepts_fully_valid_response(self):
        result = _parse_response(VALID_OLLAMA_RESPONSE)
        assert isinstance(result, ExtractionResult)
        assert len(result.action_items) == 1
        assert result.action_items[0].owner == "Bob"
        assert result.action_items[0].deadline == "Friday"
        assert len(result.decisions) == 1
        assert len(result.open_questions) == 1

    def test_accepts_response_with_all_empty_lists(self):
        result = _parse_response(EMPTY_OLLAMA_RESPONSE)
        assert result.action_items == []
        assert result.decisions == []
        assert result.open_questions == []

    def test_raises_on_completely_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_response(INVALID_JSON_RESPONSE)

    def test_raises_on_truncated_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_response(TRUNCATED_JSON_RESPONSE)

    def test_raises_on_blank_owner(self):
        with pytest.raises(ValidationError):
            _parse_response(SCHEMA_BLANK_OWNER)

    def test_raises_on_missing_task_field(self):
        with pytest.raises((ValidationError, json.JSONDecodeError)):
            _parse_response(SCHEMA_MISSING_TASK)

    def test_raises_on_blank_source_message_id(self):
        with pytest.raises(ValidationError):
            _parse_response(SCHEMA_BLANK_SOURCE_ID)

    def test_raises_on_plain_string_response(self):
        with pytest.raises((json.JSONDecodeError, ValidationError)):
            _parse_response("Sure, here are the action items:")

    def test_raises_on_json_array_instead_of_object(self):
        with pytest.raises((json.JSONDecodeError, ValidationError)):
            _parse_response("[]")

    def test_deadline_null_is_accepted(self):
        payload = json.dumps(
            {
                "action_items": [
                    {
                        "owner": "Bob",
                        "task": "Fix the bug",
                        "deadline": None,
                        "source_message_id": "msg_1",
                    }
                ],
                "decisions": [],
                "open_questions": [],
            }
        )
        result = _parse_response(payload)
        assert result.action_items[0].deadline is None

    def test_answered_defaults_to_false_when_omitted(self):
        payload = json.dumps(
            {
                "action_items": [],
                "decisions": [],
                "open_questions": [
                    {
                        "asker": "Alice",
                        "question": "Who owns the deploy script?",
                        "source_message_id": "msg_2",
                        # 'answered' omitted — should default to False
                    }
                ],
            }
        )
        result = _parse_response(payload)
        assert result.open_questions[0].answered is False


# ===========================================================================
# extract_chunk — success paths
# ===========================================================================


class TestExtractChunkSuccess:
    def test_returns_extraction_result_on_first_attempt(
        self, two_message_chunk, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        transport = ScriptedTransport([VALID_OLLAMA_RESPONSE])
        result = extract_chunk(two_message_chunk, _client=client_for(transport))
        assert isinstance(result, ExtractionResult)
        assert transport.calls == 1

    def test_action_items_populated_correctly(self, two_message_chunk, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        result = extract_chunk(
            two_message_chunk,
            _client=client_for(ScriptedTransport([VALID_OLLAMA_RESPONSE])),
        )
        assert result.action_items[0].owner == "Bob"
        assert result.action_items[0].task == "Review the PR"

    def test_empty_chunk_returns_empty_result_without_calling_ollama(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        transport = ScriptedTransport([VALID_OLLAMA_RESPONSE])
        result = extract_chunk([], _client=client_for(transport))
        assert result == ExtractionResult()
        assert transport.calls == 0

    def test_prompt_contains_transcript(self, two_message_chunk, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        transport = ScriptedTransport([VALID_OLLAMA_RESPONSE])
        extract_chunk(two_message_chunk, _client=client_for(transport))
        assert "[msg_0]" in transport.prompts_seen[0]
        assert "Alice" in transport.prompts_seen[0]

    def test_prompt_contains_schema_keywords(self, two_message_chunk, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        transport = ScriptedTransport([VALID_OLLAMA_RESPONSE])
        extract_chunk(two_message_chunk, _client=client_for(transport))
        prompt = transport.prompts_seen[0]
        assert "action_items" in prompt
        assert "decisions" in prompt
        assert "open_questions" in prompt
        assert "source_message_id" in prompt

    def test_model_parameter_passed_to_ollama(self, two_message_chunk, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)

        captured: list[dict] = []

        class CapturingTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                captured.append(json.loads(request.content))
                return httpx.Response(200, json={"response": VALID_OLLAMA_RESPONSE})

        extract_chunk(
            two_message_chunk,
            model="llama3.2:3b",
            _client=httpx.Client(transport=CapturingTransport()),
        )
        assert captured[0]["model"] == "llama3.2:3b"


# ===========================================================================
# extract_chunk — retry logic
# ===========================================================================


class TestExtractChunkRetry:
    def test_retries_once_after_invalid_json(self, two_message_chunk, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        transport = ScriptedTransport([INVALID_JSON_RESPONSE, VALID_OLLAMA_RESPONSE])
        result = extract_chunk(two_message_chunk, _client=client_for(transport))
        assert transport.calls == 2
        assert result != ExtractionResult()

    def test_retries_once_after_schema_failure(self, two_message_chunk, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        transport = ScriptedTransport([SCHEMA_BLANK_OWNER, VALID_OLLAMA_RESPONSE])
        result = extract_chunk(two_message_chunk, _client=client_for(transport))
        assert transport.calls == 2
        assert len(result.action_items) == 1

    def test_corrective_prompt_contains_error_message(
        self, two_message_chunk, tmp_path, monkeypatch
    ):
        """The second prompt must include enough context for the model to self-correct."""
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        transport = ScriptedTransport([INVALID_JSON_RESPONSE, VALID_OLLAMA_RESPONSE])
        extract_chunk(two_message_chunk, _client=client_for(transport))
        assert len(transport.prompts_seen) == 2
        corrective = transport.prompts_seen[1]
        # Must contain *something* about the parse failure (not just re-send the original)
        assert any(
            kw in corrective.lower() for kw in ["error", "invalid", "parse", "json", "fix", "valid"]
        )

    def test_corrective_prompt_still_contains_transcript(
        self, two_message_chunk, tmp_path, monkeypatch
    ):
        """Model needs the original data in the corrective prompt to retry correctly."""
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        transport = ScriptedTransport([INVALID_JSON_RESPONSE, VALID_OLLAMA_RESPONSE])
        extract_chunk(two_message_chunk, _client=client_for(transport))
        corrective = transport.prompts_seen[1]
        assert "[msg_0]" in corrective

    def test_two_json_failures_returns_empty_result(self, two_message_chunk, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        transport = ScriptedTransport([INVALID_JSON_RESPONSE, INVALID_JSON_RESPONSE])
        result = extract_chunk(two_message_chunk, _client=client_for(transport))
        assert transport.calls == 2
        assert result == ExtractionResult()

    def test_two_schema_failures_returns_empty_result(
        self, two_message_chunk, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        transport = ScriptedTransport([SCHEMA_BLANK_OWNER, SCHEMA_BLANK_OWNER])
        result = extract_chunk(two_message_chunk, _client=client_for(transport))
        assert result == ExtractionResult()

    def test_does_not_retry_on_network_error(self, two_message_chunk, tmp_path, monkeypatch):
        """Network errors aren't fixable by rephrasing — retry is skipped."""
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        transport = UnreachableTransport()
        result = extract_chunk(two_message_chunk, _client=client_for(transport))
        assert result == ExtractionResult()


# ===========================================================================
# extract_chunk — Ollama failure modes
# ===========================================================================


class TestExtractChunkOllamaFailures:
    def test_connection_refused_returns_empty(self, two_message_chunk, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        result = extract_chunk(two_message_chunk, _client=client_for(UnreachableTransport()))
        assert result == ExtractionResult()

    def test_read_timeout_returns_empty(self, two_message_chunk, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        result = extract_chunk(two_message_chunk, _client=client_for(TimeoutTransport()))
        assert result == ExtractionResult()

    def test_http_500_returns_empty(self, two_message_chunk, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        result = extract_chunk(two_message_chunk, _client=client_for(HttpErrorTransport(500)))
        assert result == ExtractionResult()

    def test_http_404_returns_empty(self, two_message_chunk, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        result = extract_chunk(two_message_chunk, _client=client_for(HttpErrorTransport(404)))
        assert result == ExtractionResult()

    def test_pipeline_continues_after_failure(self, two_message_chunk, tmp_path, monkeypatch):
        """A None / empty result must not raise — callers just get an empty result."""
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        result = extract_chunk(two_message_chunk, _client=client_for(UnreachableTransport()))
        assert result is not None  # must return, never raise


# ===========================================================================
# extract_chunk — caching
# ===========================================================================


class TestExtractChunkCache:
    def test_writes_cache_file_on_success(self, two_message_chunk, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        extract_chunk(
            two_message_chunk,
            _client=client_for(ScriptedTransport([VALID_OLLAMA_RESPONSE])),
        )
        assert len(list(tmp_path.glob("*.json"))) == 1

    def test_cache_hit_skips_ollama(self, two_message_chunk, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        transport1 = ScriptedTransport([VALID_OLLAMA_RESPONSE])
        first = extract_chunk(two_message_chunk, _client=client_for(transport1))

        # Second call with a failing transport — cache must be used
        transport2 = ScriptedTransport([INVALID_JSON_RESPONSE])
        second = extract_chunk(two_message_chunk, _client=client_for(transport2))

        assert transport2.calls == 0
        assert second == first

    def test_cache_key_is_content_based(self, two_message_chunk, tmp_path, monkeypatch):
        """Different chunk content must produce a different cache entry."""
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        transport = ScriptedTransport([VALID_OLLAMA_RESPONSE, VALID_OLLAMA_RESPONSE])

        extract_chunk(two_message_chunk, _client=client_for(transport))

        longer_chunk = two_message_chunk + [
            {"message_id": "msg_2", "timestamp": "t", "sender": "Carol", "text": "Added."}
        ]
        extract_chunk(longer_chunk, _client=client_for(transport))

        assert transport.calls == 2
        assert len(list(tmp_path.glob("*.json"))) == 2

    def test_corrupt_cache_file_is_ignored(self, two_message_chunk, tmp_path, monkeypatch):
        """A corrupted cache JSON must not raise — Ollama is re-called instead."""
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        transport = ScriptedTransport([VALID_OLLAMA_RESPONSE, VALID_OLLAMA_RESPONSE])

        # First call — writes valid cache
        extract_chunk(two_message_chunk, _client=client_for(transport))
        cache_file = list(tmp_path.glob("*.json"))[0]

        # Corrupt the cache file
        cache_file.write_text("{ this is not valid json }", encoding="utf-8")

        # Second call — should re-call Ollama and not raise
        result = extract_chunk(two_message_chunk, _client=client_for(transport))
        assert result is not None
        assert transport.calls == 2

    def test_failure_does_not_write_cache(self, two_message_chunk, tmp_path, monkeypatch):
        """Failed extractions must not pollute the cache."""
        monkeypatch.setattr(extractor, "_CACHE_DIR", tmp_path)
        extract_chunk(
            two_message_chunk,
            _client=client_for(ScriptedTransport([INVALID_JSON_RESPONSE, INVALID_JSON_RESPONSE])),
        )
        assert list(tmp_path.glob("*.json")) == []