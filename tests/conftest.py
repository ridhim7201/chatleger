"""Shared pytest fixtures for ChatLedger test suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

# Make backend importable from all test modules without repeating sys.path boilerplate
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from schema import ActionItem, Decision, ExtractionResult, OpenQuestion  # noqa: E402

# ---------------------------------------------------------------------------
# JSON payloads reused across extractor tests
# ---------------------------------------------------------------------------

VALID_OLLAMA_RESPONSE = json.dumps(
    {
        "action_items": [
            {
                "owner": "Bob",
                "task": "Review the PR",
                "deadline": "Friday",
                "source_message_id": "msg_1",
            }
        ],
        "decisions": [
            {
                "decision": "Release on June 15th",
                "made_by": "Alice",
                "source_message_id": "msg_2",
            }
        ],
        "open_questions": [
            {
                "asker": "Carol",
                "question": "Who writes the blog post?",
                "answered": False,
                "source_message_id": "msg_3",
            }
        ],
    }
)

EMPTY_OLLAMA_RESPONSE = json.dumps({"action_items": [], "decisions": [], "open_questions": []})

INVALID_JSON_RESPONSE = "this is { definitely not json ]["

TRUNCATED_JSON_RESPONSE = '{"action_items": [{"owner": "Bob"'  # cut off mid-object

SCHEMA_BLANK_OWNER = json.dumps(
    {
        "action_items": [{"owner": "", "task": "Review the PR", "source_message_id": "msg_1"}],
        "decisions": [],
        "open_questions": [],
    }
)

SCHEMA_MISSING_TASK = json.dumps(
    {
        "action_items": [{"owner": "Bob", "source_message_id": "msg_1"}],
        "decisions": [],
        "open_questions": [],
    }
)

SCHEMA_BLANK_SOURCE_ID = json.dumps(
    {
        "action_items": [{"owner": "Bob", "task": "Do something", "source_message_id": ""}],
        "decisions": [],
        "open_questions": [],
    }
)


# ---------------------------------------------------------------------------
# Message dict factory
# ---------------------------------------------------------------------------


def make_message(
    idx: int,
    sender: str = "Alice",
    text: str = "A message.",
    ts: str | None = None,
) -> dict:
    return {
        "message_id": f"msg_{idx}",
        "timestamp": ts or f"01/06/24, 09:{idx:02d}:00",
        "sender": sender,
        "text": text,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_message_chunk() -> list[dict]:
    """Minimal 2-message chunk used by extractor tests."""
    return [
        make_message(0, "Alice", "Can someone review the PR?"),
        make_message(1, "Bob", "I'll do it by Friday."),
    ]


@pytest.fixture()
def ten_message_chunk() -> list[dict]:
    """Realistic 10-message chunk covering all extraction categories."""
    senders = ["Alice", "Bob", "Carol"]
    texts = [
        "Morning everyone, ready for the standup?",
        "Morning! Let's go.",
        "Sure — I finished the auth module yesterday.",
        "Alice, can you update the changelog before EOD?",
        "Sure, done by 5 pm.",
        "What's the status on the deploy key rotation?",
        "Still pending — I'll handle it today.",
        "Decision: release date is June 15th.",
        "Who is writing the blog post announcement?",
        "That's still open, no one assigned.",
    ]
    return [make_message(i, senders[i % 3], texts[i]) for i in range(10)]


@pytest.fixture()
def full_extraction_result() -> ExtractionResult:
    """An ExtractionResult with one of each item type."""
    return ExtractionResult(
        action_items=[
            ActionItem(
                owner="Bob",
                task="Update the changelog before EOD",
                deadline="today 5 pm",
                source_message_id="msg_3",
            )
        ],
        decisions=[
            Decision(
                decision="Release date is June 15th",
                made_by="Alice",
                source_message_id="msg_7",
            )
        ],
        open_questions=[
            OpenQuestion(
                asker="Carol",
                question="Who is writing the blog post?",
                answered=False,
                source_message_id="msg_8",
            )
        ],
    )


@pytest.fixture()
def overlapping_chunk_pair() -> tuple[ExtractionResult, ExtractionResult]:
    """Two ExtractionResults simulating a 3-message overlap between chunks.

    chunk_a covers msg_0..msg_14, chunk_b covers msg_12..msg_19.
    Several items appear in both with trivial phrasing differences.
    """
    chunk_a = ExtractionResult(
        action_items=[
            ActionItem(
                owner="Bob",
                task="Update the changelog before EOD",
                deadline="today 5 pm",
                source_message_id="msg_3",
            ),
            ActionItem(
                owner="Alice",
                task="Rotate the staging deploy key",
                deadline=None,
                source_message_id="msg_5",
            ),
        ],
        decisions=[
            Decision(
                decision="Release date is June 15th",
                made_by="Alice",
                source_message_id="msg_2",
            )
        ],
        open_questions=[
            OpenQuestion(
                asker="Carol",
                question="Who is handling the blog post announcement?",
                answered=False,
                source_message_id="msg_8",
            ),
            OpenQuestion(
                asker="Alice",
                question="Has the staging deploy key been rotated yet?",
                answered=False,
                source_message_id="msg_5",
            ),
        ],
    )
    chunk_b = ExtractionResult(
        action_items=[
            # Near-duplicate of chunk_a's changelog action — trailing period
            ActionItem(
                owner="Bob",
                task="Update the changelog before EOD.",
                deadline="today 5 pm",
                source_message_id="msg_12",
            ),
            # New item not in chunk_a
            ActionItem(
                owner="Bob",
                task="Write the blog post announcement",
                deadline="June 14th",
                source_message_id="msg_13",
            ),
            # Exact duplicate of chunk_a's deploy-key action
            ActionItem(
                owner="Alice",
                task="Rotate the staging deploy key",
                deadline=None,
                source_message_id="msg_14",
            ),
        ],
        decisions=[
            # Near-duplicate release decision
            Decision(
                decision="Release date is June 15th.",
                made_by="Team",
                source_message_id="msg_12",
            ),
            # New decision
            Decision(
                decision="Carol reviews the blog post before it is published",
                made_by="Carol",
                source_message_id="msg_14",
            ),
        ],
        open_questions=[
            # Exact duplicate of Carol's question in chunk_a
            OpenQuestion(
                asker="Carol",
                question="Who is handling the blog post announcement?",
                answered=False,
                source_message_id="msg_8",
            ),
            # New question
            OpenQuestion(
                asker="Bob",
                question="What is the deadline for the blog post?",
                answered=False,
                source_message_id="msg_13",
            ),
        ],
    )
    return chunk_a, chunk_b


# ---------------------------------------------------------------------------
# Fake httpx transports
# ---------------------------------------------------------------------------


class ScriptedTransport(httpx.BaseTransport):
    """Serves a scripted list of Ollama response strings in order.

    The last response is repeated if the script is exhausted.
    Each call increments ``.calls`` so tests can assert call counts.
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0
        self.prompts_seen: list[str] = []  # record prompts for content assertions

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = request.content.decode("utf-8")
        self.prompts_seen.append(json.loads(body).get("prompt", ""))
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return httpx.Response(200, json={"response": self.responses[idx]})


class UnreachableTransport(httpx.BaseTransport):
    """Simulates Ollama not running (TCP connection refused)."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)


class TimeoutTransport(httpx.BaseTransport):
    """Simulates an Ollama request that exceeds the timeout."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)


class HttpErrorTransport(httpx.BaseTransport):
    """Returns a configurable HTTP error status code."""

    def __init__(self, status_code: int = 500) -> None:
        self.status_code = status_code

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(self.status_code, text="Internal Server Error")


@pytest.fixture()
def scripted(request) -> ScriptedTransport:
    """Parametrised fixture: pass responses list via indirect or direct marker."""
    responses = getattr(request, "param", [VALID_OLLAMA_RESPONSE])
    return ScriptedTransport(responses)