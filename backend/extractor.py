"""extractor.py — Ollama-backed extraction for ChatLedger.

Public API
----------
    extract_chunk(
        chunk: List[dict],
        model: str = "phi3:mini",
    ) -> ExtractionResult

Calls a locally-running Ollama instance at http://localhost:11434, asks it
to extract action items, decisions, and open questions from the chunk, and
validates the response against the ExtractionResult pydantic schema.

Retry / fallback
----------------
On a JSON parse error or schema validation failure the function sends a
single corrective follow-up prompt that includes the exact error message.
If that second attempt also fails, a warning is logged and an empty
ExtractionResult is returned — the pipeline never crashes on a bad chunk.

Caching
-------
Each chunk is hashed (SHA-256 of its rendered text) and the result cached
in  <backend_dir>/.cache/  as a JSON file.  Re-running the pipeline on the
same export skips the Ollama call entirely for every cached chunk.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import httpx
from pydantic import ValidationError

from schema import ExtractionResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT_SECONDS = 120.0  # SLMs can be slow on CPU; give them time

_CACHE_DIR = Path(__file__).parent / ".cache"

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

# The schema block is embedded verbatim so the model always sees the exact
# expected JSON shape next to its instructions.
_EXTRACTION_PROMPT = """\
You are an information-extraction engine built into a chat-analysis tool \
called ChatLedger.

Your job is to read the conversation transcript below and extract three \
categories of structured items:

1. ACTION ITEMS — a specific task assigned to (or accepted by) a named \
person, optionally with a deadline.
2. DECISIONS — something the group has agreed upon or resolved.
3. OPEN QUESTIONS — a question raised by a participant that has not yet \
been answered within this transcript.

RULES
-----
- Every extracted item MUST include a source_message_id taken verbatim \
from the [msg_N] tags in the transcript.  Do NOT invent IDs.
- If a field is unknown or absent, use null (for deadline) or false \
(for answered).
- owner / made_by / asker must be the speaker's name exactly as it \
appears in the transcript, not a pronoun.
- If a category has no items, return an empty list [] for it.
- Be conservative: only extract items that are clearly stated; do not \
infer items that are merely implied.

OUTPUT FORMAT
-------------
Respond with ONLY a single valid JSON object — no markdown fences, no \
commentary, no trailing text.  The object must match this schema exactly:

{
  "action_items": [
    {
      "owner":             "<name of person responsible>",
      "task":              "<concise description of the task>",
      "deadline":          "<date/time string, or null>",
      "source_message_id": "<msg_N>"
    }
  ],
  "decisions": [
    {
      "decision":          "<what was decided>",
      "made_by":           "<name of person or group who decided>",
      "source_message_id": "<msg_N>"
    }
  ],
  "open_questions": [
    {
      "asker":             "<name of person who asked>",
      "question":          "<the question text>",
      "answered":          false,
      "source_message_id": "<msg_N>"
    }
  ]
}

TRANSCRIPT
----------
{transcript}
"""

_CORRECTIVE_PROMPT = """\
Your previous response could not be parsed.  The error was:

  {error}

Re-read the transcript and respond again with ONLY a valid JSON object \
matching the schema described earlier (keys: action_items, decisions, \
open_questions).  No markdown, no commentary — JSON only.

TRANSCRIPT
----------
{transcript}
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_chunk(chunk: list[dict]) -> str:
    """Format a list of message dicts as a numbered transcript string."""
    lines = []
    for msg in chunk:
        mid = msg.get("message_id", "?")
        ts = msg.get("timestamp", "")
        sndr = msg.get("sender", "?")
        # Use 'text' (parser.py key) with fallback to 'message_text' (legacy)
        body = msg.get("text") or msg.get("message_text", "")
        lines.append(f"[{mid}] {ts} {sndr}: {body}")
    return "\n".join(lines)


def _chunk_hash(transcript: str) -> str:
    """SHA-256 hex digest of the rendered transcript — used as cache key."""
    return hashlib.sha256(transcript.encode("utf-8")).hexdigest()


def _cache_path(digest: str) -> Path:
    return _CACHE_DIR / f"{digest}.json"


def _load_cache(digest: str) -> ExtractionResult | None:
    path = _cache_path(digest)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        result = ExtractionResult.model_validate(data)
        logger.debug("Cache hit for chunk %s…", digest[:12])
        return result
    except Exception as exc:  # corrupt cache file — ignore it
        logger.warning("Ignoring corrupt cache file %s: %s", path.name, exc)
        return None


def _save_cache(digest: str, result: ExtractionResult) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _cache_path(digest).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write cache file: %s", exc)


def _call_ollama(prompt: str, model: str, client: httpx.Client) -> str:
    """POST to Ollama and return the raw response string.

    Raises httpx.HTTPError on transport / HTTP-level failure.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }
    response = client.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("response", "")


def _parse_response(raw: str) -> ExtractionResult:
    """Parse and validate a raw model response string.

    Raises json.JSONDecodeError or pydantic.ValidationError on failure.
    """
    data = json.loads(raw)
    return ExtractionResult.model_validate(data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_chunk(
    chunk: list[dict],
    model: str = "phi3:mini",
    *,
    _client: httpx.Client | None = None,  # injectable for tests
) -> ExtractionResult:
    """Extract action items, decisions, and open questions from *chunk*.

    Parameters
    ----------
    chunk:
        A list of message dicts (as produced by ``parser.parse_chat``).
    model:
        Ollama model tag.  Must be pulled locally before use.
    _client:
        Optional httpx.Client for dependency injection in tests.
        When omitted a short-lived client is created and closed internally.

    Returns
    -------
    An ``ExtractionResult`` (possibly with empty lists if both attempts
    failed or the chunk was empty).
    """
    if not chunk:
        return ExtractionResult()

    transcript = _render_chunk(chunk)
    digest = _chunk_hash(transcript)

    # ── Cache check ──────────────────────────────────────────────────────
    cached = _load_cache(digest)
    if cached is not None:
        return cached

    # ── Build first prompt ───────────────────────────────────────────────
    prompt = _EXTRACTION_PROMPT.replace("{transcript}", transcript)

    owns_client = _client is None
    client = _client or httpx.Client()

    try:
        # ── Attempt 1 ────────────────────────────────────────────────────
        last_error = ""
        try:
            raw = _call_ollama(prompt, model, client)
            result = _parse_response(raw)
            _save_cache(digest, result)
            return result

        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            logger.warning(
                "Attempt 1 parse/validation failure for chunk %s…: %s",
                digest[:12],
                last_error,
            )

        except httpx.HTTPError as exc:
            last_error = str(exc)
            logger.warning(
                "Attempt 1 HTTP failure for chunk %s…: %s",
                digest[:12],
                last_error,
            )
            # Network errors aren't fixable by rephrasing — skip retry
            return ExtractionResult()

        # ── Attempt 2 (corrective prompt) ─────────────────────────────────
        corrective = _CORRECTIVE_PROMPT.replace("{error}", last_error).replace(
            "{transcript}", transcript
        )
        try:
            raw = _call_ollama(corrective, model, client)
            result = _parse_response(raw)
            _save_cache(digest, result)
            return result

        except (json.JSONDecodeError, ValidationError, httpx.HTTPError) as exc:
            logger.warning(
                "Attempt 2 also failed for chunk %s…: %s — returning empty result.",
                digest[:12],
                exc,
            )
            return ExtractionResult()

    finally:
        if owns_client:
            client.close()


# ---------------------------------------------------------------------------
# __main__ — smoke-test against a real local Ollama instance
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    # ── Build a realistic fake chunk ──────────────────────────────────────
    fake_chunk = [
        {
            "message_id": "msg_0",
            "timestamp": "01/06/24, 09:00:00",
            "sender": "Alice",
            "text": "Morning everyone. Can we confirm the release date?",
        },
        {
            "message_id": "msg_1",
            "timestamp": "01/06/24, 09:00:45",
            "sender": "Bob",
            "text": "I think we agreed on the 15th, but let me double-check with Carol.",
        },
        {
            "message_id": "msg_2",
            "timestamp": "01/06/24, 09:01:10",
            "sender": "Carol",
            "text": "Yes — decision made last Friday: release is June 15th.",
        },
        {
            "message_id": "msg_3",
            "timestamp": "01/06/24, 09:01:30",
            "sender": "Alice",
            "text": "Great. Bob, can you update the changelog before EOD?",
        },
        {
            "message_id": "msg_4",
            "timestamp": "01/06/24, 09:01:50",
            "sender": "Bob",
            "text": "Sure, I'll have it done by 5 pm today.",
        },
        {
            "message_id": "msg_5",
            "timestamp": "01/06/24, 09:02:15",
            "sender": "Carol",
            "text": "Who's handling the blog post announcement?",
        },
        {
            "message_id": "msg_6",
            "timestamp": "01/06/24, 09:02:40",
            "sender": "Alice",
            "text": "That's still open — no one's been assigned yet.",
        },
        {
            "message_id": "msg_7",
            "timestamp": "01/06/24, 09:03:00",
            "sender": "Bob",
            "text": "I can take it if Carol reviews before we publish.",
        },
        {
            "message_id": "msg_8",
            "timestamp": "01/06/24, 09:03:20",
            "sender": "Carol",
            "text": "Works for me. So Bob writes the post, Carol reviews — agreed?",
        },
        {
            "message_id": "msg_9",
            "timestamp": "01/06/24, 09:03:35",
            "sender": "Alice",
            "text": "Agreed. Let's target having it ready by June 14th.",
        },
    ]

    print("=" * 60)
    print("ChatLedger — extractor.py smoke test")
    print("=" * 60)
    print("Model  : phi3:mini")
    print(
        f"Chunk  : {len(fake_chunk)} messages  ({fake_chunk[0]['message_id']} … {fake_chunk[-1]['message_id']})"
    )
    print(f"Ollama : {OLLAMA_URL}")
    cache_dir_display = _CACHE_DIR.relative_to(Path(__file__).parent)
    print(f"Cache  : {cache_dir_display}/")
    print()

    print("Transcript sent to model:")
    print("─" * 60)
    print(_render_chunk(fake_chunk))
    print("─" * 60)
    print()

    print("Calling Ollama … (this may take 10–60 s on CPU)")
    result = extract_chunk(fake_chunk)

    # ── Pretty-print result ───────────────────────────────────────────────
    print()
    print("═" * 60)
    print("EXTRACTION RESULT")
    print("═" * 60)

    print(f"\nAction Items  ({len(result.action_items)})")
    print("─" * 40)
    for item in result.action_items:
        print(f"  [{item.source_message_id}]  {item.owner}: {item.task}")
        if item.deadline:
            print(f"             deadline: {item.deadline}")

    print(f"\nDecisions  ({len(result.decisions)})")
    print("─" * 40)
    for dec in result.decisions:
        print(f"  [{dec.source_message_id}]  {dec.made_by}: {dec.decision}")

    print(f"\nOpen Questions  ({len(result.open_questions)})")
    print("─" * 40)
    for q in result.open_questions:
        answered = "answered" if q.answered else "open"
        print(f"  [{q.source_message_id}]  {q.asker}: {q.question}  [{answered}]")

    total = len(result.action_items) + len(result.decisions) + len(result.open_questions)
    print(f"\nTotal items extracted: {total}")

    # Run again to confirm cache is used (should be instant)
    print()
    print("Running extract_chunk again on the same chunk …")
    result2 = extract_chunk(fake_chunk)
    assert result2 == result, "Cache returned a different result!"
    print("✓  Identical result returned from cache (Ollama was NOT called again).")

    sys.exit(0)