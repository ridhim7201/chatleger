"""db.py — SQLite storage layer for ChatLedger.

Public API
----------
    init_db(db_path: str | Path) -> None
        Create all tables (idempotent — safe to call on every startup).

    insert_messages(db_path, messages: List[dict]) -> None
        Bulk-insert parsed message rows.  Uses INSERT OR IGNORE so the
        same file can be re-uploaded without duplicating rows.

    insert_extraction_result(db_path, result: ExtractionResult) -> None
        Insert action items, decisions, and open questions from one
        ExtractionResult.  Skips items whose source_message_id does not
        exist in the messages table (orphan guard).

Schema
------
    messages        — raw chat rows, message_id is the natural PK
    action_items    — auto-increment id, FK → messages.message_id
    decisions       — auto-increment id, FK → messages.message_id
    open_questions  — auto-increment id, FK → messages.message_id

All FK constraints are enforced via PRAGMA foreign_keys = ON, set on every
connection in _connect().
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from schema import ExtractionResult

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS messages (
    message_id  TEXT    PRIMARY KEY,
    timestamp   TEXT    NOT NULL,
    sender      TEXT    NOT NULL,
    text        TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS action_items (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    owner             TEXT    NOT NULL,
    task              TEXT    NOT NULL,
    deadline          TEXT,                      -- nullable
    source_message_id TEXT    NOT NULL,
    FOREIGN KEY (source_message_id) REFERENCES messages (message_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    decision          TEXT    NOT NULL,
    made_by           TEXT    NOT NULL,
    source_message_id TEXT    NOT NULL,
    FOREIGN KEY (source_message_id) REFERENCES messages (message_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS open_questions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    asker             TEXT    NOT NULL,
    question          TEXT    NOT NULL,
    answered          INTEGER NOT NULL DEFAULT 0, -- 0 = False, 1 = True
    source_message_id TEXT    NOT NULL,
    FOREIGN KEY (source_message_id) REFERENCES messages (message_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunk_cache (
    chunk_hash  TEXT PRIMARY KEY,
    result_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extraction_failures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id    TEXT    NOT NULL,
    chunk_hash  TEXT    NOT NULL,
    error       TEXT    NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


@contextmanager
def _connect(db_path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    """Yield an auto-committing connection with FK enforcement enabled.

    The connection is committed on clean exit and rolled back on exception,
    then closed in both cases.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_db(db_path: str | Path) -> None:
    """Create all tables if they do not already exist.

    Idempotent — safe to call on every application startup.  The WAL
    journal mode is also set here for better concurrent read performance.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite file.  The file is created if it
        does not exist (standard sqlite3 behaviour).
    """
    with _connect(db_path) as conn:
        conn.executescript(_DDL)


def reset_db(db_path: str | Path) -> None:
    """Delete all extracted data and messages, keeping the schema intact.

    The chunk_cache table is intentionally preserved — cached Ollama results
    remain valid regardless of which file is loaded, so there's no reason to
    re-run extraction on chunks the model has already seen.
    """
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM open_questions")
        conn.execute("DELETE FROM decisions")
        conn.execute("DELETE FROM action_items")
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM extraction_failures")


def insert_messages(db_path: str | Path, messages: list[dict]) -> None:
    """Bulk-insert parsed message rows into the messages table.

    Uses INSERT OR IGNORE so re-uploading the same export does not
    duplicate rows — the existing row is silently kept.

    Parameters
    ----------
    db_path:
        Path to the initialised SQLite database.
    messages:
        List of dicts with keys: message_id, timestamp, sender, text.
        Extra keys are ignored.
    """
    if not messages:
        return

    rows = [
        (
            m["message_id"],
            m["timestamp"],
            m["sender"],
            m.get("text") or m.get("message_text", ""),  # tolerate both key names
        )
        for m in messages
    ]

    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO messages (message_id, timestamp, sender, text)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )


def insert_extraction_result(db_path: str | Path, result: ExtractionResult) -> None:
    """Persist all items in *result* to the appropriate tables.

    Items whose source_message_id is not present in the messages table are
    skipped with a warning printed to stdout — they would violate the FK
    constraint and indicate a model hallucinating a non-existent message ID.

    Parameters
    ----------
    db_path:
        Path to the initialised SQLite database.
    result:
        An ExtractionResult (from extractor.py or merge.py) whose
        source_message_ids should already exist in the messages table.
    """
    with _connect(db_path) as conn:
        # Build a set of known message_ids for the orphan guard.
        known_ids: set[str] = {
            row[0] for row in conn.execute("SELECT message_id FROM messages").fetchall()
        }

        def _check(source_id: str, kind: str) -> bool:
            if source_id not in known_ids:
                print(
                    f"[db] WARNING: skipping {kind} — "
                    f"source_message_id {source_id!r} not in messages table"
                )
                return False
            return True

        # ── Action items ─────────────────────────────────────────────────
        conn.executemany(
            """
            INSERT INTO action_items (owner, task, deadline, source_message_id)
            VALUES (?, ?, ?, ?)
            """,
            [
                (item.owner, item.task, item.deadline, item.source_message_id)
                for item in result.action_items
                if _check(item.source_message_id, "action_item")
            ],
        )

        # ── Decisions ────────────────────────────────────────────────────
        conn.executemany(
            """
            INSERT INTO decisions (decision, made_by, source_message_id)
            VALUES (?, ?, ?)
            """,
            [
                (dec.decision, dec.made_by, dec.source_message_id)
                for dec in result.decisions
                if _check(dec.source_message_id, "decision")
            ],
        )

        # ── Open questions ───────────────────────────────────────────────
        conn.executemany(
            """
            INSERT INTO open_questions (asker, question, answered, source_message_id)
            VALUES (?, ?, ?, ?)
            """,
            [
                (q.asker, q.question, int(q.answered), q.source_message_id)
                for q in result.open_questions
                if _check(q.source_message_id, "open_question")
            ],
        )


# ---------------------------------------------------------------------------
# Read helpers (used by main.py API routes and __main__ below)
# ---------------------------------------------------------------------------


def fetch_messages(db_path: str | Path) -> list[dict]:
    with _connect(db_path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM messages ORDER BY message_id")]


def fetch_action_items(db_path: str | Path, owner: str | None = None) -> list[dict]:
    with _connect(db_path) as conn:
        if owner:
            rows = conn.execute(
                "SELECT * FROM action_items WHERE lower(owner) = lower(?) ORDER BY id",
                (owner,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM action_items ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def fetch_decisions(db_path: str | Path, made_by: str | None = None) -> list[dict]:
    with _connect(db_path) as conn:
        if made_by:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE lower(made_by) = lower(?) ORDER BY id",
                (made_by,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM decisions ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def fetch_open_questions(db_path: str | Path, asker: str | None = None) -> list[dict]:
    with _connect(db_path) as conn:
        if asker:
            rows = conn.execute(
                "SELECT * FROM open_questions WHERE lower(asker) = lower(?) ORDER BY id",
                (asker,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM open_questions ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def fetch_source_message(db_path: str | Path, message_id: str) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# __main__ — round-trip smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile

    from schema import ActionItem, Decision, ExtractionResult, OpenQuestion

    # ── 1. Set up a fresh temp database ──────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    DB = tmp.name
    print(f"Test database : {DB}\n")

    init_db(DB)
    print("init_db()         ✓  tables created")

    # ── 2. Insert messages ────────────────────────────────────────────────
    messages = [
        {
            "message_id": "msg_0",
            "timestamp": "01/06/24, 09:00:00",
            "sender": "Alice",
            "text": "Can we confirm the release date?",
        },
        {
            "message_id": "msg_1",
            "timestamp": "01/06/24, 09:00:45",
            "sender": "Bob",
            "text": "I think we agreed on June 15th.",
        },
        {
            "message_id": "msg_2",
            "timestamp": "01/06/24, 09:01:10",
            "sender": "Carol",
            "text": "Confirmed — release is June 15th.",
        },
        {
            "message_id": "msg_3",
            "timestamp": "01/06/24, 09:01:30",
            "sender": "Alice",
            "text": "Bob, can you update the changelog before EOD?",
        },
        {
            "message_id": "msg_4",
            "timestamp": "01/06/24, 09:01:50",
            "sender": "Bob",
            "text": "Sure, done by 5 pm.",
        },
        {
            "message_id": "msg_5",
            "timestamp": "01/06/24, 09:02:15",
            "sender": "Carol",
            "text": "Who's handling the blog post announcement?",
        },
    ]

    insert_messages(DB, messages)
    print(f"insert_messages() ✓  {len(messages)} rows inserted")

    # Idempotency check — inserting the same rows again must not error or duplicate
    insert_messages(DB, messages)
    count = len(fetch_messages(DB))
    assert count == len(messages), f"Expected {len(messages)}, got {count} after re-insert"
    print(f"                  ✓  re-insert is idempotent ({count} rows, no duplicates)")

    # ── 3. Insert extraction result ───────────────────────────────────────
    result = ExtractionResult(
        action_items=[
            ActionItem(
                owner="Bob",
                task="Update the changelog before EOD",
                deadline="today 5 pm",
                source_message_id="msg_3",
            ),
            ActionItem(
                owner="Alice",
                task="Confirm the release date with the team",
                deadline=None,
                source_message_id="msg_0",
            ),
        ],
        decisions=[
            Decision(
                decision="Release date is June 15th", made_by="Carol", source_message_id="msg_2"
            ),
        ],
        open_questions=[
            OpenQuestion(
                asker="Carol",
                question="Who is handling the blog post announcement?",
                answered=False,
                source_message_id="msg_5",
            ),
        ],
    )

    insert_extraction_result(DB, result)
    print("insert_extraction_result() ✓")

    # ── 4. Orphan guard test ──────────────────────────────────────────────
    orphan_result = ExtractionResult(
        action_items=[
            ActionItem(
                owner="Ghost", task="This should be skipped", source_message_id="msg_999"
            ),  # does not exist
        ],
        decisions=[],
        open_questions=[],
    )
    print("\nOrphan guard (expect 1 warning line below):")
    insert_extraction_result(DB, orphan_result)
    orphan_actions = fetch_action_items(DB, owner="Ghost")
    assert len(orphan_actions) == 0, "Orphan row should have been rejected"
    print("                  ✓  orphan skipped, FK constraint intact\n")

    # ── 5. Query and print all tables ────────────────────────────────────
    SEP = "─" * 64

    msgs = fetch_messages(DB)
    print(f"messages  ({len(msgs)} rows)")
    print(SEP)
    for m in msgs:
        print(f"  {m['message_id']:<8} {m['timestamp']}  {m['sender']:<8} {m['text']}")

    actions = fetch_action_items(DB)
    print(f"\naction_items  ({len(actions)} rows)")
    print(SEP)
    for a in actions:
        dl = f"  deadline={a['deadline']}" if a["deadline"] else ""
        print(f"  id={a['id']}  [{a['source_message_id']}]  {a['owner']}: {a['task']}{dl}")

    decs = fetch_decisions(DB)
    print(f"\ndecisions  ({len(decs)} rows)")
    print(SEP)
    for d in decs:
        print(f"  id={d['id']}  [{d['source_message_id']}]  {d['made_by']}: {d['decision']}")

    qs = fetch_open_questions(DB)
    print(f"\nopen_questions  ({len(qs)} rows)")
    print(SEP)
    for q in qs:
        status = "answered" if q["answered"] else "open"
        print(
            f"  id={q['id']}  [{q['source_message_id']}]  {q['asker']}: {q['question']}  [{status}]"
        )

    # ── 6. Filter queries ─────────────────────────────────────────────────
    print("\nFilter: action_items WHERE owner = 'Bob'")
    bob_actions = fetch_action_items(DB, owner="Bob")
    assert len(bob_actions) == 1 and bob_actions[0]["owner"] == "Bob"
    print(f"  ✓  {len(bob_actions)} row(s)  →  {bob_actions[0]['task']}")

    print("\nFilter: action_items WHERE owner = 'bob'  (case-insensitive)")
    bob_lower = fetch_action_items(DB, owner="bob")
    assert len(bob_lower) == 1
    print(f"  ✓  {len(bob_lower)} row(s)  →  case fold works")

    # ── 7. Source-message lookup ──────────────────────────────────────────
    print("\nfetch_source_message('msg_2')")
    src = fetch_source_message(DB, "msg_2")
    assert src is not None and src["sender"] == "Carol"
    print(f"  ✓  {src['sender']}: {src['text']}")

    print("\nfetch_source_message('msg_999')  (missing)")
    missing = fetch_source_message(DB, "msg_999")
    assert missing is None
    print("  ✓  returned None as expected")

    # ── 8. Round-trip assertions ──────────────────────────────────────────
    print()
    assert len(fetch_action_items(DB)) == 2
    assert len(fetch_decisions(DB)) == 1
    assert len(fetch_open_questions(DB)) == 1
    assert fetch_open_questions(DB)[0]["answered"] == 0  # stored as int
    assert fetch_decisions(DB)[0]["decision"] == "Release date is June 15th"
    print("All round-trip assertions passed. ✓")

    # ── Cleanup ───────────────────────────────────────────────────────────
    os.unlink(DB)
    print("\nTemp database deleted.")
