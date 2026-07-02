"""Tests for merge.py.

All tests go through the public merge_results() function rather than
internal helpers — if the behaviour is correct at the API boundary the
implementation details can change freely.

Covers:
  - empty input returns empty ExtractionResult
  - single chunk is returned as-is (no dedup needed)
  - near-identical action items deduplicated (same owner, similar task)
  - distinct action items for the same owner are kept separately
  - near-identical decisions deduplicated
  - distinct decisions kept
  - open question marked answered when a later action item overlaps topically
  - open question marked answered when a later decision overlaps topically
  - open question stays open when nothing in later chunks addresses it
  - duplicate open questions deduplicated; answered=True version wins
  - source_message_id from the first occurrence is preserved after dedup
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from merge import merge_results  # noqa: E402
from schema import ActionItem, Decision, ExtractionResult, OpenQuestion  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def action(owner: str, task: str, src: str = "msg_0", deadline: str | None = None) -> ActionItem:
    return ActionItem(owner=owner, task=task, deadline=deadline, source_message_id=src)


def decision(text: str, by: str = "Alice", src: str = "msg_0") -> Decision:
    return Decision(decision=text, made_by=by, source_message_id=src)


def question(asker: str, text: str, src: str = "msg_0", answered: bool = False) -> OpenQuestion:
    return OpenQuestion(asker=asker, question=text, answered=answered, source_message_id=src)


def result(*args) -> ExtractionResult:
    """Build an ExtractionResult from mixed ActionItem/Decision/OpenQuestion args."""
    return ExtractionResult(
        action_items=[a for a in args if isinstance(a, ActionItem)],
        decisions=[d for d in args if isinstance(d, Decision)],
        open_questions=[q for q in args if isinstance(q, OpenQuestion)],
    )


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_result():
    merged = merge_results([])
    assert merged == ExtractionResult()


def test_single_chunk_returned_unchanged():
    r = result(
        action("Bob", "Review the PR", "msg_1"),
        decision("Release on June 15th", "Alice", "msg_2"),
        question("Carol", "Who owns the blog post?", "msg_3"),
    )
    merged = merge_results([r])
    assert len(merged.action_items) == 1
    assert len(merged.decisions) == 1
    assert len(merged.open_questions) == 1


def test_multiple_distinct_items_all_kept():
    a = result(action("Alice", "Deploy to staging", "msg_1"))
    b = result(action("Bob", "Update the changelog", "msg_5"))
    merged = merge_results([a, b])
    assert len(merged.action_items) == 2


# ---------------------------------------------------------------------------
# Action item dedup
# ---------------------------------------------------------------------------


def test_near_identical_action_items_deduplicated():
    """Trailing period and minor whitespace must not survive as a duplicate."""
    a = result(action("Bob", "Update the changelog before EOD", "msg_1"))
    b = result(action("Bob", "Update the changelog before EOD.", "msg_8"))
    merged = merge_results([a, b])
    assert len(merged.action_items) == 1


def test_first_occurrence_source_id_preserved_on_action_dedup():
    a = result(action("Bob", "Write the release notes", "msg_2"))
    b = result(action("Bob", "Write the release notes", "msg_9"))
    merged = merge_results([a, b])
    assert merged.action_items[0].source_message_id == "msg_2"


def test_different_tasks_same_owner_both_kept():
    a = result(action("Bob", "Review the PR", "msg_1"))
    b = result(action("Bob", "Deploy to staging", "msg_5"))
    merged = merge_results([a, b])
    assert len(merged.action_items) == 2


def test_same_task_different_owners_both_kept():
    a = result(action("Alice", "Review the PR", "msg_1"))
    b = result(action("Bob", "Review the PR", "msg_2"))
    merged = merge_results([a, b])
    assert len(merged.action_items) == 2


def test_three_chunks_with_repeated_action_collapses_to_one():
    chunks = [result(action("Alice", "Rotate the deploy key", f"msg_{i}")) for i in range(3)]
    merged = merge_results(chunks)
    assert len(merged.action_items) == 1
    assert merged.action_items[0].source_message_id == "msg_0"


# ---------------------------------------------------------------------------
# Decision dedup
# ---------------------------------------------------------------------------


def test_near_identical_decisions_deduplicated():
    a = result(decision("Release date is June 15th", "Alice", "msg_2"))
    b = result(decision("Release date is June 15th.", "Team", "msg_9"))
    merged = merge_results([a, b])
    assert len(merged.decisions) == 1


def test_first_occurrence_source_id_preserved_on_decision_dedup():
    a = result(decision("Use Postgres for the database", "Alice", "msg_3"))
    b = result(decision("Use Postgres for the database", "Bob", "msg_11"))
    merged = merge_results([a, b])
    assert merged.decisions[0].source_message_id == "msg_3"


def test_distinct_decisions_both_kept():
    a = result(decision("Release on June 15th", "Alice", "msg_2"))
    b = result(decision("Use Postgres for database", "Bob", "msg_7"))
    merged = merge_results([a, b])
    assert len(merged.decisions) == 2


# ---------------------------------------------------------------------------
# Open question answered heuristic
# ---------------------------------------------------------------------------


def test_question_marked_answered_by_later_action_item():
    """'Who handles the blog post?' answered by 'Bob: write blog post'."""
    chunk_a = result(question("Carol", "Who is handling the blog post?", "msg_4"))
    chunk_b = result(action("Bob", "Write the blog post announcement", "msg_9"))
    merged = merge_results([chunk_a, chunk_b])
    carol_q = next(q for q in merged.open_questions if q.asker == "Carol")
    assert carol_q.answered is True


def test_question_marked_answered_by_later_decision():
    """'What is the release date?' answered by a release-date decision."""
    chunk_a = result(question("Bob", "What is the release date?", "msg_2"))
    chunk_b = result(decision("The release date is June 15th", "Alice", "msg_8"))
    merged = merge_results([chunk_a, chunk_b])
    bob_q = next(q for q in merged.open_questions if q.asker == "Bob")
    assert bob_q.answered is True


def test_question_stays_open_when_nothing_addresses_it():
    """A question about payments has nothing to do with blog-post actions."""
    chunk_a = result(question("Alice", "Who approves the budget?", "msg_1"))
    chunk_b = result(action("Bob", "Write the blog post", "msg_6"))
    merged = merge_results([chunk_a, chunk_b])
    alice_q = next(q for q in merged.open_questions if q.asker == "Alice")
    assert alice_q.answered is False


def test_already_answered_question_preserved():
    chunk = result(question("Alice", "Is the server down?", "msg_1", answered=True))
    merged = merge_results([chunk])
    assert merged.open_questions[0].answered is True


# ---------------------------------------------------------------------------
# Open question dedup
# ---------------------------------------------------------------------------


def test_duplicate_questions_deduplicated():
    a = result(question("Carol", "Who owns the blog post?", "msg_4"))
    b = result(question("Carol", "Who owns the blog post?", "msg_11"))
    merged = merge_results([a, b])
    assert len(merged.open_questions) == 1


def test_answered_version_wins_on_question_dedup():
    """When duplicates disagree, the answered=True copy must be kept."""
    a = result(question("Bob", "Has the key been rotated?", "msg_3", answered=False))
    b = result(question("Bob", "Has the key been rotated?", "msg_10", answered=True))
    merged = merge_results([a, b])
    assert len(merged.open_questions) == 1
    assert merged.open_questions[0].answered is True


def test_distinct_questions_same_asker_both_kept():
    a = result(question("Alice", "What is the release date?", "msg_2"))
    b = result(question("Alice", "Who writes the changelog?", "msg_6"))
    merged = merge_results([a, b])
    assert len(merged.open_questions) == 2


# ---------------------------------------------------------------------------
# Integration: realistic two-chunk overlap scenario
# ---------------------------------------------------------------------------


def test_realistic_two_chunk_overlap():
    """End-to-end scenario with overlapping chunks, mixing all three categories."""
    chunk_a = ExtractionResult(
        action_items=[
            action("Bob", "Update the changelog before EOD", "msg_3", "today 5 pm"),
            action("Alice", "Rotate the staging deploy key", "msg_5"),
        ],
        decisions=[
            decision("Release date is June 15th", "Alice", "msg_2"),
        ],
        open_questions=[
            question("Carol", "Who is handling the blog post announcement?", "msg_8"),
            question("Alice", "Has the staging deploy key been rotated yet?", "msg_5"),
        ],
    )
    chunk_b = ExtractionResult(
        action_items=[
            action("Bob", "Update the changelog before EOD.", "msg_12", "today 5 pm"),  # dup
            action("Bob", "Write the blog post announcement", "msg_13", "June 14th"),  # new
            action("Alice", "Rotate the staging deploy key", "msg_14"),  # dup
        ],
        decisions=[
            decision("Release date is June 15th.", "Team", "msg_12"),  # dup
            decision("Carol reviews the blog post before publishing", "Carol", "msg_14"),  # new
        ],
        open_questions=[
            question("Carol", "Who is handling the blog post announcement?", "msg_8"),  # dup
            question("Bob", "What is the deadline for the blog post?", "msg_13"),  # new
        ],
    )

    merged = merge_results([chunk_a, chunk_b])

    # Counts
    assert len(merged.action_items) == 3, f"Expected 3 actions, got {len(merged.action_items)}"
    assert len(merged.decisions) == 2, f"Expected 2 decisions, got {len(merged.decisions)}"
    assert (
        len(merged.open_questions) == 3
    ), f"Expected 3 questions, got {len(merged.open_questions)}"

    # First-occurrence IDs preserved
    changelog = next(i for i in merged.action_items if "changelog" in i.task.lower())
    assert changelog.source_message_id == "msg_3"

    release = next(d for d in merged.decisions if "june 15" in d.decision.lower())
    assert release.source_message_id == "msg_2"

    # Carol's question answered by Bob's blog-post action
    carol_q = next(q for q in merged.open_questions if q.asker == "Carol")
    assert carol_q.answered is True

    # Alice's deploy-key question answered by her own action item
    alice_q = next(q for q in merged.open_questions if q.asker == "Alice")
    assert alice_q.answered is True