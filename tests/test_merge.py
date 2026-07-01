import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from merge import (  # noqa: E402
    dedup_action_items,
    dedup_decisions,
    dedup_open_questions,
    merge_chunk_results,
)
from schema import ActionItem, Decision, ExtractionResult, OpenQuestion  # noqa: E402


def test_dedup_action_items_removes_near_identical_across_chunks():
    items = [
        ActionItem(owner="Bob", task="Review the PR", deadline="Friday", source_message_id="msg_1"),
        ActionItem(
            owner="Bob", task="Review the PR.", deadline="Friday", source_message_id="msg_5"
        ),
        ActionItem(owner="Alice", task="Ship the release", source_message_id="msg_9"),
    ]
    deduped = dedup_action_items(items)
    assert len(deduped) == 2
    owners = {i.owner for i in deduped}
    assert owners == {"Bob", "Alice"}


def test_dedup_action_items_keeps_distinct_tasks_for_same_owner():
    items = [
        ActionItem(owner="Bob", task="Review the PR", source_message_id="msg_1"),
        ActionItem(owner="Bob", task="Deploy to staging", source_message_id="msg_2"),
    ]
    deduped = dedup_action_items(items)
    assert len(deduped) == 2


def test_dedup_decisions_removes_near_identical():
    decisions = [
        Decision(
            decision="We will use Postgres for storage", made_by="Alice", source_message_id="msg_3"
        ),
        Decision(
            decision="We will use Postgres for storage.", made_by="Bob", source_message_id="msg_7"
        ),
    ]
    deduped = dedup_decisions(decisions)
    assert len(deduped) == 1


def test_dedup_open_questions_keeps_answered_version_on_conflict():
    questions = [
        OpenQuestion(
            asker="Alice",
            question="Who owns the deploy script?",
            answered=False,
            source_message_id="msg_2",
        ),
        OpenQuestion(
            asker="Alice",
            question="Who owns the deploy script?",
            answered=True,
            source_message_id="msg_8",
        ),
    ]
    deduped = dedup_open_questions(questions)
    assert len(deduped) == 1
    assert deduped[0].answered is True


def test_merge_chunk_results_combines_and_dedupes_across_chunks():
    result_a = ExtractionResult(
        action_items=[ActionItem(owner="Bob", task="Review the PR", source_message_id="msg_1")],
        decisions=[],
        open_questions=[
            OpenQuestion(
                asker="Alice",
                question="Who will review the PR?",
                answered=False,
                source_message_id="msg_0",
            )
        ],
    )
    result_b = ExtractionResult(
        action_items=[ActionItem(owner="Bob", task="Review the PR", source_message_id="msg_1")],
        decisions=[
            Decision(decision="Bob will review the PR", made_by="Bob", source_message_id="msg_1")
        ],
        open_questions=[],
    )

    merged = merge_chunk_results([result_a, result_b])

    # action item deduped across chunks
    assert len(merged.action_items) == 1
    # decision kept
    assert len(merged.decisions) == 1
    # open question marked answered via heuristic match against later decision
    assert len(merged.open_questions) == 1
    assert merged.open_questions[0].answered is True


def test_merge_chunk_results_empty_input_returns_empty_lists():
    merged = merge_chunk_results([])
    assert merged.action_items == []
    assert merged.decisions == []
    assert merged.open_questions == []
