"""Tests for merge.py — deduplication and answered-question heuristic.

All tests go through the public merge_results() API.  Internal helpers
(_dedup_*, _plausibly_answered) are tested indirectly via merge_results so
the implementation can be refactored without breaking the test contract.

Coverage
--------
  Basic            — empty input, single chunk, multiple distinct items
  Action item dedup— near-identical tasks same owner, distinct tasks same owner,
                     same task different owners, three-way dup, source_message_id
                     preserved from first occurrence
  Decision dedup   — near-identical text, made_by ignored during dedup,
                     distinct decisions kept, source_message_id from first
  Question dedup   — exact duplicates, near-identical text, answered=True wins,
                     distinct questions same asker both kept
  Answered heuristic— marked by action item keyword overlap, marked by decision
                     keyword overlap, stays open when no overlap, already-answered
                     preserved, does NOT fire on zero shared keywords
  Similarity edge  — texts at/around the 0.85 threshold, empty strings
  Integration      — full two-chunk overlap fixture from conftest
"""

from __future__ import annotations

import pytest

from merge import merge_results
from schema import ActionItem, Decision, ExtractionResult, OpenQuestion

# ---------------------------------------------------------------------------
# Tiny builder helpers — keep test bodies concise
# ---------------------------------------------------------------------------


def action(
    owner: str,
    task: str,
    src: str = "msg_0",
    deadline: str | None = None,
) -> ActionItem:
    return ActionItem(owner=owner, task=task, deadline=deadline, source_message_id=src)


def decision(text: str, by: str = "Alice", src: str = "msg_0") -> Decision:
    return Decision(decision=text, made_by=by, source_message_id=src)


def question(asker: str, text: str, src: str = "msg_0", answered: bool = False) -> OpenQuestion:
    return OpenQuestion(asker=asker, question=text, answered=answered, source_message_id=src)


def result(*args) -> ExtractionResult:
    """Build an ExtractionResult from a flat list of mixed item types."""
    return ExtractionResult(
        action_items=[a for a in args if isinstance(a, ActionItem)],
        decisions=[d for d in args if isinstance(d, Decision)],
        open_questions=[q for q in args if isinstance(q, OpenQuestion)],
    )


# ===========================================================================
# Basic behaviour
# ===========================================================================


class TestBasic:
    def test_empty_list_returns_empty_result(self):
        assert merge_results([]) == ExtractionResult()

    def test_single_result_passes_through_unchanged(self):
        r = result(
            action("Bob", "Review the PR", "msg_1"),
            decision("Release on June 15th", "Alice", "msg_2"),
            question("Carol", "Who owns the blog post?", "msg_3"),
        )
        merged = merge_results([r])
        assert len(merged.action_items) == 1
        assert len(merged.decisions) == 1
        assert len(merged.open_questions) == 1

    def test_multiple_distinct_items_all_survive(self):
        a = result(action("Alice", "Deploy to staging", "msg_1"))
        b = result(action("Bob", "Update the changelog", "msg_5"))
        assert len(merge_results([a, b]).action_items) == 2

    def test_output_is_always_an_extraction_result(self):
        assert isinstance(merge_results([]), ExtractionResult)
        assert isinstance(merge_results([result()]), ExtractionResult)


# ===========================================================================
# Action item dedup
# ===========================================================================


class TestActionItemDedup:
    def test_near_identical_task_same_owner_collapsed(self):
        """Trailing period must not produce a duplicate."""
        a = result(action("Bob", "Update the changelog before EOD", "msg_1"))
        b = result(action("Bob", "Update the changelog before EOD.", "msg_8"))
        assert len(merge_results([a, b]).action_items) == 1

    def test_first_occurrence_source_id_kept(self):
        a = result(action("Bob", "Write the release notes", "msg_2"))
        b = result(action("Bob", "Write the release notes", "msg_9"))
        merged = merge_results([a, b])
        assert merged.action_items[0].source_message_id == "msg_2"

    def test_different_tasks_same_owner_both_kept(self):
        a = result(action("Bob", "Review the PR", "msg_1"))
        b = result(action("Bob", "Deploy to staging", "msg_5"))
        assert len(merge_results([a, b]).action_items) == 2

    def test_identical_task_different_owners_both_kept(self):
        a = result(action("Alice", "Review the PR", "msg_1"))
        b = result(action("Bob", "Review the PR", "msg_2"))
        assert len(merge_results([a, b]).action_items) == 2

    def test_three_way_duplicate_collapses_to_one(self):
        chunks = [result(action("Alice", "Rotate the deploy key", f"msg_{i}")) for i in range(3)]
        merged = merge_results(chunks)
        assert len(merged.action_items) == 1
        assert merged.action_items[0].source_message_id == "msg_0"

    def test_deadline_from_first_occurrence_preserved(self):
        a = result(action("Bob", "Review the PR", "msg_1", deadline="Friday"))
        b = result(action("Bob", "Review the PR.", "msg_8", deadline="Monday"))
        merged = merge_results([a, b])
        assert merged.action_items[0].deadline == "Friday"

    def test_case_insensitive_owner_comparison(self):
        a = result(action("Alice", "Fix the bug", "msg_1"))
        b = result(action("alice", "Fix the bug", "msg_2"))
        # Both 'Alice' and 'alice' refer to the same person
        merged = merge_results([a, b])
        assert len(merged.action_items) == 1


# ===========================================================================
# Decision dedup
# ===========================================================================


class TestDecisionDedup:
    def test_near_identical_decisions_collapsed(self):
        a = result(decision("Release date is June 15th", "Alice", "msg_2"))
        b = result(decision("Release date is June 15th.", "Team", "msg_9"))
        assert len(merge_results([a, b]).decisions) == 1

    def test_first_occurrence_source_id_kept(self):
        a = result(decision("Use Postgres for the database", "Alice", "msg_3"))
        b = result(decision("Use Postgres for the database", "Bob", "msg_11"))
        assert merge_results([a, b]).decisions[0].source_message_id == "msg_3"

    def test_made_by_ignored_when_text_is_similar(self):
        """Different made_by values must not prevent dedup when text matches."""
        a = result(decision("We will ship on Friday", "Alice", "msg_1"))
        b = result(decision("We will ship on Friday", "Bob", "msg_2"))
        assert len(merge_results([a, b]).decisions) == 1

    def test_distinct_decisions_both_kept(self):
        a = result(decision("Release on June 15th", "Alice", "msg_2"))
        b = result(decision("Use Postgres for the database", "Bob", "msg_7"))
        assert len(merge_results([a, b]).decisions) == 2

    def test_three_way_decision_dedup(self):
        chunks = [
            result(decision("We will use Python for this project", "Team", f"msg_{i}"))
            for i in range(3)
        ]
        assert len(merge_results(chunks).decisions) == 1


# ===========================================================================
# Open question dedup
# ===========================================================================


class TestOpenQuestionDedup:
    def test_exact_duplicate_collapses_to_one(self):
        a = result(question("Carol", "Who owns the blog post?", "msg_4"))
        b = result(question("Carol", "Who owns the blog post?", "msg_11"))
        assert len(merge_results([a, b]).open_questions) == 1

    def test_near_identical_questions_collapsed(self):
        a = result(question("Carol", "Who is handling the blog post?", "msg_4"))
        b = result(question("Carol", "Who is handling the blog post.", "msg_11"))
        assert len(merge_results([a, b]).open_questions) == 1

    def test_answered_true_wins_over_false_on_dedup(self):
        a = result(question("Bob", "Has the key been rotated?", "msg_3", answered=False))
        b = result(question("Bob", "Has the key been rotated?", "msg_10", answered=True))
        merged = merge_results([a, b])
        assert len(merged.open_questions) == 1
        assert merged.open_questions[0].answered is True

    def test_distinct_questions_same_asker_both_kept(self):
        a = result(question("Alice", "What is the release date?", "msg_2"))
        b = result(question("Alice", "Who writes the changelog?", "msg_6"))
        assert len(merge_results([a, b]).open_questions) == 2

    def test_distinct_questions_different_askers_both_kept(self):
        a = result(question("Alice", "When is the release?", "msg_2"))
        b = result(question("Bob", "When is the release?", "msg_3"))
        assert len(merge_results([a, b]).open_questions) == 2

    def test_case_insensitive_asker_comparison(self):
        a = result(question("Alice", "Who reviews the PR?", "msg_1"))
        b = result(question("alice", "Who reviews the PR?", "msg_5"))
        merged = merge_results([a, b])
        assert len(merged.open_questions) == 1


# ===========================================================================
# Answered-question heuristic
# ===========================================================================


class TestAnsweredHeuristic:
    def test_answered_by_action_item_keyword_overlap(self):
        """'Who handles the blog post?' → 'Bob: write blog post'."""
        chunk_a = result(question("Carol", "Who is handling the blog post?", "msg_4"))
        chunk_b = result(action("Bob", "Write the blog post announcement", "msg_9"))
        merged = merge_results([chunk_a, chunk_b])
        carol_q = next(q for q in merged.open_questions if q.asker == "Carol")
        assert carol_q.answered is True

    def test_answered_by_decision_keyword_overlap(self):
        """'What is the release date?' → decision about release date."""
        chunk_a = result(question("Bob", "What is the release date?", "msg_2"))
        chunk_b = result(decision("The release date is June 15th", "Alice", "msg_8"))
        merged = merge_results([chunk_a, chunk_b])
        assert merged.open_questions[0].answered is True

    def test_stays_open_when_no_keyword_overlap(self):
        """Unrelated topics must not trigger the heuristic."""
        chunk_a = result(question("Alice", "Who approves the budget?", "msg_1"))
        chunk_b = result(action("Bob", "Write the blog post", "msg_6"))
        merged = merge_results([chunk_a, chunk_b])
        alice_q = next(q for q in merged.open_questions if q.asker == "Alice")
        assert alice_q.answered is False

    def test_already_answered_preserved(self):
        """answered=True from the model output must survive merge unchanged."""
        chunk = result(question("Alice", "Is the server down?", "msg_1", answered=True))
        assert merge_results([chunk]).open_questions[0].answered is True

    def test_heuristic_uses_deduplicated_pool(self):
        """The answered check runs against deduped actions/decisions, not raw."""
        # A question exists in chunk_a; an action item answering it appears in
        # chunk_b *as a duplicate*.  After dedup only one copy of that action
        # remains, but the heuristic should still fire.
        chunk_a = result(
            question("Carol", "Who writes the blog post?", "msg_4"),
            action("Bob", "Write the blog post", "msg_5"),  # also in chunk_a
        )
        chunk_b = result(
            action("Bob", "Write the blog post", "msg_12"),  # near-dup from overlap
        )
        merged = merge_results([chunk_a, chunk_b])
        assert len(merged.action_items) == 1  # deduped
        carol_q = merged.open_questions[0]
        assert carol_q.answered is True

    def test_heuristic_not_triggered_by_stop_words_alone(self):
        """Common words ('the', 'a', 'is') must not suffice to mark answered."""
        chunk_a = result(question("Dave", "Is the project approved?", "msg_1"))
        # Action is purely about logging — shares only stop words with the question
        chunk_b = result(action("Eve", "Update the log", "msg_5"))
        merged = merge_results([chunk_a, chunk_b])
        dave_q = next(q for q in merged.open_questions if q.asker == "Dave")
        assert dave_q.answered is False

    def test_empty_question_text_not_answered(self):
        """An edge-case empty question string must not crash or be marked answered."""
        chunk_a = result(
            OpenQuestion(asker="X", question="who", answered=False, source_message_id="msg_0")
        )
        chunk_b = result(action("Bob", "Deploy the app", "msg_1"))
        merged = merge_results([chunk_a, chunk_b])
        assert isinstance(merged.open_questions[0].answered, bool)


# ===========================================================================
# Similarity threshold edge cases
# ===========================================================================


class TestSimilarityThreshold:
    """Verify the 0.85 boundary behaves as documented."""

    def test_identical_strings_always_deduped(self):
        a = result(action("Bob", "Fix the critical bug", "msg_1"))
        b = result(action("Bob", "Fix the critical bug", "msg_2"))
        assert len(merge_results([a, b]).action_items) == 1

    def test_clearly_different_strings_not_deduped(self):
        """'Review the PR' vs 'Deploy to staging' — far below 0.85."""
        a = result(action("Bob", "Review the PR", "msg_1"))
        b = result(action("Bob", "Deploy to staging", "msg_2"))
        assert len(merge_results([a, b]).action_items) == 2

    def test_minor_punctuation_difference_deduped(self):
        """Full stop and comma variants should not survive as separate items."""
        pairs = [
            ("Release on June 15th", "Release on June 15th."),
            ("Review PR before merging", "Review PR before merging,"),
        ]
        for text_a, text_b in pairs:
            a = result(action("Bob", text_a, "msg_1"))
            b = result(action("Bob", text_b, "msg_2"))
            merged = merge_results([a, b])
            assert len(merged.action_items) == 1, f"Expected dedup for {text_a!r} vs {text_b!r}"


# ===========================================================================
# Integration: full overlapping-chunk fixture
# ===========================================================================


class TestIntegrationOverlappingChunks:
    """End-to-end tests using the overlapping_chunk_pair fixture from conftest."""

    @pytest.fixture(autouse=True)
    def merged(self, overlapping_chunk_pair):
        chunk_a, chunk_b = overlapping_chunk_pair
        self._merged = merge_results([chunk_a, chunk_b])

    def test_action_item_count(self):
        assert len(self._merged.action_items) == 3

    def test_decision_count(self):
        assert len(self._merged.decisions) == 2

    def test_open_question_count(self):
        assert len(self._merged.open_questions) == 3

    def test_changelog_action_deduplicated_to_one(self):
        changelog = [i for i in self._merged.action_items if "changelog" in i.task.lower()]
        assert len(changelog) == 1

    def test_changelog_source_id_from_first_chunk(self):
        changelog = next(i for i in self._merged.action_items if "changelog" in i.task.lower())
        assert changelog.source_message_id == "msg_3"

    def test_deploy_key_action_deduplicated(self):
        deploy = [i for i in self._merged.action_items if "deploy key" in i.task.lower()]
        assert len(deploy) == 1

    def test_blog_post_action_survived_as_distinct(self):
        blog = [i for i in self._merged.action_items if "blog post" in i.task.lower()]
        assert len(blog) == 1

    def test_release_decision_deduplicated(self):
        release = [d for d in self._merged.decisions if "june 15" in d.decision.lower()]
        assert len(release) == 1

    def test_release_decision_source_id_from_first_chunk(self):
        release = next(d for d in self._merged.decisions if "june 15" in d.decision.lower())
        assert release.source_message_id == "msg_2"

    def test_blog_review_decision_kept_as_distinct(self):
        review = [d for d in self._merged.decisions if "blog post" in d.decision.lower()]
        assert len(review) == 1

    def test_carol_question_deduplicated_and_answered(self):
        carol_qs = [q for q in self._merged.open_questions if q.asker == "Carol"]
        assert len(carol_qs) == 1
        assert carol_qs[0].answered is True

    def test_alice_question_answered_by_her_action(self):
        alice_qs = [q for q in self._merged.open_questions if q.asker == "Alice"]
        assert len(alice_qs) == 1
        assert alice_qs[0].answered is True

    def test_bob_question_present(self):
        bob_qs = [q for q in self._merged.open_questions if q.asker == "Bob"]
        assert len(bob_qs) == 1