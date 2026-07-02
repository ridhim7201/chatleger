"""merge.py — Combine and deduplicate per-chunk extraction results for ChatLedger.

Public API
----------
    merge_results(results: List[ExtractionResult]) -> ExtractionResult

Takes the list of ExtractionResult objects produced by extractor.extract_chunk
(one per chunk) and returns a single consolidated ExtractionResult with:

  * Near-duplicate action items removed (same owner + similar task text).
  * Near-duplicate decisions removed (similar decision text).
  * Open questions marked answered=True when a later chunk's decisions or
    action items plausibly address the same topic (keyword-overlap heuristic —
    see _plausibly_answered() for caveats).

Similarity
----------
All text comparisons use difflib.SequenceMatcher.ratio(), which returns a
value in [0.0, 1.0].  The default dedup threshold is 0.85 — high enough to
collapse "Review the PR" vs "Review the PR." but low enough to keep
"Review the PR" and "Approve the PR" as separate items.

The answered-question heuristic uses a lower threshold (0.40) plus a
keyword-overlap check, intentionally lenient because question phrasings
and resolution phrasings rarely mirror each other word-for-word.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from schema import ActionItem, Decision, ExtractionResult, OpenQuestion

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

#: SequenceMatcher.ratio() threshold for deduplicating action items/decisions.
DEDUP_THRESHOLD: float = 0.85

#: ratio() threshold used by the answered-question heuristic.  Lower because
#: a question like "Who writes the blog post?" and an action item like "Bob:
#: write blog post" share few literal characters despite being the same topic.
ANSWER_RATIO_THRESHOLD: float = 0.40

#: Minimum number of meaningful words that must overlap between a question and
#: a potential answer before the ratio check is even applied.  Filters out
#: false positives driven purely by common stop-words.
ANSWER_MIN_KEYWORD_OVERLAP: int = 2

#: Words too common to count as meaningful keyword overlap.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "it",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "but",
        "be",
        "by",
        "we",
        "i",
        "you",
        "he",
        "she",
        "they",
        "this",
        "that",
        "with",
        "can",
        "will",
        "do",
        "did",
        "has",
        "have",
        "are",
        "was",
        "were",
        "who",
        "what",
        "when",
        "where",
        "how",
        "any",
        "all",
        "my",
        "our",
        "your",
    }
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ratio(a: str, b: str) -> float:
    """SequenceMatcher similarity ratio for two strings (case-insensitive)."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _keywords(text: str) -> set[str]:
    """Return the non-stop-word tokens from *text*, lower-cased."""
    tokens = set(text.lower().split())
    return tokens - _STOP_WORDS


def _plausibly_answered(
    question: OpenQuestion,
    action_items: list[ActionItem],
    decisions: list[Decision],
) -> bool:
    """Heuristic: decide whether *question* was addressed by any item.

    HEURISTIC — NOT GUARANTEED CORRECT.
    ------------------------------------
    Two tests are applied per candidate; Test 2 is only reached when Test 1
    has already found at least one shared keyword, preventing pure
    character-level coincidences from triggering a false positive.

    1. Keyword overlap: at least ANSWER_MIN_KEYWORD_OVERLAP non-stop-words
       from the question appear in the combined text of a potential answer.
       This catches "Who owns the blog post?" → "Bob: write the blog post"
       even though the exact phrasing differs.

    2. SequenceMatcher ratio >= ANSWER_RATIO_THRESHOLD, applied only when
       keyword_overlap >= 1.  Catches near-verbatim restatements with minor
       rephrasing, e.g. "Has the deploy key been rotated?" →
       "Alice: Rotate the staging deploy key", while ignoring pairs that
       share incidental characters but no meaningful words.

    IMPORTANT: gating ratio behind keyword_overlap >= 1 is the key design
    choice.  Without it, unrelated short strings can reach ratio ≈ 0.47
    from shared common words alone ("the", "to", etc.), producing false
    positives.  The gate ensures ratio is only a tiebreaker, never the
    sole signal.

    False negatives are still possible when a resolution is phrased with
    completely different vocabulary from the question.
    """
    q_keywords = _keywords(question.question)
    if not q_keywords:
        return False

    candidates: list[str] = [f"{a.owner} {a.task}" for a in action_items] + [
        d.decision for d in decisions
    ]

    for candidate_text in candidates:
        c_keywords = _keywords(candidate_text)
        keyword_overlap = len(q_keywords & c_keywords)

        # Test 1 — strong keyword overlap alone is sufficient
        if keyword_overlap >= ANSWER_MIN_KEYWORD_OVERLAP:
            return True

        # Test 2 — ratio as tiebreaker, but only when at least one keyword
        # already overlaps.  Gating here prevents incidental character
        # similarity between completely unrelated strings from firing.
        if (
            keyword_overlap >= 1
            and _ratio(question.question, candidate_text) >= ANSWER_RATIO_THRESHOLD
        ):
            return True

    return False


def _dedup_action_items(items: list[ActionItem]) -> list[ActionItem]:
    """Remove near-duplicate action items, keeping the first occurrence.

    Two items are considered duplicates when they share the same owner
    (case-insensitive) AND their task texts have a similarity ratio >=
    DEDUP_THRESHOLD.  Keeping the first occurrence preserves the
    source_message_id that appeared earliest in the conversation.
    """
    kept: list[ActionItem] = []
    for candidate in items:
        is_dup = any(
            candidate.owner.lower() == existing.owner.lower()
            and _ratio(candidate.task, existing.task) >= DEDUP_THRESHOLD
            for existing in kept
        )
        if not is_dup:
            kept.append(candidate)
    return kept


def _dedup_decisions(decisions: list[Decision]) -> list[Decision]:
    """Remove near-duplicate decisions, keeping the first occurrence.

    Two decisions are duplicates when their decision texts have a similarity
    ratio >= DEDUP_THRESHOLD (made_by is ignored — the same decision can be
    attributed slightly differently across overlapping chunks).
    """
    kept: list[Decision] = []
    for candidate in decisions:
        is_dup = any(
            _ratio(candidate.decision, existing.decision) >= DEDUP_THRESHOLD for existing in kept
        )
        if not is_dup:
            kept.append(candidate)
    return kept


def _dedup_questions(questions: list[OpenQuestion]) -> list[OpenQuestion]:
    """Remove near-duplicate open questions, keeping the answered version
    when duplicates disagree on answered status.

    Two questions are duplicates when they share the same asker
    (case-insensitive) AND their question texts have similarity >= DEDUP_THRESHOLD.
    """
    kept: list[OpenQuestion] = []
    for candidate in questions:
        dup_idx = next(
            (
                i
                for i, existing in enumerate(kept)
                if candidate.asker.lower() == existing.asker.lower()
                and _ratio(candidate.question, existing.question) >= DEDUP_THRESHOLD
            ),
            None,
        )
        if dup_idx is None:
            kept.append(candidate)
        elif candidate.answered and not kept[dup_idx].answered:
            # Prefer the version that carries answered=True
            kept[dup_idx] = candidate
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_results(results: list[ExtractionResult]) -> ExtractionResult:
    """Merge a list of per-chunk ExtractionResult objects into one.

    Steps
    -----
    1. Concatenate all items from every chunk into flat lists.
    2. Deduplicate action items (same owner + similar task).
    3. Deduplicate decisions (similar decision text).
    4. Run the answered-question heuristic across the *deduplicated* pool
       of action items and decisions (so a question is only marked answered
       if its topic genuinely appears in the consolidated results).
    5. Deduplicate open questions (same asker + similar question text),
       keeping the answered=True version when variants disagree.

    Parameters
    ----------
    results:
        Ordered list of ExtractionResult objects, one per chunk, in the
        order the chunks appear in the conversation (earliest first).

    Returns
    -------
    A single ExtractionResult with deduplicated, answer-annotated items.
    Returns an empty ExtractionResult when *results* is empty.
    """
    if not results:
        return ExtractionResult()

    # Step 1 — flatten
    all_items: list[ActionItem] = [i for r in results for i in r.action_items]
    all_decisions: list[Decision] = [d for r in results for d in r.decisions]
    all_questions: list[OpenQuestion] = [q for r in results for q in r.open_questions]

    # Step 2 — dedup actions and decisions
    deduped_items = _dedup_action_items(all_items)
    deduped_decisions = _dedup_decisions(all_decisions)

    # Step 3 — annotate answered status using deduplicated action/decision pool
    annotated_questions = [
        q.model_copy(
            update={
                "answered": q.answered or _plausibly_answered(q, deduped_items, deduped_decisions)
            }
        )
        for q in all_questions
    ]

    # Step 4 — dedup questions (keep answered version on conflict)
    deduped_questions = _dedup_questions(annotated_questions)

    return ExtractionResult(
        action_items=deduped_items,
        decisions=deduped_decisions,
        open_questions=deduped_questions,
    )


# ---------------------------------------------------------------------------
# __main__ — verify dedup and answered-heuristic with hand-built overlapping
# ExtractionResult objects
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Build two overlapping ExtractionResult objects ────────────────────
    #
    # chunk_a covers messages 0-14, chunk_b covers messages 12-19.
    # The three-message overlap means some items appear in both chunks with
    # trivially different phrasing — exactly the kind of near-duplicate that
    # the dedup logic must collapse.

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
            ),
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
            # Near-duplicate of chunk_a's first item — slight punctuation diff
            ActionItem(
                owner="Bob",
                task="Update the changelog before EOD.",  # trailing period
                deadline="today 5 pm",
                source_message_id="msg_12",
            ),
            # New item that didn't appear in chunk_a
            ActionItem(
                owner="Bob",
                task="Write the blog post announcement",
                deadline="June 14th",
                source_message_id="msg_13",
            ),
            # The deploy-key action is now explicitly assigned — close phrasing
            ActionItem(
                owner="Alice",
                task="Rotate the staging deploy key",
                deadline="ASAP",
                source_message_id="msg_14",
            ),
        ],
        decisions=[
            # Near-duplicate of chunk_a's release-date decision — extra word
            Decision(
                decision="Release date is June 15th.",
                made_by="Team",
                source_message_id="msg_12",
            ),
            # New decision about the blog post review
            Decision(
                decision="Carol reviews the blog post before it is published",
                made_by="Carol",
                source_message_id="msg_14",
            ),
        ],
        open_questions=[
            # Same question as chunk_a — should be deduped
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

    # ── Run the merge ──────────────────────────────────────────────────────
    merged = merge_results([chunk_a, chunk_b])

    # ── Display results ────────────────────────────────────────────────────
    SEP = "─" * 62

    print("ChatLedger — merge.py verification")
    print("=" * 62)
    print("Input chunks : 2  (simulating window=15, overlap=3)")
    print(
        f"  chunk_a: {len(chunk_a.action_items)} actions, "
        f"{len(chunk_a.decisions)} decisions, "
        f"{len(chunk_a.open_questions)} questions"
    )
    print(
        f"  chunk_b: {len(chunk_b.action_items)} actions, "
        f"{len(chunk_b.decisions)} decisions, "
        f"{len(chunk_b.open_questions)} questions"
    )
    print()

    print(f"Action Items after merge  ({len(merged.action_items)})")
    print(SEP)
    for item in merged.action_items:
        dl = f"  deadline: {item.deadline}" if item.deadline else ""
        print(f"  [{item.source_message_id}]  {item.owner}: {item.task}{dl}")

    print()
    print(f"Decisions after merge  ({len(merged.decisions)})")
    print(SEP)
    for dec in merged.decisions:
        print(f"  [{dec.source_message_id}]  {dec.made_by}: {dec.decision}")

    print()
    print(f"Open Questions after merge  ({len(merged.open_questions)})")
    print(SEP)
    for q in merged.open_questions:
        status = "✓ answered" if q.answered else "○ open"
        print(f"  [{q.source_message_id}]  {q.asker}: {q.question}  [{status}]")

    # ── Assertions ────────────────────────────────────────────────────────
    print()
    print("Invariant assertions")
    print(SEP)

    # 1. Bob's changelog action should appear exactly once
    bob_changelog = [
        i for i in merged.action_items if i.owner == "Bob" and "changelog" in i.task.lower()
    ]
    assert len(bob_changelog) == 1, f"Expected 1 changelog action, got {len(bob_changelog)}"
    print("✓  Bob's changelog action deduplicated to 1 item")

    # 2. Alice's deploy-key action should appear exactly once
    alice_key = [
        i for i in merged.action_items if i.owner == "Alice" and "deploy key" in i.task.lower()
    ]
    assert len(alice_key) == 1, f"Expected 1 deploy-key action, got {len(alice_key)}"
    print("✓  Alice's deploy-key action deduplicated to 1 item")

    # 3. Bob's blog-post action should survive (not a dup of anything)
    bob_blog = [
        i for i in merged.action_items if i.owner == "Bob" and "blog post" in i.task.lower()
    ]
    assert len(bob_blog) == 1, f"Expected 1 blog-post action, got {len(bob_blog)}"
    print("✓  Bob's blog-post action kept (distinct item)")

    # 4. Release-date decision appears exactly once
    release = [d for d in merged.decisions if "june 15" in d.decision.lower()]
    assert len(release) == 1, f"Expected 1 release decision, got {len(release)}"
    print("✓  Release-date decision deduplicated to 1 item")

    # 5. Blog-post-review decision kept (distinct)
    blog_review = [d for d in merged.decisions if "blog post" in d.decision.lower()]
    assert len(blog_review) == 1, f"Expected 1 blog-review decision, got {len(blog_review)}"
    print("✓  Blog-post-review decision kept (distinct item)")

    # 6. Carol's blog-post question deduplicated to 1
    carol_q = [
        q for q in merged.open_questions if q.asker == "Carol" and "blog post" in q.question.lower()
    ]
    assert len(carol_q) == 1, f"Expected 1 Carol blog-post question, got {len(carol_q)}"
    print("✓  Carol's blog-post question deduplicated to 1 item")

    # 7. Carol's question should be marked answered (Bob's blog-post action covers it)
    assert carol_q[0].answered, "Carol's blog-post question should be answered"
    print("✓  Carol's blog-post question marked answered=True (keyword overlap with Bob's action)")

    # 8. Alice's deploy-key question should be answered (Alice's action covers it)
    alice_q = [
        q
        for q in merged.open_questions
        if q.asker == "Alice" and "deploy key" in q.question.lower()
    ]
    assert len(alice_q) == 1
    assert alice_q[0].answered, "Alice's deploy-key question should be answered"
    print("✓  Alice's deploy-key question marked answered=True")

    # 9. Bob's blog-post deadline question — may or may not be answered
    #    (just check it's present; don't assert the answered value)
    bob_q = [q for q in merged.open_questions if q.asker == "Bob"]
    assert len(bob_q) == 1, f"Expected 1 Bob question, got {len(bob_q)}"
    print(f"✓  Bob's deadline question present (answered={bob_q[0].answered})")

    print()
    print("All assertions passed.")