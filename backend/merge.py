"""Merge chunk-level extraction results into deduplicated final lists."""

from __future__ import annotations

from difflib import SequenceMatcher

from schema import ActionItem, Decision, ExtractionResult, OpenQuestion

SIMILARITY_THRESHOLD = 0.85


def _similar(a: str, b: str, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() >= threshold


def dedup_action_items(items: list[ActionItem]) -> list[ActionItem]:
    kept: list[ActionItem] = []
    for item in items:
        if any(
            item.owner.lower() == existing.owner.lower() and _similar(item.task, existing.task)
            for existing in kept
        ):
            continue
        kept.append(item)
    return kept


def dedup_decisions(items: list[Decision]) -> list[Decision]:
    kept: list[Decision] = []
    for item in items:
        if any(_similar(item.decision, existing.decision) for existing in kept):
            continue
        kept.append(item)
    return kept


def _mark_answered(
    questions: list[OpenQuestion], items: list[ActionItem], decisions: list[Decision]
) -> list[OpenQuestion]:
    """Heuristic: a question is answered if a later-occurring decision or
    action item shares enough textual overlap with the question (same
    asker mentioned, or strong topical similarity)."""
    updated: list[OpenQuestion] = []
    candidate_texts: list[tuple[str, str]] = [(d.decision, d.made_by) for d in decisions] + [
        (i.task, i.owner) for i in items
    ]

    for q in questions:
        answered = q.answered
        if not answered:
            for text, _actor in candidate_texts:
                if _similar(q.question, text, threshold=0.4):
                    answered = True
                    break
        updated.append(q.model_copy(update={"answered": answered}))
    return updated


def dedup_open_questions(questions: list[OpenQuestion]) -> list[OpenQuestion]:
    kept: list[OpenQuestion] = []
    for q in questions:
        existing_idx = next(
            (
                i
                for i, k in enumerate(kept)
                if q.asker.lower() == k.asker.lower() and _similar(q.question, k.question)
            ),
            None,
        )
        if existing_idx is None:
            kept.append(q)
        elif q.answered and not kept[existing_idx].answered:
            # Prefer the answered version if duplicates disagree.
            kept[existing_idx] = q
    return kept


def merge_chunk_results(results: list[ExtractionResult]) -> ExtractionResult:
    """Combine all per-chunk ExtractionResult objects into one deduplicated
    final ExtractionResult."""
    all_items: list[ActionItem] = [i for r in results for i in r.action_items]
    all_decisions: list[Decision] = [d for r in results for d in r.decisions]
    all_questions: list[OpenQuestion] = [q for r in results for q in r.open_questions]

    deduped_items = dedup_action_items(all_items)
    deduped_decisions = dedup_decisions(all_decisions)

    questions_with_answers = _mark_answered(all_questions, deduped_items, deduped_decisions)
    deduped_questions = dedup_open_questions(questions_with_answers)

    return ExtractionResult(
        action_items=deduped_items,
        decisions=deduped_decisions,
        open_questions=deduped_questions,
    )
