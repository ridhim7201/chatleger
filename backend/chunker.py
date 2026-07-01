"""chunker.py — Overlapping message-window chunking for ChatLedger.

Public API
----------
    chunk_messages(
        messages:    List[dict],
        window_size: int = 15,
        overlap:     int = 3,
    ) -> List[List[dict]]

Each returned inner list is a *chunk*: a slice of the original message list.
Chunks overlap so that context that straddles a boundary (e.g. a question
in chunk N that gets answered in chunk N+1) is visible to both extraction
calls.

Overlap mechanics
-----------------
Given window_size=5 and overlap=2, a 12-message list produces:

    chunk_0:  msg_0  – msg_4    (indices 0..4)
    chunk_1:  msg_3  – msg_7    (indices 3..7)   ← starts 2 before end of chunk_0
    chunk_2:  msg_6  – msg_10   (indices 6..10)
    chunk_3:  msg_9  – msg_11   (indices 9..11)  ← last chunk, may be shorter

The step between chunk starts is  step = window_size - overlap.
The final chunk always runs to the end of the list; it will be shorter than
window_size when len(messages) % step != 0.

Constraints (enforced with ValueError)
---------------------------------------
* window_size >= 1
* overlap >= 0
* overlap < window_size   (otherwise the window never advances)
"""

from __future__ import annotations


def chunk_messages(
    messages: list[dict],
    window_size: int = 15,
    overlap: int = 3,
) -> list[list[dict]]:
    """Split *messages* into overlapping windows.

    Parameters
    ----------
    messages:
        Ordered list of message dicts as returned by ``parser.parse_chat``.
        Each dict must have at least a ``"message_id"`` key, but this
        function is otherwise schema-agnostic.
    window_size:
        Maximum number of messages per chunk.
    overlap:
        How many messages at the *end* of chunk N are repeated at the
        *start* of chunk N+1, ensuring cross-boundary context is not lost.

    Returns
    -------
    A list of chunks.  Each chunk is a list of message dicts (references
    into the original list — no copying of dict contents).  Returns an
    empty list when *messages* is empty.

    Raises
    ------
    ValueError
        If window_size < 1, overlap < 0, or overlap >= window_size.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= window_size:
        raise ValueError(
            f"overlap ({overlap}) must be < window_size ({window_size}); "
            "otherwise the window never advances"
        )

    if not messages:
        return []

    step = window_size - overlap
    chunks: list[list[dict]] = []
    start = 0

    while start < len(messages):
        end = min(start + window_size, len(messages))
        chunks.append(messages[start:end])
        if end == len(messages):
            break  # reached the final message — stop
        start += step

    return chunks


# ---------------------------------------------------------------------------
# __main__ — verify overlap logic with 20 dummy messages
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Build 20 fake message dicts ──────────────────────────────────────
    N = 20
    dummy = [
        {
            "message_id": f"msg_{i}",
            "timestamp": f"01/06/24, 09:{i:02d}:00",
            "sender": ["Alice", "Bob", "Carol"][i % 3],
            "text": f"Message number {i}.",
        }
        for i in range(N)
    ]

    # ── Run chunker with default params (window=15, overlap=3) ───────────
    WINDOW, OVERLAP = 15, 3
    chunks = chunk_messages(dummy, window_size=WINDOW, overlap=OVERLAP)

    STEP = WINDOW - OVERLAP
    print(f"Messages : {N}")
    print(f"window   : {WINDOW}")
    print(f"overlap  : {OVERLAP}")
    print(f"step     : {STEP}  (= window - overlap)")
    print(f"Chunks   : {len(chunks)}")
    print()

    # ── Print chunk boundaries ────────────────────────────────────────────
    header = f"{'chunk':<8}  {'first msg':<10}  {'last msg':<10}  {'size':>4}  overlap with prev"
    print(header)
    print("-" * len(header))

    for idx, chunk in enumerate(chunks):
        first = chunk[0]["message_id"]
        last = chunk[-1]["message_id"]
        size = len(chunk)

        if idx == 0:
            overlap_note = "—  (first chunk)"
        else:
            prev_chunk = chunks[idx - 1]
            # The overlapping messages are the first `overlap` rows of this
            # chunk, which should equal the last `overlap` rows of the prev.
            shared = chunk[:OVERLAP]
            shared_ids = [m["message_id"] for m in shared]
            overlap_note = f"shares {shared_ids}"

        print(f"chunk_{idx:<2}  {first:<10}  {last:<10}  {size:>4}  {overlap_note}")

    # ── Verify the overlap invariant programmatically ─────────────────────
    print()
    print(
        "Invariant check (each chunk's first `overlap` msgs == prev chunk's last `overlap` msgs):"
    )
    all_ok = True
    for idx in range(1, len(chunks)):
        tail_of_prev = [m["message_id"] for m in chunks[idx - 1][-OVERLAP:]]
        head_of_curr = [m["message_id"] for m in chunks[idx][:OVERLAP]]
        ok = tail_of_prev == head_of_curr
        status = "✓" if ok else "✗ MISMATCH"
        print(
            f"  chunk_{idx - 1} tail {tail_of_prev}  ==  chunk_{idx} head {head_of_curr}  {status}"
        )
        if not ok:
            all_ok = False

    print()
    print("All invariants satisfied." if all_ok else "ERROR: invariant violated — check the logic.")

    # ── Also demonstrate a small case (window=5, overlap=2) side-by-side ─
    print()
    print("─" * 60)
    print("Small example  window=5  overlap=2  (12 messages)")
    print("─" * 60)
    small = dummy[:12]
    small_chunks = chunk_messages(small, window_size=5, overlap=2)
    for idx, chunk in enumerate(small_chunks):
        ids = [m["message_id"] for m in chunk]
        print(f"  chunk_{idx}: {ids}")