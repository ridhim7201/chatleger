"""parser.py — WhatsApp chat export ingestion for ChatLedger.

Public API
----------
    parse_chat(filepath: str) -> List[dict]

Each returned dict has:
    {
        "message_id": "msg_0",   # sequential, zero-based
        "timestamp":  "01/06/24, 09:15:32",
        "sender":     "Alice",
        "text":       "Hey, can we reschedule: Tuesday works better",
    }

Edge cases handled
------------------
* Multi-line messages — continuation lines have no timestamp prefix and are
  appended to the previous message's text.
* System messages — lines that have a timestamp but no "Sender: text" colon
  (e.g. "Messages and calls are end-to-end encrypted.") are stored with
  sender="__system__" so callers can filter them out.
* Colons in message text — we split only on the FIRST colon after the sender
  field, so "Note: don't forget" parses correctly.
* Empty lines — skipped.
* UTF-8 BOM — stripped.
* Missing/malformed header lines at the very start — silently skipped (no
  previous message to attach them to).
* Both bracket and bracket-less export variants:
    [DD/MM/YY, HH:MM:SS] Sender: text
     DD/MM/YY, HH:MM - Sender: text
"""

from __future__ import annotations

import re
import sys

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches the timestamp+sender header in both bracket and bracket-less forms.
# Group layout:
#   ts    — full timestamp string (date + time, without brackets/dashes)
#   date  — DD/MM/YY or DD/MM/YYYY
#   time  — HH:MM or HH:MM:SS, optionally followed by AM/PM
#   rest  — everything after the closing "] " or " - ", which is either
#           "Sender: text" (normal) or a bare system notice with no colon.
_HEADER_RE = re.compile(
    r"^\[?"  # optional opening bracket
    r"(?P<date>\d{1,2}/\d{1,2}/\d{2,4})"  # date
    r",\s*"  # comma separator
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\u202f?[APap][Mm])?)"  # time (NNBSP before AM/PM)
    r"\]?"  # optional closing bracket
    r"[ \t]*[-\u2009]?[ \t]*"  # optional " - " separator (thin space too)
    r"(?P<rest>.+)$",  # everything else
    re.UNICODE,
)

# Splits "Sender Name: message text" on the FIRST colon only.
# Sender names can be anything up to 80 chars that isn't itself a colon.
# This naturally handles colons inside the message body.
_SENDER_RE = re.compile(r"^(?P<sender>[^:]{1,80}):\s*(?P<text>.*)$", re.DOTALL)

# Known system-message suffixes that never have a "Sender: text" structure.
# We check these as a fast pre-filter before falling back to the regex.
_SYSTEM_SUBSTRINGS = (
    "end-to-end encrypted",
    "joined using this group",
    "left",
    "was added",
    "changed the group",
    "changed their phone number",
    "created group",
    "Messages and calls",
    "<Media omitted>",
    "This message was deleted",
)


def _is_system_rest(rest: str) -> bool:
    """Return True if the post-timestamp text looks like a system notice."""
    # A system notice has no "Sender: body" structure.
    # Quick path: check known substrings first.
    for frag in _SYSTEM_SUBSTRINGS:
        if frag in rest:
            return True
    # Slow path: no colon at all → definitely system.
    return ":" not in rest


# ---------------------------------------------------------------------------
# Sample export (15 messages, covers all edge cases)
# ---------------------------------------------------------------------------

SAMPLE_EXPORT = """\
[01/06/24, 09:00:00] Alice: Good morning everyone! Ready for the standup?
[01/06/24, 09:00:45] Bob: Morning! Give me two minutes.
[01/06/24, 09:01:10] Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.
[01/06/24, 09:02:00] Carol: Sure, I'll start.
Yesterday I finished the auth module.
Today I'm moving on to the dashboard.
No blockers right now.
[01/06/24, 09:03:15] Alice: Nice. Bob, what about the API?
[01/06/24, 09:03:45] Bob: Still blocked on staging: the deploy key expired.
[01/06/24, 09:04:00] Alice: I'll sort that — action item for me.
[01/06/24, 09:04:30] Carol: Should we move the retro? Tuesday works better for me.
[01/06/24, 09:04:55] Bob: +1, Tuesday is fine.
[01/06/24, 09:05:10] Alice: Agreed. Decision: retro moves to Tuesday 3 pm.
[01/06/24, 09:05:30] Carol: Works for me!
[01/06/24, 09:06:00] Bob: Quick question: who owns the Figma handoff?
[01/06/24, 09:06:20] Alice: That's Dave — he said he'd send it by EOD.
[01/06/24, 09:06:45] Carol: Perfect. I've set a reminder: check Figma at 5 pm.
[01/06/24, 09:07:00] Alice: Great standup everyone. Talk later!
"""


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------


def parse_chat(filepath: str) -> list[dict]:
    """Parse a WhatsApp-exported .txt file into a list of message dicts.

    Parameters
    ----------
    filepath:
        Path to the .txt export file (UTF-8, with or without BOM).

    Returns
    -------
    List of dicts, each with keys:
        message_id, timestamp, sender, text
    """
    with open(filepath, encoding="utf-8-sig") as fh:  # utf-8-sig strips BOM
        raw = fh.read()

    return _parse_raw(raw)


def _parse_raw(raw: str) -> list[dict]:
    """Internal: parse already-loaded text.  Separated for testability."""
    # Normalise line endings
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    rows: list[dict] = []
    seq = 0

    for line in raw.split("\n"):
        if not line.strip():
            continue  # blank line — skip

        header_match = _HEADER_RE.match(line)

        if header_match:
            # ---- New message boundary ----
            date = header_match.group("date")
            time = header_match.group("time")
            timestamp = f"{date}, {time}"
            rest = header_match.group("rest").strip()

            if _is_system_rest(rest):
                sender = "__system__"
                text = rest
            else:
                sender_match = _SENDER_RE.match(rest)
                if sender_match:
                    sender = sender_match.group("sender").strip()
                    text = sender_match.group("text").strip()
                else:
                    # Has a colon somewhere but didn't parse — treat as system
                    sender = "__system__"
                    text = rest

            rows.append(
                {
                    "message_id": f"msg_{seq}",
                    "timestamp": timestamp,
                    "sender": sender,
                    "text": text,
                }
            )
            seq += 1

        else:
            # ---- Continuation line (no timestamp) ----
            if rows:
                # Append to the most recent message, preserving newlines
                rows[-1]["text"] = (
                    rows[-1]["text"] + "\n" + line.rstrip() if rows[-1]["text"] else line.rstrip()
                )
            # else: orphan continuation with no prior message — silently skip

    return rows


# ---------------------------------------------------------------------------
# __main__ — generate sample file, parse it, print results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile

    sample_path = os.path.join(tempfile.gettempdir(), "chatledger_sample.txt")

    # Write the sample export
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_EXPORT)

    print(f"Sample file written to: {sample_path}\n")

    # Parse it
    rows = parse_chat(sample_path)

    # Pretty-print
    col_w = {"id": 8, "ts": 19, "sender": 12}
    header = (
        f"{'msg_id':<{col_w['id']}}  "
        f"{'timestamp':<{col_w['ts']}}  "
        f"{'sender':<{col_w['sender']}}  "
        f"text"
    )
    print(header)
    print("-" * 90)

    for row in rows:
        # Truncate long / multi-line text for display
        display_text = row["text"].replace("\n", " ↵ ")
        if len(display_text) > 55:
            display_text = display_text[:52] + "..."
        print(
            f"{row['message_id']:<{col_w['id']}}  "
            f"{row['timestamp']:<{col_w['ts']}}  "
            f"{row['sender']:<{col_w['sender']}}  "
            f"{display_text}"
        )

    print(f"\n{len(rows)} messages parsed.")

    # Verify edge cases programmatically
    print("\n── Edge case verification ──")

    # System message
    system_rows = [r for r in rows if r["sender"] == "__system__"]
    print(f"System messages:       {len(system_rows)}  {[r['message_id'] for r in system_rows]}")

    # Multi-line message (Carol's standup update)
    multiline = [r for r in rows if "\n" in r["text"]]
    print(f"Multi-line messages:   {len(multiline)}  {[r['message_id'] for r in multiline]}")
    if multiline:
        lines = multiline[0]["text"].split("\n")
        print(f"  └─ {multiline[0]['message_id']} has {len(lines)} lines: {lines}")

    # Colon-in-text messages
    colon_in_text = [r for r in rows if r["sender"] != "__system__" and ":" in r["text"]]
    print(f"Messages with colon in text: {len(colon_in_text)}")
    for r in colon_in_text:
        print(f"  └─ [{r['message_id']}] sender={r['sender']!r}  text={r['text']!r}")

    sys.exit(0)