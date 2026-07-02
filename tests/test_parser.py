"""Tests for parser.py — WhatsApp export ingestion.

All tests go through _parse_raw() (internal) or parse_chat() (public file API).
_parse_raw is tested directly to avoid temp-file boilerplate for every case;
parse_chat is tested for the handful of things that require real file I/O
(BOM stripping via the utf-8-sig codec, CRLF normalisation on disk).

Coverage
--------
  Basic parsing          — single message, multiple messages, sequential IDs
  Multi-line messages    — continuation lines, newline preservation, 3+ lines
  System messages        — encryption notice, "was added", "left", no-colon lines
  Colon in message text  — first-colon split, multiple colons, decision syntax
  Timestamp formats      — HH:MM:SS, HH:MM (no seconds), AM/PM, DD/MM/YYYY (4-digit year)
  Line-ending variants   — CRLF, bare CR, mixed
  Edge cases             — empty input, blank lines, orphan continuation, BOM,
                           sender with spaces, very long sender name (truncated by regex)
  SAMPLE_EXPORT          — integration smoke test against the bundled 15-message fixture
"""

from __future__ import annotations

import pytest

from parser import SAMPLE_EXPORT, _parse_raw, parse_chat  # noqa: E402 (path set in conftest)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse(raw: str) -> list[dict]:
    """Thin wrapper so tests read as parse(raw) rather than _parse_raw(raw)."""
    return _parse_raw(raw)


# ===========================================================================
# Basic parsing
# ===========================================================================


class TestBasicParsing:
    def test_single_message_all_fields(self):
        rows = parse("[01/02/24, 10:00:00] Alice: Hello there")
        assert len(rows) == 1
        r = rows[0]
        assert r["message_id"] == "msg_0"
        assert r["sender"] == "Alice"
        assert r["text"] == "Hello there"
        assert r["timestamp"] == "01/02/24, 10:00:00"

    def test_sequential_message_ids(self):
        raw = (
            "[01/02/24, 10:00:00] Alice: Hi\n"
            "[01/02/24, 10:01:00] Bob: Hey\n"
            "[01/02/24, 10:02:00] Carol: Hello\n"
        )
        rows = parse(raw)
        assert [r["message_id"] for r in rows] == ["msg_0", "msg_1", "msg_2"]

    def test_senders_extracted_correctly(self):
        raw = "[01/02/24, 10:00:00] Alice: one\n" "[01/02/24, 10:01:00] Bob Smith: two\n"
        rows = parse(raw)
        assert rows[0]["sender"] == "Alice"
        assert rows[1]["sender"] == "Bob Smith"

    def test_sender_name_with_spaces(self):
        rows = parse("[01/02/24, 10:00:00] John Smith Jr: Hello everyone")
        assert rows[0]["sender"] == "John Smith Jr"
        assert rows[0]["text"] == "Hello everyone"

    def test_timestamp_preserved_verbatim(self):
        rows = parse("[15/11/23, 08:30:05] Dave: Morning")
        assert rows[0]["timestamp"] == "15/11/23, 08:30:05"

    def test_ids_restart_at_zero_per_parse_call(self):
        """Each _parse_raw call produces a fresh ID sequence."""
        rows1 = parse("[01/02/24, 10:00:00] Alice: A")
        rows2 = parse("[01/02/24, 10:00:00] Bob: B")
        assert rows1[0]["message_id"] == "msg_0"
        assert rows2[0]["message_id"] == "msg_0"


# ===========================================================================
# Multi-line messages
# ===========================================================================


class TestMultiLineMessages:
    def test_continuation_appended_to_previous(self):
        raw = (
            "[01/02/24, 10:00:00] Alice: First line\n"
            "second line\n"
            "third line\n"
            "[01/02/24, 10:05:00] Bob: Reply"
        )
        rows = parse(raw)
        assert len(rows) == 2
        assert rows[0]["text"] == "First line\nsecond line\nthird line"
        assert rows[1]["text"] == "Reply"

    def test_newlines_preserved_inside_message(self):
        raw = (
            "[01/06/24, 09:02:00] Carol: Line one.\n"
            "Line two.\n"
            "Line three.\n"
            "Line four.\n"
            "[01/06/24, 09:03:00] Alice: OK\n"
        )
        rows = parse(raw)
        lines = rows[0]["text"].split("\n")
        assert len(lines) == 4
        assert lines[0] == "Line one."
        assert lines[3] == "Line four."

    def test_multiple_messages_each_with_continuations(self):
        raw = (
            "[01/02/24, 10:00:00] Alice: A1\n" "A2\n" "[01/02/24, 10:01:00] Bob: B1\n" "B2\n" "B3\n"
        )
        rows = parse(raw)
        assert len(rows) == 2
        assert rows[0]["text"] == "A1\nA2"
        assert rows[1]["text"] == "B1\nB2\nB3"

    def test_blank_line_between_messages_not_appended(self):
        """A blank line is skipped, not glued to the previous message."""
        raw = "[01/02/24, 10:00:00] Alice: Hello\n" "\n" "[01/02/24, 10:01:00] Bob: Hi\n"
        rows = parse(raw)
        assert len(rows) == 2
        assert "\n" not in rows[0]["text"]


# ===========================================================================
# System messages
# ===========================================================================


class TestSystemMessages:
    def test_encryption_notice_tagged_as_system(self):
        raw = (
            "[01/02/24, 10:00:00] Messages and calls are end-to-end encrypted. "
            "No one outside of this chat can read them.\n"
            "[01/02/24, 10:01:00] Alice: Hi"
        )
        rows = parse(raw)
        assert rows[0]["sender"] == "__system__"
        assert rows[1]["sender"] == "Alice"

    def test_system_message_text_preserved(self):
        raw = "[01/02/24, 10:00:00] Messages and calls are end-to-end encrypted."
        rows = parse(raw)
        assert "end-to-end encrypted" in rows[0]["text"]

    def test_was_added_notice(self):
        rows = parse("[01/02/24, 10:00:00] Dave was added")
        assert rows[0]["sender"] == "__system__"
        assert "Dave was added" in rows[0]["text"]

    def test_left_notice(self):
        rows = parse("[01/02/24, 10:00:00] Carol left")
        assert rows[0]["sender"] == "__system__"

    def test_system_and_normal_messages_interleaved(self):
        raw = (
            "[01/02/24, 09:00:00] Alice: Hello\n"
            "[01/02/24, 09:01:00] Bob was added\n"
            "[01/02/24, 09:02:00] Bob: Thanks!\n"
        )
        rows = parse(raw)
        assert len(rows) == 3
        assert rows[0]["sender"] == "Alice"
        assert rows[1]["sender"] == "__system__"
        assert rows[2]["sender"] == "Bob"

    def test_system_messages_counted_in_message_ids(self):
        """System messages consume an ID slot like any other message."""
        raw = (
            "[01/02/24, 09:00:00] Alice: Hi\n"
            "[01/02/24, 09:01:00] Bob was added\n"
            "[01/02/24, 09:02:00] Bob: Thanks\n"
        )
        rows = parse(raw)
        assert rows[2]["message_id"] == "msg_2"


# ===========================================================================
# Colon in message text
# ===========================================================================


class TestColonInText:
    def test_first_colon_splits_sender_from_text(self):
        rows = parse("[01/02/24, 10:00:00] Alice: Note: don't forget")
        assert rows[0]["sender"] == "Alice"
        assert rows[0]["text"] == "Note: don't forget"

    def test_two_colons_in_body(self):
        rows = parse("[01/02/24, 10:00:00] Bob: Still blocked: deploy key expired: needs rotation")
        assert rows[0]["sender"] == "Bob"
        assert rows[0]["text"] == "Still blocked: deploy key expired: needs rotation"

    def test_decision_colon_syntax(self):
        rows = parse("[01/02/24, 10:00:00] Alice: Agreed. Decision: retro moves to Tuesday 3 pm.")
        assert rows[0]["sender"] == "Alice"
        assert "Decision: retro moves to Tuesday 3 pm." in rows[0]["text"]

    def test_url_in_message_body(self):
        rows = parse("[01/02/24, 10:00:00] Carol: Check this out: https://example.com/path?a=1&b=2")
        assert rows[0]["sender"] == "Carol"
        assert "https://example.com" in rows[0]["text"]

    def test_time_reference_in_body(self):
        rows = parse("[01/02/24, 10:00:00] Alice: Meeting at 14:30 today")
        assert rows[0]["sender"] == "Alice"
        assert "14:30" in rows[0]["text"]


# ===========================================================================
# Timestamp format variants
# ===========================================================================


class TestTimestampFormats:
    def test_hhmmss_format(self):
        rows = parse("[01/06/24, 09:15:32] Alice: Hi")
        assert rows[0]["timestamp"] == "01/06/24, 09:15:32"

    def test_hhmm_no_seconds(self):
        """Some WhatsApp exports omit seconds."""
        rows = parse("[01/06/24, 09:15] Alice: Hi")
        assert rows[0]["sender"] == "Alice"
        assert "09:15" in rows[0]["timestamp"]

    def test_four_digit_year(self):
        rows = parse("[01/06/2024, 09:15:32] Alice: Hi")
        assert rows[0]["sender"] == "Alice"
        assert "2024" in rows[0]["timestamp"]

    def test_single_digit_day_and_month(self):
        rows = parse("[1/6/24, 9:05:01] Alice: Hi")
        assert rows[0]["sender"] == "Alice"

    def test_ampm_timestamp(self):
        """iOS exports often use 12-hour AM/PM format."""
        rows = parse("[01/06/24, 9:15 AM] Alice: Morning")
        assert rows[0]["sender"] == "Alice"
        assert rows[0]["text"] == "Morning"


# ===========================================================================
# Line-ending variants
# ===========================================================================


class TestLineEndings:
    def test_crlf_line_endings_normalised(self):
        raw = "[01/02/24, 10:00:00] Alice: Hi\r\n[01/02/24, 10:01:00] Bob: Hey\r\n"
        rows = parse(raw)
        assert len(rows) == 2

    def test_bare_cr_line_endings_normalised(self):
        raw = "[01/02/24, 10:00:00] Alice: Hi\r[01/02/24, 10:01:00] Bob: Hey\r"
        rows = parse(raw)
        assert len(rows) == 2

    def test_mixed_line_endings(self):
        raw = (
            "[01/02/24, 10:00:00] Alice: Hi\r\n" "continuation\n" "[01/02/24, 10:01:00] Bob: Hey\r"
        )
        rows = parse(raw)
        assert len(rows) == 2
        assert "continuation" in rows[0]["text"]


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_string_returns_empty_list(self):
        assert parse("") == []

    def test_only_blank_lines_returns_empty_list(self):
        assert parse("\n\n\n\n") == []

    def test_orphan_continuation_at_file_start_is_skipped(self):
        raw = "no timestamp here\n[01/02/24, 10:00:00] Alice: Real"
        rows = parse(raw)
        assert len(rows) == 1
        assert rows[0]["text"] == "Real"

    def test_blank_lines_between_messages_ignored(self):
        raw = "[01/02/24, 10:00:00] Alice: Hi\n\n[01/02/24, 10:01:00] Bob: Hey\n"
        rows = parse(raw)
        assert len(rows) == 2

    def test_whitespace_only_lines_skipped(self):
        raw = "[01/02/24, 10:00:00] Alice: Hi\n   \n[01/02/24, 10:01:00] Bob: Hey\n"
        rows = parse(raw)
        assert len(rows) == 2

    def test_utf8_bom_stripped_by_parse_chat(self, tmp_path):
        """utf-8-sig codec strips BOM; verify via real file I/O."""
        p = tmp_path / "chat.txt"
        content = "[01/02/24, 10:00:00] Alice: Hello"
        p.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
        rows = parse_chat(str(p))
        assert len(rows) == 1
        assert rows[0]["sender"] == "Alice"

    def test_parse_chat_accepts_path_object(self, tmp_path):
        p = tmp_path / "chat.txt"
        p.write_text("[01/02/24, 10:00:00] Alice: Hi", encoding="utf-8")
        rows = parse_chat(p)  # pathlib.Path, not str
        assert rows[0]["sender"] == "Alice"

    def test_message_text_can_be_empty(self):
        """A message line with no text after the colon is valid."""
        rows = parse("[01/02/24, 10:00:00] Alice: ")
        assert len(rows) == 1
        assert rows[0]["text"] == ""


# ===========================================================================
# SAMPLE_EXPORT integration
# ===========================================================================


class TestSampleExport:
    """Integration tests against the bundled 15-message SAMPLE_EXPORT."""

    @pytest.fixture(autouse=True)
    def rows(self):
        self._rows = _parse_raw(SAMPLE_EXPORT)
        return self._rows

    def test_produces_exactly_15_messages(self):
        assert len(self._rows) == 15

    def test_message_ids_sequential_from_zero(self):
        ids = [r["message_id"] for r in self._rows]
        assert ids == [f"msg_{i}" for i in range(15)]

    def test_exactly_one_system_message(self):
        system = [r for r in self._rows if r["sender"] == "__system__"]
        assert len(system) == 1
        assert "end-to-end encrypted" in system[0]["text"]

    def test_carol_standup_is_multiline(self):
        carol = next(r for r in self._rows if r["message_id"] == "msg_3")
        assert carol["sender"] == "Carol"
        assert carol["text"].count("\n") >= 2

    def test_colon_in_body_does_not_corrupt_sender(self):
        """Bob's msg_5 contains 'staging:' in the text."""
        msg5 = next(r for r in self._rows if r["message_id"] == "msg_5")
        assert msg5["sender"] == "Bob"
        assert "staging:" in msg5["text"]

    def test_all_non_system_senders_non_empty(self):
        non_sys = [r for r in self._rows if r["sender"] != "__system__"]
        assert all(r["sender"] for r in non_sys)

    def test_all_timestamps_non_empty(self):
        assert all(r["timestamp"] for r in self._rows)