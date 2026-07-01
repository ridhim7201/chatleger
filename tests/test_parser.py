import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from parser import parse_whatsapp_export  # noqa: E402


def test_basic_single_line_message():
    raw = "[01/02/24, 10:00:00] Alice: Hello there"
    msgs = parse_whatsapp_export(raw)
    assert len(msgs) == 1
    assert msgs[0].message_id == "msg_0"
    assert msgs[0].sender == "Alice"
    assert msgs[0].message_text == "Hello there"
    assert msgs[0].timestamp == "01/02/24 10:00:00"


def test_multiple_messages_sequential_ids():
    raw = (
        "[01/02/24, 10:00:00] Alice: Hi\n"
        "[01/02/24, 10:01:00] Bob: Hey Alice\n"
        "[01/02/24, 10:02:00] Alice: How's it going?"
    )
    msgs = parse_whatsapp_export(raw)
    assert [m.message_id for m in msgs] == ["msg_0", "msg_1", "msg_2"]
    assert [m.sender for m in msgs] == ["Alice", "Bob", "Alice"]


def test_multiline_message_continuation():
    raw = (
        "[01/02/24, 10:00:00] Alice: First line\n"
        "second line of same message\n"
        "third line too\n"
        "[01/02/24, 10:05:00] Bob: A reply"
    )
    msgs = parse_whatsapp_export(raw)
    assert len(msgs) == 2
    assert msgs[0].message_text == "First line\nsecond line of same message\nthird line too"
    assert msgs[1].message_text == "A reply"


def test_missing_timestamp_leading_line_is_skipped():
    raw = "this has no timestamp and no prior message\n" "[01/02/24, 10:00:00] Alice: Real message"
    msgs = parse_whatsapp_export(raw)
    assert len(msgs) == 1
    assert msgs[0].message_text == "Real message"


def test_malformed_lines_are_skipped_or_attached():
    raw = (
        "[01/02/24, 10:00:00] Alice: Valid message\n"
        "===random garbage line===\n"
        "[bad-timestamp-format] Charlie: Not parseable as header\n"
    )
    msgs = parse_whatsapp_export(raw)
    # Both garbage lines lack a valid timestamp match, so they are folded
    # into the preceding valid message as continuations.
    assert len(msgs) == 1
    assert "random garbage line" in msgs[0].message_text


def test_empty_input_returns_empty_list():
    assert parse_whatsapp_export("") == []
    assert parse_whatsapp_export("\n\n\n") == []


def test_sender_with_colon_in_message_body():
    raw = "[01/02/24, 10:00:00] Alice: Note: don't forget the meeting"
    msgs = parse_whatsapp_export(raw)
    assert len(msgs) == 1
    assert msgs[0].sender == "Alice"
    assert msgs[0].message_text == "Note: don't forget the meeting"


def test_blank_lines_between_messages_are_ignored():
    raw = "[01/02/24, 10:00:00] Alice: Hi\n" "\n" "[01/02/24, 10:01:00] Bob: Hey\n"
    msgs = parse_whatsapp_export(raw)
    assert len(msgs) == 2


def test_utf8_bom_is_stripped():
    raw = "\ufeff[01/02/24, 10:00:00] Alice: Hello"
    msgs = parse_whatsapp_export(raw)
    assert len(msgs) == 1
    assert msgs[0].sender == "Alice"
