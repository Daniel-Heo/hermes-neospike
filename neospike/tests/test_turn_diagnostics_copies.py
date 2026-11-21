"""진단 블록: 모델이 답변 끝에 따라 쓴 사본 제거와 빈 줄 없는 줄 단위 형식."""
from __future__ import annotations

from neospike.debug import TurnTraceLog, strip_block_copies

HEADER = "**Neospike: diagnostics**"
OLD_FORMAT_COPY = (
    "\n\n---\n\n" + HEADER + "\n\nRetrieved context: none.\n\n"
    "No saves: Neospike received no successful write operation."
)


def new_format_copy() -> str:
    log = TurnTraceLog()
    log.start("copy")
    return log.append("copy", "", enabled=True)


def traced_append(response: str) -> str:
    log = TurnTraceLog()
    log.start("k")
    log.recall("k", "first remembered row\nsecond remembered row")
    log.change("k", "The project uses Python.", "project")
    return log.append("k", response, enabled=True)


def test_one_or_two_copied_blocks_leave_exactly_one_block():
    for copies in (1, 2):
        response = "Answer." + new_format_copy() * copies
        block = traced_append(response)
        assert block.count(HEADER) == 1
        assert block.startswith("Answer.\n\n---\n" + HEADER)
        assert block.count("---") == 1


def test_old_format_copy_is_removed():
    block = traced_append("Answer." + OLD_FORMAT_COPY + new_format_copy())
    assert block.count(HEADER) == 1
    assert "Neospike received no successful write" not in block
    assert block.startswith("Answer.\n\n---\n" + HEADER)


def test_title_quoted_in_the_body_is_kept():
    body = (f"The reply ends with a {HEADER} block.\n\n{HEADER}\n\n"
            "That line above is only an example; this sentence is ordinary prose.")
    block = traced_append(body + new_format_copy())
    assert block.startswith(body + "\n\n---\n")
    assert block.count(HEADER) == 3


def test_block_has_no_blank_lines_and_breaks_every_line():
    block = traced_append("Answer.")
    tail = block.split("\n---\n", 1)[1]
    rows = tail.split("\n")
    assert "" not in rows
    assert all(row.endswith("\\") for row in rows[:-1]) and not rows[-1].endswith("\\")
    assert rows[0] == HEADER + "\\"
    assert "\\> first remembered row\\" in rows and "\\> second remembered row\\" in rows
    assert not any(row.startswith("> ") for row in rows)  # no blockquote to swallow the next lines


def test_turn_without_a_trace_returns_copies_untouched():
    response = "Answer." + OLD_FORMAT_COPY + new_format_copy()
    log = TurnTraceLog()
    log.start("k")
    assert log.append("k", response, enabled=False) == response
    assert log.append("never-started", response, enabled=True) == response


MODEL_WRITTEN_COPY = (
    "\n\n---\n\n" + HEADER + "\n\nRetrieved context: none.\n\nMemory changes:\n"
    "- Saved (global): The user prefers short answers that put the conclusion first and the reasons after it.\n"
    "* Saved (project): Indexing runs on Python.\n"
    "\nObserved results:\nCorrected an earlier note about the release schedule.\n"
)


def test_copy_with_dash_bullets_and_unbulleted_section_text_is_removed():
    block = traced_append("Answer." + MODEL_WRITTEN_COPY)
    assert block.count(HEADER) == 1
    assert "prefers short answers" not in block and "release schedule" not in block
    assert block.startswith("Answer.\n\n---\n" + HEADER)


def test_body_list_above_the_copy_is_kept():
    body = "Next steps:\n- Run the tests\n* Review the diff\n• Ship it"
    block = traced_append(body + MODEL_WRITTEN_COPY)
    assert block.startswith(body + "\n\n---\n" + HEADER)
    assert block.count(HEADER) == 1


def test_strip_keeps_reply_without_copy_byte_for_byte():
    response = "Line one\r\n\r\n- item\r\n> quote\r\n---\r\n"
    assert strip_block_copies(response) == response
