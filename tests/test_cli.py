import pytest

from whisper_memo.cli import BARS, format_bullets, level_bar, parse_args


def test_plain_text_is_unchanged():
    assert format_bullets("Just one thought.") == "Just one thought."


@pytest.mark.parametrize(
    "text",
    [
        "buy milk new line call mom",
        "Buy milk. New line. Call mom.",
        "buy milk, new-line, call mom",
        "buy milk newline call mom",
    ],
)
def test_new_line_makes_bullets(text):
    assert format_bullets(text).replace(".", "") == "- Buy milk\n- Call mom"


def test_trailing_new_line_adds_no_empty_bullet():
    assert format_bullets("one thing new line") == "- One thing"


def test_new_line_inside_a_word_is_ignored():
    assert format_bullets("renewlines are fine") == "renewlines are fine"


def test_level_bar_range():
    assert level_bar(0.0) == BARS[0]
    assert level_bar(1.0) == BARS[-1]
    levels = [BARS.index(level_bar(r)) for r in (1e-4, 1e-3, 1e-2, 1e-1)]
    assert levels == sorted(levels)


def test_defaults():
    args = parse_args([])
    assert (args.device, args.language, args.no_clipboard) == ("auto", "en", False)


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("whisper-memo ")
