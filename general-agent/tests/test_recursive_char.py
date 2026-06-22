from ingestion.splitters.recursive_char import clean_text, split_text

SEPS = ["\n\n", "。", ". ", " ", ""]


def test_short_text_single_chunk():
    assert split_text("hello world", 100, "\n\n", SEPS) == ["hello world"]


def test_fixed_separator_splits_paragraphs():
    assert split_text("a\n\nb\n\nc", 100, "\n\n", SEPS) == ["a", "b", "c"]


def test_oversized_piece_split_by_space_into_equal_chunks():
    text = " ".join(["aaaa"] * 10)  # 49 chars, no newline
    out = split_text(text, 20, "", SEPS)
    assert out == ["a" * 20, "a" * 20]


def test_char_level_fallback_when_no_separator():
    out = split_text("a" * 50, 20, "", [""])
    assert out == ["a" * 20, "a" * 20, "a" * 10]


def test_clean_collapses_and_strips_markers():
    assert clean_text("<|x|>") == "<x>"
    assert clean_text("a\n\n\n\nb") == "a\n\nb"
    assert clean_text("a    b") == "a b"
    assert clean_text("a\x00b") == "ab"
