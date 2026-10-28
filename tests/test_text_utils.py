from src.text_utils import normalize_text, split_into_chunks


def test_snake_case_in_backticks_is_spoken_as_words():
    assert normalize_text("Call `user_name` now.") == "Call user name now."


def test_camel_case_in_backticks_is_split():
    assert normalize_text("Use `getUserName` here.") == "Use get User Name here."


def test_plain_camel_case_words_are_untouched():
    assert normalize_text("We use PostgreSQL and JavaScript.") == "We use PostgreSQL and JavaScript."


def test_pronunciation_replaces_whole_words_only():
    prons = {"nginx": "engine X"}
    assert normalize_text("Run nginx now.", prons) == "Run engine X now."
    assert normalize_text("The nginxy thing.", prons) == "The nginxy thing."


def test_code_fences_are_removed():
    out = normalize_text("Intro.\n```python\nprint('x')\n```\nOutro.")
    assert "print" not in out and "Intro." in out and "Outro." in out


def test_smart_quotes_and_dashes():
    assert normalize_text("It\u2019s fast \u2014 really.") == "It's fast , really."


def test_paragraphs_and_sentences():
    chunks = split_into_chunks("One. Two.\n\nThree.")
    assert [c.text for c in chunks] == ["One.", "Two.", "Three."]
    assert [c.paragraph_end for c in chunks] == [False, True, True]


def test_abbreviations_do_not_split_sentences():
    chunks = split_into_chunks("Use a list, e.g. a tuple. Done.")
    assert [c.text for c in chunks] == ["Use a list, e.g. a tuple.", "Done."]


def test_long_sentence_is_split_under_limit():
    sentence = ", ".join(["word"] * 80) + "."
    chunks = split_into_chunks(sentence, max_chars=100)
    assert len(chunks) > 1 and all(len(c.text) <= 100 for c in chunks)


def test_empty_text_gives_no_chunks():
    assert split_into_chunks("   \n\n  ") == []
