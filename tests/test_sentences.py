from backtalk.brains._sentences import SentenceBuffer, ThinkFilter


def test_sentences_complete_only_at_punctuation_plus_space():
    b = SentenceBuffer()
    assert b.feed("Hello there. How") == ["Hello there."]
    assert b.feed(" are you? I'm fine") == ["How are you?"]
    assert b.flush() == "I'm fine"
    assert b.flush() is None


def test_several_sentences_in_one_delta():
    b = SentenceBuffer()
    assert b.feed("One! Two. Three? ") == ["One!", "Two.", "Three?"]
    assert b.flush() is None


def test_punctuation_without_following_space_waits():
    # "3.5" or a sentence still streaming must not split early
    b = SentenceBuffer()
    assert b.feed("It costs 3.5") == []
    assert b.feed(" dollars. Ok") == ["It costs 3.5 dollars."]


def test_behavior_kept_from_warmbrain_abbreviations_split():
    # documented, unchanged behavior: a smarter rule is a separate change
    assert SentenceBuffer().feed("Mr. Smith left. ") == ["Mr.", "Smith left."]


def test_flush_mid_reply_then_continue():
    # the content_block_stop case: pre-tool filler flushes, then more text
    b = SentenceBuffer()
    b.feed("On it, let me grab that")
    assert b.flush() == "On it, let me grab that"
    assert b.feed("Found it. ") == ["Found it."]


def test_think_filter_drops_inline_reasoning():
    f = ThinkFilter()
    assert f.feed("<think>hmm, the user wants</think>Hello.") == "Hello."
    assert f.flush() == ""


def test_think_filter_tags_split_across_chunks():
    f = ThinkFilter()
    out = "".join(f.feed(c) for c in
                  ["Hi <thi", "nk>secret plan</th", "ink> there", "."])
    assert out + f.flush() == "Hi  there."


def test_think_filter_holds_only_possible_tag_prefixes():
    f = ThinkFilter()
    assert f.feed("a < b") == "a < b"
    assert f.feed(" and x <") == " and x "
    assert f.feed("3") == "<3"


def test_think_filter_unclosed_block_is_never_spoken():
    f = ThinkFilter()
    assert f.feed("<think>still going") == ""
    assert f.flush() == ""
