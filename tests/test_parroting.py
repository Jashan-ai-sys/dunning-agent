"""The agent must not hand the customer their own words back.

Regression tests for a live call on 2026-09-04 (voice_call 23, case 11) where
the agent echoed three times out of three:

    user:      हाँ जी।
    assistant: हाँ जी।
    user:      तो मैं क्या करूँ?
    assistant: तो मैं क्या करूँ?

Each echo satisfied `_model_spoke_this_turn`, which suppressed the follow-up
LLM run, so the echo *was* the turn and the real answer arrived one turn late.
The call ended `intent=unclear` with nothing recorded.
"""

import pytest

from app.voice.pipecat_agent import ALREADY_GREETED, _is_echo


@pytest.mark.parametrize(
    ("spoken", "heard"),
    [
        ("हाँ जी।", "हाँ जी।"),
        ("हाँ जी", "हाँ जी।"),  # punctuation must not hide an echo
        ("तो मैं क्या करूँ?", "तो मैं क्या करूँ?"),
        # The real failure was not verbatim: the model joined the previous
        # utterance to the current one.
        ("या। सर पेमेंट में मैं बैंक सर में डाउन था उस टाइम।",
         "सर पेमेंट में मैं बैंक सर में डाउन था उस टाइम।"),
    ],
)
def test_the_calls_own_echoes_are_caught(spoken, heard):
    assert _is_echo(spoken, heard) is True


@pytest.mark.parametrize(
    ("spoken", "heard"),
    [
        # The reply the greet stage is now explicitly asked to give.
        ("जी शुक्रिया।", "हाँ जी।"),
        # A real answer that happens to share a word with the question.
        ("आपका ₹499 का पेमेंट फेल हो गया है।", "तो मैं क्या करूँ?"),
        ("क्या आपको पता है कि पेमेंट क्यों फेल हो गया?", "हाँ जी।"),
        ("मैं आपको लिंक भेज देता हूँ।", "ठीक है, भेज दीजिए।"),
    ],
)
def test_real_replies_are_not_mistaken_for_echoes(spoken, heard):
    assert _is_echo(spoken, heard) is False


def test_short_customer_utterances_do_not_gag_the_agent():
    """"जी।" is both a legitimate acknowledgement and something the customer
    says. Refusing it would suppress the very reply we now ask for."""
    assert _is_echo("जी।", "जी।") is False


def test_empty_input_is_not_an_echo():
    assert _is_echo("", "हाँ जी।") is False
    assert _is_echo("हाँ जी।", "") is False


def test_greet_stage_names_a_permitted_reply():
    """The old instruction forbade every kind of content and then demanded
    speech, which is what left echoing as the only available move. Whatever
    this text becomes, it has to tell the model something it MAY say."""
    assert "never repeat the customer's own words" in ALREADY_GREETED
    assert "short acknowledgement in your own words" in ALREADY_GREETED
    # No literal phrase dictated -- the words are the model's to choose.
    assert "जी शुक्रिया" not in ALREADY_GREETED
    # And it must still hold the compliance line it was written for.
    assert "until identity is confirmed" in ALREADY_GREETED


# -- the stall the parroting fix exposed -------------------------------------

def test_an_acknowledgement_is_shorter_than_any_real_stage_line():
    """`_model_spoke_this_turn` suppresses the follow-up run when the model has
    already spoken. A 3-word ack is not the stage's content, and counting it as
    speech cost 5.7s of dead air on a live call -- the customer said "हेलो" to
    check the line was still open.

    This pins the gap the threshold sits in: acknowledgements below it, real
    stage lines above it.
    """
    from app.voice.pipecat_agent import _ECHO_STRIP, ACK_MAX_CHARS

    acks = ["जी शुक्रिया।", "जी।", "बिल्कुल।", "समझा।"]
    stage_lines = [
        "क्या आपको पता है कि पेमेंट क्यों नहीं हो पाया?",
        "आपके सब्सक्रिप्शन का ४९९ रुपये का पेमेंट नहीं हो पाया है क्योंकि आपका कार्ड एक्सपायर हो गया है।",
        "मैं आपको पेमेंट लिंक एसएमएस पर भेज देता हूँ।",
    ]

    for ack in acks:
        assert len(_ECHO_STRIP.sub("", ack)) <= ACK_MAX_CHARS, ack
    for line in stage_lines:
        assert len(_ECHO_STRIP.sub("", line)) > ACK_MAX_CHARS, line
