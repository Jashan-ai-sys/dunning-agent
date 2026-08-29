"""The opening line, rendered instead of generated.

The greeting is the one turn with no customer input to reason about, so asking
the LLM for it costs a full round trip -- measured around 0.5s -- at the moment
the customer has just said hello and is listening hardest.
"""

import pytest

from app.voice.flow import GREETING, greeting_for

CONTEXT = {"company_name": "Acme", "customer_name": "Asha Rao"}


def test_the_hindi_greeting_is_devanagari():
    """The pipeline is pinned to Devanagari end to end; a romanised greeting
    would be the exact output the STT and TTS config exists to prevent."""
    line = greeting_for("hi", CONTEXT)
    assert any("ऀ" <= ch <= "ॿ" for ch in line), line
    assert "namaste" not in line.lower()


@pytest.mark.parametrize("language", ["hi", "hinglish", "en"])
def test_the_names_are_filled_in(language):
    line = greeting_for(language, CONTEXT)
    assert "Acme" in line
    assert "Asha Rao" in line
    assert "{" not in line, "an unrendered placeholder would be spoken aloud"


def test_an_unknown_language_falls_back_to_hindi():
    assert greeting_for("ta", CONTEXT) == greeting_for("hi", CONTEXT)


def test_missing_context_does_not_produce_a_placeholder():
    """A call body missing a name must not make the agent say '{customer_name}'."""
    line = greeting_for("hi", {})
    assert "{" not in line


def test_every_language_asks_who_it_is_speaking_to():
    """Identity confirmation before any billing detail is the compliance rule
    the greet node exists for; the rendered line must not skip it."""
    for language in GREETING:
        line = greeting_for(language, CONTEXT)
        assert "Asha Rao" in line
        assert line.strip().endswith("?"), f"{language}: {line}"


# --- the re-greeting regression --------------------------------------------


def test_speaking_the_greeting_strikes_the_instruction_to_greet():
    """The bug this fixes, caught on a live call.

    Recording the greeting as an assistant turn is necessary and not
    sufficient: the greet node still says "Open the call. Greet them", so the
    model does exactly that a second time and the customer hears the
    introduction twice.
    """
    from app.voice.flow import DUNNING_FLOW
    from app.voice.pipecat_agent import DunningSession

    session = DunningSession(
        {
            "customer_name": "Asha Rao",
            "company_name": "Acme",
            "amount_spoken": "499 रुपये",
            "failure_reason": "test",
            "suggested_route": "test route",
            "halt_note": "",
            "language_hint": "",
            "_language": "hi",
        }
    )
    opening = session.opening_line()
    assert opening is not None

    session.note_greeting_spoken(opening)
    messages = session.llm_context.get_messages()
    stage = next(m for m in messages if str(m.get("content", "")).startswith("[STAGE: "))

    assert "ALREADY greeted" in stage["content"]
    # The rest of the node still applies -- identity confirmation and its edges.
    assert "Available labels:" in stage["content"]
    assert "identity_confirmed" in stage["content"]
    # And the line we spoke is on the record as ours.
    assert any(
        m.get("role") == "assistant" and m.get("content") == opening for m in messages
    )
    assert DUNNING_FLOW is not None
