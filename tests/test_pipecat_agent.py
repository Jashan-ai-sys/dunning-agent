"""The Pipecat wiring, tested at the seams that do not need audio.

The transport, the websocket and the LLM all need something we cannot have in a
test run. What we can test is everything that decides what those carry: the
context built from the dispatcher's body, the text that reaches the voice, and
the guarantee that one call writes one outcome.

The VAD assertion looks paranoid until you know why it is there: Pipecat 1.0
moved `vad_analyzer` off the transport, and both param objects silently drop
unknown fields. Putting it back in the wrong place would not raise -- the agent
would start, sound configured, and never detect a turn.
"""


from app.voice.intents import CallIntent
from app.voice.pipecat_agent import (
    HINDI_CLIP_GROUPS,
    SILENT_NODES,
    TRANSPORT_PARAMS,
    DunningSession,
    SpokenFormFilter,
    build_services,
    build_turn_strategies,
    build_vad,
    context_from_body,
    normalize_for_speech,
)

BODY = {
    "customer_name": "Asha Rao",
    "amount_paise": 49_900,
    "failure_reason": "कार्ड एक्सपायर हो गया था",
    "preferred_language": "hi",
    "recovery_case_id": 7,
    "phone": "+919000000000",
}


# --- the context the dispatcher hands over ------------------------------


def test_the_body_becomes_a_renderable_context():
    context = context_from_body(BODY)
    session = DunningSession(context)

    assert session.recovery_case_id == 7
    assert session.language == "hi"
    # Rendering is what would fail on a real call if a key were missing.
    assert "Asha Rao" in session.walker.instructions()
    # ...but not the amount: the greeting withholds it until identity is
    # confirmed, so a wrong number never hears someone's billing details.
    assert "499 रुपये" not in session.walker.instructions()

    session.walker.transition("identity_confirmed")
    assert "499 रुपये" in session.walker.instructions()


def test_a_pre_rendered_amount_wins():
    """The dispatcher may know a spoken form we cannot rebuild from paise."""
    context = context_from_body({**BODY, "amount_spoken": "साढ़े चार सौ रुपये"})
    assert context["amount_spoken"] == "साढ़े चार सौ रुपये"


def test_an_empty_body_still_renders():
    """A console run must not read a literal '{customer_name}' down the line."""
    session = DunningSession(context_from_body({}))
    assert "{" not in session.walker.instructions()
    assert session.recovery_case_id is None


# --- what actually reaches the voice ------------------------------------


async def test_the_filter_rewrites_only_for_speech():
    """Amounts, times and markdown are spoken forms, not context edits."""
    filter_ = SpokenFormFilter("hi")
    assert await filter_.filter("**500000 rupees**") == "5 लाख rupees"
    assert await filter_.filter("7:30 PM") == "शाम साढ़े सात बजे"


async def test_english_keeps_english_forms():
    filter_ = SpokenFormFilter("en")
    assert "लाख" not in await filter_.filter("500000 rupees")


def test_normalisation_is_idempotent():
    """It runs per TTS chunk; a second pass must not corrupt the first."""
    once = normalize_for_speech("₹500000 चाहिए", "hi")
    assert normalize_for_speech(once, "hi") == once


# --- the wiring that fails silently -------------------------------------


def test_vad_is_not_on_the_transport():
    """Pipecat 1.0 moved it; Pydantic would drop it here without a word."""
    for name, factory in TRANSPORT_PARAMS.items():
        assert not hasattr(factory(), "vad_analyzer"), name


def test_vad_carries_the_tuned_values():
    """Blostem's threshold carries over; its silence duration deliberately does
    not -- see `test_vad_stop_secs_matches_what_pipecat_calibrates_against`."""
    from app.config import get_settings

    settings = get_settings()
    params = build_vad().params
    assert params.confidence == settings.vad_activation_threshold
    assert params.stop_secs != settings.vad_min_silence_duration


# --- one call, one outcome ----------------------------------------------


async def test_the_tool_enum_is_rebuilt_per_node():
    """The model picks a label valid *here*, never a destination."""
    session = DunningSession(context_from_body(BODY))
    assert [
        p for p in session.tools().standard_tools[0].properties["label"]["enum"]
    ] == ["identity_confirmed", "not_the_customer"]

    session.walker.transition("identity_confirmed")
    labels = session.tools().standard_tools[0].properties["label"]["enum"]
    assert "identity_confirmed" not in labels
    assert "acknowledged" in labels


async def test_finalising_twice_writes_once(monkeypatch):
    """The graph finalises at its terminal node; `bot` finalises again after
    teardown so a hang-up is still recorded. The second must be free."""
    calls = []

    async def fake_finalise_call(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("app.voice.pipecat_agent.finalise_call", fake_finalise_call)

    session = DunningSession(context_from_body(BODY))
    session.walker.transition("identity_confirmed")
    session.walker.transition("acknowledged")
    session.walker.transition("reason_given")
    session.walker.transition("pay_now")

    await session.finalise()
    await session.finalise()

    assert len(calls) == 1
    assert calls[0]["recovery_case_id"] == 7
    assert calls[0]["result"].intent == CallIntent.RETRY_NOW
    assert calls[0]["result"].final_node_id == "pay_now"


async def test_an_abandoned_call_records_unclear(monkeypatch):
    """Hung up during the greeting: no intent, but the attempt is evidence."""
    calls = []

    async def fake_finalise_call(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("app.voice.pipecat_agent.finalise_call", fake_finalise_call)

    session = DunningSession(context_from_body(BODY))
    await session.finalise()

    assert calls[0]["result"].intent == CallIntent.UNCLEAR
    assert calls[0]["result"].final_node_id == "greet"


# --- turn-taking ---------------------------------------------------------


def test_the_customer_is_not_cut_off_by_a_clock():
    """A silence timer would file a mid-sentence pause as a finished turn."""
    strategies = build_turn_strategies()
    stop = strategies.stop[0]
    assert type(stop).__name__ == "TurnAnalyzerUserTurnStopStrategy"


def test_a_backchannel_cannot_interrupt_the_agent():
    """"हाँ" while the agent speaks must not steal the turn.

    MinWords has to *replace* the VAD start strategy, not join it: start
    strategies race, so a VAD start would fire on the first syllable and make
    the word threshold meaningless.
    """
    start = build_turn_strategies().start
    assert [type(s).__name__ for s in start] == ["MinWordsUserTurnStartStrategy"]


def test_vad_stop_secs_matches_what_pipecat_calibrates_against():
    """0.2 is a documented contract, not a taste: STT p99 values assume it."""
    assert build_vad().params.stop_secs == 0.2


def test_sarvam_does_not_second_guess_the_turn():
    """On 1.7.0 server VAD is never wired to the aggregator; it only mutes the
    flush that makes our own turn detection fast."""
    stt, _, _ = build_services("hi")
    assert stt._settings.vad_signals is False


# --- backchannel ---------------------------------------------------------


def test_the_clips_are_devanagari():
    """Romanised clips would flip script inside a 300ms sound."""
    for group, clips in HINDI_CLIP_GROUPS.items():
        assert len(clips) >= 2, f"{group} would repeat itself"
        for clip in clips:
            assert any("ऀ" <= ch <= "ॿ" for ch in clip), clip


class FakeGate:
    def __init__(self) -> None:
        self.enabled = True


def test_the_agent_stops_murmuring_when_a_charge_is_disputed():
    """Agreement noises while someone disputes a charge read as conceding it."""
    session = DunningSession(context_from_body(BODY))
    session.backchannel_gate = FakeGate()

    session.walker.transition("identity_confirmed")
    session.apply_backchannel_policy()
    assert session.backchannel_gate.enabled

    session.walker.transition("disputes_charge")
    session.apply_backchannel_policy()
    assert not session.backchannel_gate.enabled


def test_every_sensitive_node_is_reachable_and_silenced():
    """A node named in SILENT_NODES that no longer exists would silently stop
    protecting anything."""
    from app.voice.flow import DUNNING_FLOW

    node_ids = {node.id for node in DUNNING_FLOW.nodes}
    assert SILENT_NODES <= node_ids


def test_the_policy_is_a_no_op_without_a_pipeline():
    """demo_call and the tests build a session with no backchannel at all."""
    session = DunningSession(context_from_body(BODY))
    assert session.backchannel_gate is None
    session.apply_backchannel_policy()  # must not raise


def test_the_transcript_is_evidence_not_context():
    session = DunningSession(context_from_body(BODY))
    session.llm_context.set_messages(
        [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "हाँ जी, आशा बोल रही हूँ"},
            {"role": "assistant", "content": "धन्यवाद"},
        ]
    )
    transcript = session._transcript()

    assert "instructions" not in transcript
    assert "user: हाँ जी, आशा बोल रही हूँ" in transcript
    assert "assistant: धन्यवाद" in transcript
