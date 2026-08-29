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


import logging

import pytest
from pipecat.frames.frames import ErrorFrame

from app.constants import CallStatus
from app.voice.flow import halt_note
from app.voice.intents import CallIntent, outcome_for
from app.voice.pipecat_agent import (
    HINDI_CLIP_GROUPS,
    SILENT_NODES,
    TRANSPORT_PARAMS,
    DunningSession,
    SpokenFormFilter,
    build_backchannel,
    build_services,
    build_turn_strategies,
    build_vad,
    context_from_body,
    normalize_for_speech,
    watch_tts_failures,
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


@pytest.mark.parametrize("value", [False, "false", "False", "0", "", None])
def test_a_halt_flag_that_crossed_a_wire_as_a_string_stays_false(value):
    """Twilio delivers every parameter as a string, where bool("false") is True.

    Reading it wrong tells the agent the subscription has stopped when it has
    not, which changes what it says on a live money call.
    """
    context = context_from_body({**BODY, "subscription_halted": value})
    assert context["halt_note"] == halt_note(False)


@pytest.mark.parametrize("value", [True, "true", "1"])
def test_a_genuine_halt_still_reaches_the_agent(value):
    context = context_from_body({**BODY, "subscription_halted": value})
    assert context["halt_note"] == halt_note(True)


@pytest.mark.parametrize("value", [7, "7"])
def test_the_case_id_is_an_int_however_it_arrived(value):
    """Twilio sends "7"; persistence needs 7.

    Left as a string it reaches asyncpg as a parameter for an integer column and
    is rejected -- and persistence swallows its own failures on purpose, so the
    call would sound perfect and never send the payment link.
    """
    session = DunningSession(context_from_body({**BODY, "recovery_case_id": value}))
    assert session.recovery_case_id == 7


@pytest.mark.parametrize("value", [None, "", "not-a-case"])
def test_an_unusable_case_id_becomes_no_case_rather_than_a_bad_one(value):
    """A demo run with no case must write nothing, not write to case 0."""
    session = DunningSession(context_from_body({**BODY, "recovery_case_id": value}))
    assert session.recovery_case_id is None


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


def _enum(session) -> list[str]:
    return session.tools().standard_tools[0].properties["label"]["enum"]


def test_the_tool_schema_does_not_change_mid_call():
    """It is sent with every request, at the front of what a provider hashes
    for its prefix cache. A schema that changed per stage would cost a hit on
    the turn after every transition -- the turns with the most context to save.
    """
    session = DunningSession(context_from_body(BODY))
    before = _enum(session)

    for label in ["identity_confirmed", "acknowledged", "reason_given"]:
        session.walker.transition(label)
        assert _enum(session) == before


def test_the_enum_covers_every_label_in_the_flow():
    """The old per-node version froze at greet's labels, because set_tools runs
    once -- so the schema offered `identity_confirmed` while the node was asking
    for `acknowledged`. It only worked because Gemini treats an enum as advice.
    """
    session = DunningSession(context_from_body(BODY))
    from app.voice.flow import DUNNING_FLOW

    assert set(_enum(session)) == {
        edge.label for node in DUNNING_FLOW.nodes for edge in node.edges
    }


def test_a_label_valid_elsewhere_is_still_refused_here():
    """The safety the narrow enum used to provide now lives where it belongs:
    a broad schema costs a rejected tool call, never a wrong branch."""
    from app.voice.walker import InvalidTransition

    session = DunningSession(context_from_body(BODY))
    assert "pay_now" in _enum(session)  # advertised...

    with pytest.raises(InvalidTransition):
        session.walker.transition("pay_now")  # ...but not from `greet`


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


# --- a mute agent ---------------------------------------------------------


async def _finalise_with_tts_failures(monkeypatch, failures: list[str], label: str):
    calls = []

    async def fake_finalise_call(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("app.voice.pipecat_agent.finalise_call", fake_finalise_call)

    session = DunningSession(context_from_body(BODY))
    session.walker.transition("identity_confirmed")
    session.walker.transition("acknowledged")
    session.walker.transition("reason_given")
    session.walker.transition(label)
    for failure in failures:
        session.note_tts_failure(failure)

    await session.finalise()
    return calls[0]["result"]


async def test_a_call_the_customer_could_not_hear_records_no_intent(monkeypatch):
    """The customer answered questions that were never spoken.

    Cartesia ran out of credits mid-call. Nothing else broke: the model kept
    generating, the graph kept advancing on the customer's replies, and the
    walker arrived at `pay_now` from a conversation that only one side heard.
    """
    result = await _finalise_with_tts_failures(
        monkeypatch, ["quota_exceeded: 2 credits remaining"], "pay_now"
    )

    assert result.intent == CallIntent.UNCLEAR
    assert result.status == CallStatus.FAILED
    assert "quota_exceeded" in result.error


async def test_a_mute_call_never_suppresses_the_customer(monkeypatch):
    """The reason this is worth a test of its own.

    `declined` bans the person from every future call. Reaching it because
    they heard silence and gave up would end the relationship over our own
    outage, and nothing downstream would ever question it.
    """
    result = await _finalise_with_tts_failures(monkeypatch, ["quota_exceeded"], "declined")

    assert result.intent == CallIntent.UNCLEAR
    assert not outcome_for(result.intent).suppress_contact


async def test_the_node_reached_is_still_recorded(monkeypatch):
    """The intent is discarded; the evidence is not. Without the node and the
    transcript there is no way to tell this call from one nobody answered."""
    result = await _finalise_with_tts_failures(monkeypatch, ["quota_exceeded"], "pay_now")

    assert result.final_node_id == "pay_now"
    assert result.transitions


async def test_a_call_that_was_heard_keeps_its_intent(monkeypatch):
    """The guard must not fire on every call -- that would be an outage of
    its own, silently retrying customers who already agreed to pay."""
    result = await _finalise_with_tts_failures(monkeypatch, [], "pay_now")

    assert result.intent == CallIntent.RETRY_NOW
    assert result.status == CallStatus.COMPLETED
    assert result.error is None


async def test_the_real_tts_service_reports_its_failures():
    """The event name has to be the one Pipecat actually raises.

    Everything else here is exercised by calling `note_tts_failure` directly,
    which proves nothing about the wiring. This drives a real
    `CartesiaTTSService` through its own event machinery: get the name wrong
    and the guard is dead code that reports every mute call as a success.
    """
    _, _, tts = build_services("hi")
    session = DunningSession(context_from_body(BODY))
    watch_tts_failures(tts, session)

    await tts.push_error_frame(ErrorFrame(error="quota_exceeded"))

    assert session.tts_failures == ["quota_exceeded"]


def test_only_the_first_silent_line_is_logged(caplog):
    """A spent quota fails every utterance for the rest of the call. Twenty
    identical lines bury the one that explains the outage."""
    session = DunningSession(context_from_body(BODY))

    with caplog.at_level(logging.ERROR, logger="app.voice.pipecat_agent"):
        for _ in range(5):
            session.note_tts_failure("quota_exceeded")

    assert len(session.tts_failures) == 5
    assert sum("nothing the agent says" in r.message for r in caplog.records) == 1


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


def test_no_clip_is_something_a_customer_would_say():
    """The parrot bug, caught on a live call.

    The customer said "हाँ जी" and heard "हाँ जी" back half a second later. It
    was not the model repeating them -- it was the backchannel, whose clip
    library happened to contain the four most common acknowledgements an
    Indian customer uses. A listening noise has to be something a listener
    says and a payer does not.
    """
    customer_words = {"हाँ।", "जी।", "हाँ जी।", "जी हाँ।", "हाँ", "जी"}
    for group, clips in HINDI_CLIP_GROUPS.items():
        for clip in clips:
            assert clip not in customer_words, f"{group}: {clip} is what they say to us"


def test_the_backchannel_will_not_fire_on_a_short_utterance():
    """"हाँ जी" is half a second and complete. There is no sentence to murmur
    underneath, so anything said there is a reply, not a backchannel."""
    backchannel = build_backchannel("hi")
    assert backchannel._params.min_speech_before_eligible_s >= 1.0


def test_the_backchannel_lands_mid_sentence_not_after_the_pause():
    """It fires on its own VAD seeing a stop. A long stop puts the clip after
    the customer has finished, which is a reply; a short one puts it in the
    breath while they are still going."""
    backchannel = build_backchannel("hi")
    assert backchannel._params.vad_stop_secs <= 0.1


def test_every_tool_the_flow_asks_for_is_advertised():
    """The drift that cost a live call.

    A fourth tool was added and the flow was told to use it, but the MCP
    allow-list was not updated. Pipecat refused the call -- "not in the
    currently advertised tool set" -- after the model had already told the
    customer it was sending a link. The filter is a real safety mechanism and
    it worked; what was missing was anything to notice the two had diverged.
    """
    import re

    from app.voice.flow import DUNNING_FLOW, SYSTEM_STYLE
    from app.voice.pipecat_agent import MCP_TOOLS

    prompts = SYSTEM_STYLE + " ".join(
        node.prompt or "" for node in DUNNING_FLOW.nodes
    )
    asked_for = {
        name
        for name in re.findall(r"`(\w+)`", prompts)
        if name.startswith(("send_", "get_"))
    }
    assert asked_for, "no tools referenced -- the pattern stopped matching"

    missing = asked_for - set(MCP_TOOLS)
    assert not missing, f"the flow asks for {missing}, which the model cannot call"


def test_the_transition_tool_is_not_in_the_mcp_list():
    """`transition` stays in-process: it is graph traversal, not an external
    capability, and it must not be able to fail over a transport."""
    from app.voice.pipecat_agent import MCP_TOOLS

    assert "transition" not in MCP_TOOLS


# --- Sarvam server-side VAD tuning -----------------------------------------


def test_nothing_is_sent_to_sarvam_unless_we_have_an_opinion():
    from app.config import Settings
    """Unset is not the same as any value we could pick. Passing a guess for a
    parameter we have no view on replaces a tuned default with an untuned one."""
    from app.voice.pipecat_agent import sarvam_vad_overrides

    assert sarvam_vad_overrides(Settings()) == {}


def test_only_the_configured_parameters_are_sent():
    from app.config import Settings
    from app.voice.pipecat_agent import sarvam_vad_overrides

    overrides = sarvam_vad_overrides(
        Settings(
            sarvam_start_speech_volume_threshold=-35.0,
            sarvam_min_speech_frames=4,
        )
    )
    assert overrides == {
        "start_speech_volume_threshold": -35.0,
        "min_speech_frames": 4,
    }


def test_the_names_match_what_pipecat_accepts():
    """A typo here is silent: pydantic settings objects drop unknown fields, so
    the parameter would simply never reach Sarvam and the tuning would appear
    to do nothing."""
    import dataclasses

    from pipecat.services.sarvam.stt import SarvamSTTSettings

    from app.config import Settings
    from app.voice.pipecat_agent import sarvam_vad_overrides

    accepted = {f.name for f in dataclasses.fields(SarvamSTTSettings)}
    ours = sarvam_vad_overrides(
        Settings(
            sarvam_start_speech_volume_threshold=-35.0,
            sarvam_min_speech_frames=4,
            sarvam_interrupt_min_speech_frames=6,
            sarvam_positive_speech_threshold=0.6,
            sarvam_negative_speech_threshold=0.3,
            sarvam_pre_speech_pad_frames=2,
        )
    )
    unknown = set(ours) - accepted
    assert not unknown, f"pipecat would silently drop {unknown}"
