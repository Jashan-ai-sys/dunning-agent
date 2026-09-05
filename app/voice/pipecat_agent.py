"""The dunning conversation on Pipecat, as an alternative to the LiveKit agent.

    uv run --group pipecat python -m app.voice.pipecat_agent

Everything above the transport is shared with the LiveKit path: the same
``DUNNING_FLOW``, the same ``GraphWalker``, the same policy, the same
persistence. Only the plumbing differs. That was the point of keeping traversal
out of the transport layer -- swapping frameworks touches this file and nothing
in ``walker.py``, ``policy.py`` or ``outcomes.py``.

Why Pipecat is worth having alongside LiveKit: it ships serializers for Plivo,
Exotel and Twilio, so an Indian telephony leg is a transport swap rather than a
SIP trunk negotiation. LiveKit remains the better browser-demo path.

A note on the Blostem backend: it runs Pipecat 0.0.108, whose frame processors
are not API-compatible with 1.x. Its *pure* modules ported directly (see
``app/voice/normalizers``); its processors did not, and are re-implemented here
against the current API where they earn their place.
"""

import asyncio
import logging
import os
import re
import sys

from dotenv import load_dotenv
from mcp import StdioServerParameters
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    FunctionCallResultProperties,
    InterimTranscriptionFrame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.mcp_service import MCPClient
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat_backchannel import Backchannel, BackchannelParams
from pipecat_backchannel.cache import FileClipCache
from pipecat_backchannel.processor import BackchannelProcessor

from app.config import get_settings
from app.constants import CallStatus
from app.store import utcnow
from app.voice.announcements import is_carrier_announcement
from app.voice.call_body import load_call_body
from app.voice.flow import DUNNING_FLOW, greeting_for, halt_note, language_hint
from app.voice.graph import NodeKind
from app.voice.guardrails import build_guardrail
from app.voice.instrumentation import TimedSileroVAD, TimedSmartTurnAnalyzer
from app.voice.intents import CallIntent
from app.voice.local_services import LocalSTTService, LocalTTSService
from app.voice.normalizers import (
    normalize_amounts_for_tts,
    normalize_for_hindi_tts,
    normalize_times_for_tts,
    strip_markdown,
)
from app.voice.outcomes import CallResult
from app.voice.persistence import duration_since, finalise_call, open_call_record
from app.voice.reasons import spoken_reason
from app.voice.spoken import spoken_amount
from app.voice.walker import GraphWalker, InvalidTransition

logger = logging.getLogger(__name__)

load_dotenv()

#: Used when the runner starts a session with no body -- `pipecat run` from a
#: checkout does this. Never used on a dispatched recovery call, which always
#: carries a case. Mirrors SAMPLE_CONTEXT on the LiveKit path.
SAMPLE_BODY = {
    "customer_name": "Asha",
    "amount_paise": 49_900,
    "failure_reason": "आपके कार्ड में पर्याप्त बैलेंस नहीं था",
    "preferred_language": "hi",
}


def normalize_for_speech(text: str, language: str = "hi") -> str:
    """Apply the deterministic rewrites just before synthesis.

    Order matters: markdown first, then amounts and times (which insert
    Devanagari unit words), then the script pass over what remains. The LLM
    context keeps the original -- only these bytes are rewritten.
    """
    spoken = strip_markdown(text)
    spoken, _ = normalize_amounts_for_tts(spoken, lang=language)
    spoken, _ = normalize_times_for_tts(spoken, lang=language)
    if language in ("hi", "hinglish"):
        spoken, _ = normalize_for_hindi_tts(spoken)
    return spoken


#: Devanagari, because the clips are recorded through the same Cartesia voice
#: that is pinned to `hi` -- romanised "haan ji" is the script flip that makes
#: a Hindi voice stumble, and it would stumble here in the one place we cannot
#: afford it: a 300ms clip with no context to recover in.
#:
#: Two clips minimum per group, or the same sound repeats and reads as a loop.
#: Deliberately excludes "हाँ।", "जी।", "हाँ जी।" and "जी हाँ।".
#:
#: They are the four most common things an Indian customer says on a call, so
#: hearing one back half a second later does not read as listening -- it reads
#: as the agent parroting. That happened on a live call: the customer said
#: "हाँ जी" and the agent said "हाँ जी" straight back.
#:
#: What is left is what a listener says and a payer does not: a hum, an
#: "अच्छा", a "बिल्कुल".
#: Replaces the greet node's own prompt once we have spoken the greeting
#: ourselves. It has to *replace* rather than precede it: "do not greet
#: again" sitting above "Open the call. Greet them" is a contradiction, and
#: the model resolves it by greeting -- which it did, on a live call. The
#: node's edges are untouched, so identity confirmation and the
#: wrong-number path work exactly as before.
#: "React to what they say" was the whole of the old instruction here, and it
#: is what produced the second parroting bug on a live call. Read it with the
#: rest of the stage in front of you: do not greet, do not introduce yourself,
#: do not name the company, do not mention the payment -- and then, from
#: `stage_instructions`, say something in the same response as the tool call.
#: To a bare "हाँ जी।" there is no permitted content left, and the only move
#: the model has is to hand the words back. It did, three times out of three,
#: and each echo entered the context as an assistant turn and taught it the
#: pattern for the next one.
#:
#: The fix is to name the permitted move rather than forbid every other one. A
#: three-word acknowledgement is something to say, and telling the model the
#: next stage carries the content removes the pressure to fill this turn.
#: Punctuation and spacing only -- the two Devanagari danda forms and the ASCII
#: equivalents Sarvam sometimes returns. Comparing without them means "हाँ जी।"
#: and "हाँ जी" are recognised as the same words, which is the whole point.
_ECHO_STRIP = re.compile(r"[।॥.,!?\s]+")

#: Stripped length below which a model turn is an acknowledgement rather than
#: the stage's content. "जी शुक्रिया।" is 10 characters once punctuation goes;
#: the shortest real stage line measured on a call -- "क्या आपको पता है कि
#: पेमेंट क्यों नहीं हो पाया?" -- is 38. The gap is wide enough that the exact
#: threshold does not matter much, which is the property you want in a
#: heuristic that decides whether to spend a round trip.
ACK_MAX_CHARS = 24


def _is_echo(spoken: str, heard: str) -> bool:
    """Is the model's line just the customer's words handed back?

    Substring rather than equality, in both directions. The observed failure
    was not always verbatim: the model returned the previous utterance joined
    to the current one ("या। सर पेमेंट में..."), so an exact match would have
    missed the case that actually happened on the call.

    Very short lines are exempt. "जी।" is a legitimate acknowledgement and is
    also something the customer says; refusing it would suppress the one reply
    the greet stage is now explicitly asked to give.
    """
    a = _ECHO_STRIP.sub("", spoken)
    b = _ECHO_STRIP.sub("", heard)
    if not a or not b or len(b) < 4:
        return False
    return a in b or b in a


ALREADY_GREETED = (
    "You have already introduced yourself and asked whether you are "
    "speaking to the customer. They are answering that question now. Do "
    "NOT greet, introduce yourself, or say the company name again, and "
    "never repeat the customer's own words back to them. If they confirm "
    "it is them, give a short acknowledgement in your own words -- at most "
    "three words. It must be plain courtesy: thanks, or simple assent of "
    "the kind a call-centre agent uses. Do NOT praise them or congratulate "
    "them for confirming who they are; there is nothing to be pleased "
    "about yet. Then call the transition tool in the same response. The "
    "next stage carries "
    "what to say about the payment, so there is nothing else you need to "
    "fill this turn with. Do not mention the failed payment until identity "
    "is confirmed -- this is someone's billing information."
)


HINDI_CLIP_GROUPS = {
    "continue": ["हूँ।", "अच्छा।"],
    "affirm": ["बिल्कुल।", "समझा।"],
    # No trailing ellipsis on "अच्छा": Cartesia renders the pause literally and
    # returned a 1.5s clip, three times the others. A backchannel that long
    # stops being a murmur underneath the customer and starts interrupting.
    "thinking": ["हम्म।", "अच्छा"],
    "surprise": ["अच्छा?", "ओह।", "अरे।"],
}

#: Nodes where sounding agreeable is a compliance problem, not a nicety.
#: Murmuring "हाँ जी" while a customer disputes a charge reads as the agent
#: conceding the dispute; doing it after they have refused reads as pressure;
#: doing it to a wrong number is talking to someone who never opted in.
SILENT_NODES = frozenset({"dispute", "declined", "wrong_number"})


#: One per process per language, not one per call. The clip library and its
#: cache are process-wide state; building a fresh one per call would re-read
#: (or worse, re-record) 13 clips on every answered phone call.
_BACKCHANNELS: dict[str, Backchannel] = {}


def get_backchannel(language: str) -> Backchannel:
    """The process-wide backchannel for a language, built on first use."""
    if language not in _BACKCHANNELS:
        _BACKCHANNELS[language] = build_backchannel(language)
    return _BACKCHANNELS[language]


async def prewarm_backchannel(language: str) -> bool:
    """Record the clips before anyone is on the line.

    Recording takes a few seconds against Cartesia, and it happens inside
    pipeline startup -- so without this the *first* customer of a fresh deploy
    hears silence while their own backchannel is being synthesised. Cached on
    disk afterwards, so this is a no-op from the second call on.

    Never fatal: a bot that cannot murmur is still a bot that can collect money.
    """
    try:
        _, _, tts = build_services(language)
        return await get_backchannel(language).prewarm(tts=tts)
    except Exception:  # noqa: BLE001 - listening sounds are not worth a call
        logger.warning("could not prewarm backchannel clips", exc_info=True)
        return False


def build_backchannel(language: str) -> Backchannel:
    """The listening sounds, in the bot's own voice.

    Worth the cost on this call specifically: an Indian billing conversation is
    carried by continuous acknowledgement, and silence while someone explains a
    failed payment reads as a dead line -- they repeat themselves, or hang up.

    It takes no turn and never enters the LLM context, so it cannot corrupt
    graph traversal or the transcript we file as evidence. What it does cost is
    a second VAD and turn analyzer running while the customer speaks.

    The cache is keyed by language so switching does not replay Hindi clips at
    an English speaker; delete `.clip_cache` after changing the voice.
    """
    hindi = language in ("hi", "hinglish")
    return Backchannel(
        clip_groups=HINDI_CLIP_GROUPS if hindi else None,
        cache=FileClipCache(f".clip_cache/{language}"),
        params=BackchannelParams(
            # A dunning call is not a chat. Firing on every eligible pause
            # sounds eager, and eager is the wrong register when asking someone
            # for money they have already failed to pay.
            fire_probability=0.5,
            cooldown_s=3.5,
            # A backchannel belongs *underneath* someone who is still talking.
            # The library fires after its own VAD sees a pause, so a long
            # stop makes it land once they have finished -- which is a reply,
            # not a murmur, and is what made it sound like the agent was
            # answering rather than listening. A very short stop puts it in the
            # breath mid-sentence instead.
            vad_stop_secs=0.05,
            # And never on a short utterance. "हाँ जी" is half a second long
            # and complete; there is no sentence to murmur underneath, so
            # anything we say there is a response. Only speak while somebody is
            # actually explaining something.
            min_speech_before_eligible_s=1.5,
        ),
        # Below the default: a backchannel belongs underneath the person who
        # still has the floor, and on a phone leg it competes with their voice.
        volume=0.5,
    )


class SpokenFormFilter(BaseTextFilter):
    """Rewrites text on its way into Cartesia, and nowhere else.

    A text filter rather than a frame processor because this is exactly the
    seam Pipecat provides for it: the filter sits inside the TTS service, so
    the LLM context keeps the model's original words and only the synthesised
    bytes are normalised. A processor spliced between LLM and TTS would rewrite
    the assistant message the aggregator records too, and the transcript would
    stop being evidence of what the model actually said.
    """

    def __init__(self, language: str) -> None:
        self._language = language

    async def filter(self, text: str) -> str:
        return normalize_for_speech(text, self._language)


def resolve_sarvam_model(configured: str) -> str:
    """Pin to what the installed Pipecat actually accepts.

    `saaras:v4` is current at Sarvam and is what the LiveKit plugin runs, but
    Pipecat 1.7.0 validates against its own table and rejects it outright. Down-
    grading the shared setting would drag the working LiveKit path back a
    version for no reason, so the fallback lives here.

    The fallback is not "any supported model": it must still support `mode`,
    `language` and VAD params, or the transcribe mode and Hindi pin we depend on
    are silently dropped. Once Pipecat adds v4 the setting passes through and
    this does nothing.
    """
    from pipecat.services.sarvam.stt import MODEL_CONFIGS

    if configured in MODEL_CONFIGS:
        return configured

    usable = [
        name
        for name, config in MODEL_CONFIGS.items()
        if config.supports_mode and config.supports_language and config.supports_vad_params
    ]
    if not usable:
        raise RuntimeError(
            f"pipecat rejects '{configured}' and offers no model supporting "
            f"mode + language + VAD. Available: {sorted(MODEL_CONFIGS)}"
        )

    fallback = sorted(usable)[-1]
    logger.warning(
        "pipecat does not support Sarvam '%s'; using '%s' on this path "
        "(the LiveKit path is unaffected)",
        configured,
        fallback,
    )
    return fallback


def build_llm(system_instruction: str | None = None):
    """Gemini via Vertex AI by default, matching the LiveKit path.

    Vertex authenticates with a service account rather than an API key, which
    keeps the LLM inside the same GCP project and IAM boundary as everything
    else. Pipecat splits these across two classes where LiveKit used one flag,
    so the choice is explicit here.

    ``system_instruction`` is passed to the service rather than left as a system
    message in the context, and the two are mutually exclusive -- Pipecat warns
    and ignores the context message if both are set. Gemini sends it as a
    separate field ahead of the conversation, which makes it the front of the
    cacheable prefix.
    """
    settings = get_settings()

    if settings.llm_provider == "local":
        # Reuses the switch the LiveKit path already had -- `llm_provider` plus
        # the three `local_llm_*` values -- rather than a second one meaning the
        # same thing. Adding a parallel setting shadowed `local_llm_model` and
        # silently pointed `agent.py` at a model name its own base URL did not
        # serve, which is the failure mode duplicated configuration always has.
        #
        # Any OpenAI-compatible server works; vLLM and SGLang both expose one.
        # `system_instruction` becomes an ordinary system message on this path
        # (see DunningSession.__init__), because that is how the OpenAI shape
        # carries it -- Gemini's separate field is a Gemini nicety.
        from pipecat.services.openai.llm import OpenAILLMService

        logger.info(
            "using the local LLM at %s (%s)",
            settings.local_llm_base_url,
            settings.local_llm_model,
        )
        return OpenAILLMService(
            api_key=settings.local_llm_api_key,
            base_url=settings.local_llm_base_url,
            settings=OpenAILLMService.Settings(
                model=settings.local_llm_model,
                temperature=settings.llm_temperature,
            ),
        )

    if not settings.google_use_vertex:
        return GoogleLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            settings=GoogleLLMService.Settings(
                model=settings.gemini_model,
                temperature=settings.llm_temperature,
                system_instruction=system_instruction,
            ),
        )

    from pipecat.services.google.vertex.llm import GoogleVertexLLMService

    if not settings.google_cloud_project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for the Vertex path")

    return GoogleVertexLLMService(
        # Left to None so google-genai falls back to ADC, which is what
        # GOOGLE_APPLICATION_CREDENTIALS already sets up for the rest of the app.
        credentials_path=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        project_id=settings.google_cloud_project,
        location=settings.google_cloud_location,
        settings=GoogleVertexLLMService.Settings(
            model=settings.gemini_model,
            temperature=settings.llm_temperature,
            system_instruction=system_instruction,
        ),
    )


def sarvam_vad_overrides(settings) -> dict:
    """The saaras:v3 VAD parameters we have chosen to set, and only those.

    Unset means Sarvam's default, which is not the same as any value we could
    pick -- passing a guess for a parameter we have no opinion on would replace
    a tuned default with an untuned one. So only what is configured is sent.

    These are the fine-grained version of ``high_vad_sensitivity``. Both are
    used together on purpose: the flag makes Sarvam listen harder, and these
    put floors under what it is allowed to hear as speech. The failure they
    exist to fix is background noise arriving as three transcribed words, which
    is enough to clear the barge-in threshold and take the floor from the
    agent mid-sentence.
    """
    mapping = {
        "start_speech_volume_threshold": settings.sarvam_start_speech_volume_threshold,
        "min_speech_frames": settings.sarvam_min_speech_frames,
        "interrupt_min_speech_frames": settings.sarvam_interrupt_min_speech_frames,
        "positive_speech_threshold": settings.sarvam_positive_speech_threshold,
        "negative_speech_threshold": settings.sarvam_negative_speech_threshold,
        "pre_speech_pad_frames": settings.sarvam_pre_speech_pad_frames,
    }
    chosen = {k: v for k, v in mapping.items() if v is not None}
    if chosen:
        logger.info("sarvam VAD overrides: %s", chosen)
    return chosen


def build_services(language: str, system_instruction: str | None = None) -> tuple:
    """STT, LLM and TTS, configured exactly as the LiveKit path is."""
    settings = get_settings()

    if settings.local_speech_url.strip():
        return build_local_services(language, system_instruction)

    stt = SarvamSTTService(
        api_key=os.environ["SARVAM_API_KEY"],
        # 16 kHz: Sarvam performs best there, and native 8 kHz telephony audio
        # starves the model and garbles Hindi.
        sample_rate=16_000,
        # Sarvam kills idle sockets after ~60s; a customer thinking in silence
        # is enough to hit it.
        keepalive_timeout=30.0,
        keepalive_interval=5.0,
        settings=SarvamSTTService.Settings(
            model=resolve_sarvam_model(settings.sarvam_stt_model),
            language=settings.sarvam_language,
            # Deliberately OFF, which is the opposite of the LiveKit path.
            #
            # Sarvam's server VAD is good, but on Pipecat 1.7.0 nothing wires
            # its speech events into the user aggregator -- the service does not
            # override `service_metadata_frame`, so it never requests
            # `ExternalUserTurnStrategies` (that landed after this release).
            # Turning it on therefore does not hand Sarvam the turn; it just
            # adds a second, unheard opinion.
            #
            # Worse, `vad_signals=True` suppresses `flush_signal`, so when our
            # turn detector decides the customer has finished, nothing tells
            # Sarvam to finalise -- the turn then waits on the p99 fallback
            # timer instead of the transcript that was ready. Off, the flush
            # fires on our VAD's stop and the transcript lands immediately.
            vad_signals=False,
            # Independent of `vad_signals`, and worth having on a phone leg.
            #
            # Pipecat sends this as its own connection parameter whether or not
            # VAD signals are enabled, so it tunes how readily Sarvam's own
            # segmenter hears speech without handing it the turn. That is the
            # half we want: narrowband carrier audio carries 0.05-0.86% of its
            # energy above 4 kHz, which is why VADs under-trigger on the
            # customer channel while firing happily on our own TTS.
            #
            # Supported on saaras:v3, which is what resolve_sarvam_model()
            # actually returns here -- v4 is rejected by Pipecat 1.7.0.
            high_vad_sensitivity=True,
            **sarvam_vad_overrides(settings),
        ),
        mode="transcribe",
    )

    llm = build_llm(system_instruction)

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice=settings.cartesia_voice,
            # Cartesia pronounces digits according to the configured language,
            # so this has to match what the normalizers produce.
            language="hi" if language != "en" else "en",
        ),
        text_filters=[SpokenFormFilter(language)],
    )
    return stt, llm, tts


def build_local_services(language: str, system_instruction: str | None = None) -> tuple:
    """The same three services, served from our own GPU.

    Selected by setting `LOCAL_SPEECH_URL`; unset, nothing here runs and the
    vendor path above is untouched. That is the point of the switch being one
    branch in one function: the local stack is a deployment choice, not a fork
    of the agent, and everything above -- graph, walker, policy, outcome -- is
    identical either way.

    The LLM is deliberately still `build_llm`. Pointing it at the local Gemma
    is a `base_url` change inside that function, and doing it here would put
    two independent swaps behind one flag: if a call then went wrong there
    would be no way to tell which half did it.

    The same spoken-form filter is applied. Those rewrites turn digits and
    times into the words a Hindi speaker actually says, and they are a property
    of the language rather than of the vendor -- VoxCPM needs them exactly as
    much as Cartesia did.
    """
    settings = get_settings()
    base_url = settings.local_speech_url.strip()
    logger.info("using the local speech stack at %s", base_url)

    stt = LocalSTTService(
        base_url=base_url,
        language=language,
        # 16 kHz for the same reason Sarvam gets it: native 8 kHz telephony
        # audio starves an Indic model and the Hindi comes back garbled.
        sample_rate=16_000,
    )
    llm = build_llm(system_instruction)
    tts = LocalTTSService(
        base_url=base_url,
        text_filters=[SpokenFormFilter(language)],
    )
    return stt, llm, tts


def build_vad() -> SileroVADAnalyzer:
    """Blostem's production telephony values, mapped onto Pipecat's own names.

    Belongs on the user aggregator, not the transport. Pipecat 1.0 moved it,
    and because both param objects are Pydantic models that drop unknown
    fields, a `TransportParams(vad_analyzer=...)` is accepted in silence -- the
    bot starts, sounds configured, and never detects a turn.

    Sarvam's server-side VAD still drives turn-taking here (`vad_signals=True`
    makes the STT service request external turn strategies). This analyzer is
    what catches an interruption while the agent is mid-sentence.
    """
    settings = get_settings()
    return TimedSileroVAD(
        params=VADParams(
            confidence=settings.vad_activation_threshold,
            # NOT settings.vad_min_silence_duration (0.3), which is a 0.0.108
            # value and means something else there. In 1.x `stop_secs` is a
            # low-level detection threshold, not the wait before replying --
            # that lives in the stop strategy. Pipecat documents 0.2 and
            # calibrates every STT p99 against it; raising it collapses the
            # STT wait window and delays turns. The LiveKit path keeps 0.3,
            # where the parameter still has its old meaning.
            stop_secs=0.2,
            # 0.4 was Blostem's value, held high after they reverted an 0.2
            # experiment as too eager. We are back at 0.2 deliberately: the
            # word threshold that compensates for it is higher here than it
            # was there (min_words=3, vs their VAD-only start), so a cough or a
            # syllable of line noise still has to become three words before it
            # takes the floor. If the agent starts getting cut off mid
            # sentence, this is the first knob to put back.
            start_secs=0.2,
            min_volume=0.6,
        )
    )


def build_turn_strategies() -> UserTurnStrategies:
    """Who decides the customer has finished talking.

    Two decisions, both driven by how this specific call goes wrong.

    Stopping: a customer explaining why they could not pay does not speak in
    clean sentences -- "मेरे पास... अभी पैसे नहीं हैं" has a gap in the middle
    that a pure silence timer reads as the end of a turn. Cutting them off
    there is both rude and expensive: a truncated utterance is what the model
    labels, so we would file `retry_later` as a refusal. The smart-turn model
    judges completeness from the audio instead of the clock, and its own
    `stop_secs` is the backstop for when it stays unsure.

    Starting: `MinWordsUserTurnStartStrategy` replaces the VAD start strategy
    rather than joining it -- start strategies race, and a VAD start would fire
    on the first syllable and make the word count irrelevant. It only applies
    the threshold while the agent is speaking; once the agent is silent a
    single word starts the turn. That is exactly the behaviour this call needs,
    because Hindi speakers backchannel constantly ("हाँ", "जी", "अच्छा") and
    every one of those would otherwise cut the agent off mid-sentence. Three
    rather than two: two-word acknowledgements ("हाँ जी", "ठीक है", "अच्छा जी")
    are just as common as one-word ones, and were still taking the floor.

    `use_interim` is off, and deliberately so rather than by oversight. It asks
    the strategy to count words as partial transcripts stream in, which is what
    would let a real barge-in cut the agent off two words into an interruption.
    We do not get partials: `mode="transcribe"` with `vad_signals=False` yields
    one final transcript per turn, and production logs confirm every trigger
    arrives with `interim_transcription=False`. Leaving the flag on advertised
    streaming barge-in that the pipeline could not deliver.

    What the strategy still does, on finals, is the job it was chosen for: a
    one-word "हाँ" while the agent is speaking does not take the floor. What it
    no longer claims to do is interrupt mid-utterance -- with no partials, that
    is Silero's `start_secs` alone.
    """
    return UserTurnStrategies(
        start=[MinWordsUserTurnStartStrategy(min_words=3)],
        stop=[
            TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=TimedSmartTurnAnalyzer(
                    # Generous on purpose: this only applies when the model has
                    # classified the turn as *incomplete*, i.e. someone is
                    # visibly still thinking. Rushing them here is the failure
                    # mode we are buying our way out of.
                    params=SmartTurnParams(stop_secs=3.0)
                )
            )
        ],
    )




#: The closing line is generated only after the graph reaches a terminal node:
#: a Vertex round trip, then TTS, then several seconds of speech. It has to be
#: given time to start before we can sensibly wait for it to end.
CLOSING_START_TIMEOUT = 4.0
#: And a ceiling on the whole thing, so a TTS failure cannot hold the line open
#: on a customer who has already been told everything they need.
CLOSING_TOTAL_TIMEOUT = 20.0
#: A little air after the last audio frame. Hanging up on the final syllable
#: reads as a dropped call rather than a finished one.
CLOSING_GRACE = 0.6


async def _let_the_closing_line_finish(session) -> None:
    """Hold the line until the agent has actually stopped speaking.

    This used to be `sleep(2.0)`, which was a guess, and a wrong one: the
    closing turn is produced *after* the terminal transition, so two seconds cut
    the customer off about a second into "I am sending you a secure payment link
    for 499 rupees, thank you". On a call where they had just agreed to pay, the
    last thing they heard was the line going dead.

    The in-flight line has to be let go first. A terminal transition is decided
    *while the agent is still speaking* the turn that carried it -- on
    voice_call 31 that was "ठीक है, मैं आपको पेमेंट लिंक भेज रहा हूँ" -- so
    `speaking` is already set when this is called. Waiting on it returned
    immediately, the next wait caught the end of that same sentence, and the
    call hung up before the closing line had even been generated. The customer
    agreed to pay, the link went out, and the line went dead with no goodbye.

    So: let the current utterance finish, then wait for the *next* one. The
    closing line needs a Vertex round trip before it can start, which is ample
    room between the two waits.

    Every wait is bounded. A closing line that never arrives, or never ends,
    must not keep a customer on a call that is over.
    """
    if session.speaking.is_set():
        try:
            await asyncio.wait_for(
                session.stopped_speaking.wait(), timeout=CLOSING_TOTAL_TIMEOUT
            )
        except TimeoutError:
            logger.warning("the turn that ended the call never stopped playing; ending")
            return

    # Re-arm, so the next wait catches the closing line rather than the one
    # that just finished.
    session.speaking.clear()
    session.stopped_speaking.clear()

    try:
        await asyncio.wait_for(session.speaking.wait(), timeout=CLOSING_START_TIMEOUT)
    except TimeoutError:
        logger.info("no closing line began within %.1fs; ending the call", CLOSING_START_TIMEOUT)
        return

    try:
        await asyncio.wait_for(session.stopped_speaking.wait(), timeout=CLOSING_TOTAL_TIMEOUT)
    except TimeoutError:
        logger.warning(
            "closing line still playing after %.0fs; ending anyway", CLOSING_TOTAL_TIMEOUT
        )
        return

    await asyncio.sleep(CLOSING_GRACE)


class AnnouncementFilter(FrameProcessor):
    """Drops transcripts that are the network talking, not the customer.

    Sits between STT and the user aggregator, so an announcement never becomes
    a conversation turn at all -- rather than being seen and then reasoned
    about, which is how it satisfied the identity gate on voice_call 31.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if is_carrier_announcement(frame.text):
                logger.info("dropped a carrier announcement: %r", frame.text[:60])
                return

        await self.push_frame(frame, direction)


class SpeechWatcher(FrameProcessor):
    """Tracks whether the agent is currently speaking.

    Exists because the call used to end on a fixed `sleep(2.0)` after the graph
    reached a terminal node. The closing line is generated *after* that
    transition -- a Vertex round trip, then TTS, then several seconds of Hindi
    -- so two seconds cut the customer off roughly a second into "I am sending
    you a secure payment link for 499 rupees, thank you". They heard the call
    drop mid-sentence, immediately after agreeing to pay.

    Placed after `transport.output()`, so the frames it watches describe audio
    that has actually gone to the caller rather than audio that has merely been
    synthesised.
    """

    def __init__(self, session) -> None:
        super().__init__()
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._session.speaking.set()
            self._session.stopped_speaking.clear()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._session.speaking.clear()
            self._session.stopped_speaking.set()
        await self.push_frame(frame, direction)


class DunningSession:
    """One call: the graph, the context, and the tool that moves between nodes."""

    def __init__(self, context: dict) -> None:
        self.walker = GraphWalker(DUNNING_FLOW, context)
        self.language = context.get("_language", "hi")
        self.llm_context = LLMContext()
        # No system message here on the Gemini path: the preamble goes to the
        # service as `system_instruction` (see build_llm), and Gemini would
        # ignore a second one anyway. The context holds only the conversation,
        # which is what makes it append-only and therefore cacheable.
        #
        # The local path has no such field. vLLM speaks the OpenAI shape, where
        # the system prompt is simply the first message -- so it goes in here
        # instead. Still append-only, still cacheable; only the carrier differs.
        opening: list[dict] = []
        if get_settings().llm_provider == "local":
            opening.append({"role": "system", "content": self.walker.preamble()})
        opening.append(self._stage_message())
        self.llm_context.set_messages(opening)
        self.finished = asyncio.Event()
        #: Set while the agent is speaking, cleared when it stops. Read at the
        #: end of the call so the closing line is not cut off mid-sentence.
        self.speaking = asyncio.Event()
        self.stopped_speaking = asyncio.Event()
        self.recovery_case_id: int | None = context.get("_recovery_case_id")
        self.amount_paise: int | None = context.get("_amount_paise")
        self.voice_call_id: int | None = None
        self.started_at = utcnow()
        self._finalised = False
        #: Consecutive rejected transitions since the last accepted one.
        self._rejections = 0
        #: Errors raised by the TTS service -- lines the agent believes it
        #: delivered and the customer never heard. See `note_tts_failure`.
        self.tts_failures: list[str] = []
        #: Set once the pipeline is built; None on the paths that run without one.
        self.backchannel_gate: BackchannelProcessor | None = None

    def tools(self) -> ToolsSchema:
        """One tool, declared once, listing every label in the flow.

        Deliberately not regenerated per node, for two reasons that point the
        same way.

        The tool schema is sent on every request alongside the system
        instruction, at the front of what a provider hashes for its prefix
        cache. A schema that changes at each stage changes that prefix, so it
        would cost a cache hit on the turn after every transition -- the exact
        turns where the context is longest.

        It was also simply wrong. `set_tools` is called once at startup, so the
        per-node version froze at the greet node's labels and then disagreed
        with the instructions at every later stage: the schema offered
        `identity_confirmed` while the node was asking for `acknowledged`. It
        only worked because Gemini treats an enum as guidance.

        Which labels are legal *right now* is not the schema's job. The node
        instructions name them, and `GraphWalker.transition` refuses anything
        else -- so a wrong label costs a rejected tool call, not a wrong branch.
        """
        graph = self.walker.graph
        conditions = {
            edge.label: edge.condition for node in graph.nodes for edge in node.edges
        }
        return ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="transition",
                    description=(
                        "Move the conversation to the next stage. Call this as soon "
                        "as a label matches what the customer just said; if none "
                        "does yet, ask one short clarifying question instead. Only "
                        "the labels listed in your CURRENT stage instructions are "
                        "accepted -- this list is every label in the call, and most "
                        "of them belong to stages you are not at. Any other label "
                        "is rejected and you will be asked again."
                    ),
                    properties={
                        "label": {
                            "type": "string",
                            "enum": list(graph.labels),
                            "description": "\n".join(
                                f"{label}: {conditions[label]}" for label in graph.labels
                            ),
                        }
                    },
                    required=["label"],
                )
            ]
        )

    async def on_transition(self, params) -> None:
        """Walk one edge, then re-point the model at the new node."""
        label = (params.arguments or {}).get("label", "")
        try:
            self.walker.transition(label, utterance=self._last_user_said())
        except InvalidTransition as exc:
            await self._reject_transition(params, label, exc)
            return

        self._rejections = 0
        logger.info("transition %s -> %s", label, self.walker.node.id)
        self.apply_backchannel_policy()

        # The handoff rides on the tool result rather than a rewritten system
        # message. Rewriting changed the prefix on every stage change, so every
        # turn after a transition was a guaranteed cache miss; this only ever
        # appends, so the prefix the model has already seen stays byte-identical.
        # Framed as a directive, not a data blob. A tool result carries far less
        # weight with the model than a system instruction, and the first version
        # of this returned {"moved_to", "instructions"} -- which the model read
        # as information and then went back to whatever the system prompt last
        # told it to do, re-greeting the customer on every turn.
        # Only ask the model again if it has not already spoken.
        #
        # A transition is a tool call that produces no audio, so by default the
        # model has to be run a second time before the customer hears anything
        # -- two Vertex round trips for one turn, measured at +0.565s (1.87x)
        # against turns that did not transition.
        #
        # The prompt asks it to answer *and* transition in the same response.
        # When it complies there is nothing left to generate, so suppressing
        # the follow-up costs a round trip and changes nothing the caller
        # hears. When it does not comply, the follow-up is the only thing
        # standing between the customer and silence, so this checks rather than
        # assumes: batching is an optimisation, not a promise the model keeps.
        # Unconditional, and the check above is kept only as a measurement.
        # Two live calls settled it: whatever the model batches alongside the
        # transition is written against the stage it is LEAVING, so it cannot
        # contain the line the new stage exists to deliver. Suppressing the
        # follow-up did not save a redundant sentence -- it dropped the only
        # sentence that mattered, and the new stage sat undelivered until the
        # customer spoke again. Measured at 5.7s of dead air after an
        # acknowledgement, and 3030ms of logged silence at the next
        # transition, both ended by the customer saying "हेलो" to check the
        # line was still open.
        #
        # Showing the model every stage up front would let one inference both
        # move and speak, but the map would carry the amount into the greet
        # stage -- before identity is confirmed -- and reveal the later options
        # before the reason is asked. Two tests forbid exactly that, so the
        # round trip stays. It costs ~0.55s; the silence cost 5.7s.
        spoke = self._model_spoke_this_turn()
        await params.result_callback(
            {
                "previous_stage": "finished -- do not repeat it",
                "now_at_stage": self.walker.node.id,
                "do_this_now": self.walker.stage_instructions(),
            },
            properties=FunctionCallResultProperties(run_llm=True),
        )
        logger.info(
            "transition -> %s; model %s alongside the tool call (%r); "
            "follow-up %s",
            self.walker.node.id,
            "spoke" if spoke else "stayed silent",
            self._assistant_text_this_turn()[:60],
            "running regardless",
        )

        if self.walker.finished:
            self.finished.set()

    async def _reject_transition(self, params, label: str, exc: InvalidTransition) -> None:
        """Refuse the move and give the model what it needs to correct itself.

        The graph refusing an illegal label was never in doubt. What went wrong
        on a live call is what came next: the rejection was returned as a bare
        `{"error": ...}`, the model read it as information rather than as a
        thing to act on, and the call ended at a node it had left conversation-
        ally ten turns earlier -- recording nothing for a customer who had
        agreed to pay.

        So it is framed the way an accepted transition is: where you are, and
        what you may do from here. The node's own prompt is deliberately left
        out. It opens with the line the model has already delivered, and
        re-sending it is how you get an agent that announces the failed payment
        twice.

        Re-running the model is capped. A rejection is one of the few places
        the model can answer its own output, and a run of invalid labels would
        otherwise spend Vertex calls and dead air on a loop the customer sits
        through. Past the cap the conversation simply carries on: the graph has
        not moved, and the next customer turn is another chance at the right
        label.
        """
        self._rejections += 1
        retrying = self._rejections <= MAX_TRANSITION_RETRIES
        logger.warning(
            "rejected transition %r (%d in a row, %s): %s",
            label,
            self._rejections,
            "asking again" if retrying else "letting the call continue",
            exc,
        )
        await params.result_callback(
            {
                "not_moved": str(exc),
                "still_at_stage": self.walker.node.id,
                "do_this_now": (
                    "You have not moved. Do not repeat what you have already "
                    "said. Choose the label below that matches what the "
                    "customer just said, or ask one short clarifying question "
                    f"instead.\n\nAvailable labels:\n{self.walker.moves()}"
                ),
            },
            properties=FunctionCallResultProperties(run_llm=retrying),
        )

    def opening_line(self) -> str | None:
        """The greeting to speak on connect, or None to let the model write it.

        None when the setting is off, or when the call did not start at the
        greeting node -- a resumed or replayed call is mid-conversation, and
        opening it with "hello, is that Asha" would be wrong.
        """
        if not get_settings().cached_greeting_enabled:
            return None
        if self.walker.node.kind is not NodeKind.START:
            return None
        return greeting_for(self.language, self.walker.context)

    def note_tts_failure(self, error: str) -> None:
        """Record that a line the agent 'said' never reached the customer.

        Only the first is logged. A quota that has run out fails every
        utterance for the rest of the call, and twenty identical lines buries
        the one that matters.
        """
        if not self.tts_failures:
            logger.error("nothing the agent says is being heard: %s", error)
        self.tts_failures.append(error)

    def note_greeting_spoken(self, text: str) -> None:
        """Strike out the part of the greet node we have already performed.

        The node's instruction reads "Open the call. Greet them" -- so the
        model, doing exactly what it is told, greets a second time. That is not
        a hypothetical: it happened on a live call and the customer heard the
        introduction twice. Negating it does not work either; "do not greet
        again" directly above "Open the call. Greet them" is a contradiction,
        and the model resolves it by greeting. So the directive is replaced.

        Deliberately does NOT add the greeting to the context as an assistant
        turn, which is what it used to do. The greeting is spoken by pushing a
        ``TTSSpeakFrame`` through the pipeline, and ``aggregators.assistant()``
        sits at the end of that pipeline -- so it records the line on its way
        past, exactly as it records anything else the bot says. Writing it here
        as well put the greeting in the context twice.

        The customer never heard it twice; the model did. On a live call the
        second copy landed *after* the customer's reply, so the history the
        model reasoned from showed it repeating itself the moment someone
        answered -- which is not a habit worth teaching it, on a call whose
        next job is to discuss the customer's money.

        The gap this leaves is the turn between the greeting being spoken and
        the aggregator committing it. ``ALREADY_GREETED`` covers exactly that:
        it states outright that the introduction has happened, so a model that
        cannot yet see the line in its history is still told not to repeat it.
        """
        messages = self.llm_context.get_messages()
        for i, message in enumerate(messages):
            content = message.get("content")
            if isinstance(content, str) and content.startswith("[STAGE: "):
                messages[i] = {
                    "role": "user",
                    "content": (
                        f"[STAGE: {self.walker.node.id}]\n"
                        f"{self.walker.stage_instructions(override_prompt=ALREADY_GREETED)}"
                    ),
                }
                break
        self.llm_context.set_messages(messages)

    def _model_spoke_this_turn(self) -> bool:
        """Did the model produce speech in the same response as the tool call?

        Reads the context rather than trusting the prompt. The assistant's
        latest message carries the tool call; if it also carries non-empty
        text, the customer is already being spoken to and running the LLM
        again would only add a second sentence they did not need.

        An echo does not count. This asked only whether the text was non-empty,
        and on a live call the model answered "हाँ जी।" to a customer who had
        just said "हाँ जी।" -- which passed, suppressed the follow-up run, and
        left the echo as the whole turn. The real answer then arrived one turn
        late, every time, for the rest of the call.

        Treating an echo as silence is the conservative direction: the cost of
        being wrong is one extra Vertex round trip, and the cost of the old
        behaviour was a customer hearing themselves instead of an answer.
        """
        spoken = self._assistant_text_this_turn()
        if not spoken:
            return False

        heard = self._last_user_said()
        if heard and _is_echo(spoken, heard):
            logger.warning(
                "model echoed the customer (%r); treating the turn as silent "
                "so the follow-up run produces a real reply",
                spoken[:80],
            )
            return False

        # An acknowledgement is not the turn's content. The greet stage asks
        # for exactly one -- "जी शुक्रिया।" -- and counting it as speech
        # suppressed the follow-up run, so the stage we had just moved to was
        # never delivered. Measured on a live call: 5.7s of dead air after the
        # ack, ending only because the customer said "हेलो" to check we were
        # still there, and 3030ms of silence at the next transition.
        #
        # Letting the follow-up run costs one Vertex round trip (~0.5s) and
        # produces what a person would actually say: the acknowledgement and
        # then the reason for the call, in one breath.
        if len(_ECHO_STRIP.sub("", spoken)) <= ACK_MAX_CHARS:
            logger.info(
                "model said only an acknowledgement (%r); running the follow-up "
                "so the new stage is delivered without waiting for the customer",
                spoken[:40],
            )
            return False
        return True

    def _assistant_text_this_turn(self) -> str:
        """The text on the assistant's latest message, across content shapes."""
        for message in reversed(self.llm_context.get_messages()):
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return " ".join(
                    str(part.get("text", "")).strip()
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ).strip()
            return ""
        return ""

    def _stage_message(self) -> dict:
        """The opening stage, as the first turn of the conversation.

        A `system` role would collide with `system_instruction`; Gemini keeps
        the latter and warns. `user` would read as the customer speaking. The
        stage is framed as a directive the assistant has just been handed.
        """
        return {
            "role": "user",
            "content": f"[STAGE: {self.walker.node.id}]\n{self.walker.stage_instructions()}",
        }

    def apply_backchannel_policy(self) -> None:
        """Silence the listening sounds where agreement would be misread.

        Called on every transition rather than set once, because a call can
        reach a sensitive node from several directions -- `disputes_charge`
        exists on three different nodes.
        """
        if self.backchannel_gate is None:
            return
        allowed = self.walker.node.id not in SILENT_NODES
        if self.backchannel_gate.enabled != allowed:
            logger.info(
                "backchannel %s at node '%s'",
                "enabled" if allowed else "silenced",
                self.walker.node.id,
            )
        self.backchannel_gate.enabled = allowed

    def _last_user_said(self) -> str | None:
        for message in reversed(self.llm_context.get_messages()):
            if message.get("role") == "user":
                content = message.get("content")
                return str(content).strip() or None
        return None

    async def finalise(self) -> None:
        """Apply the detected intent to the case, exactly once.

        Called when the graph reaches a terminal node, and again after the
        pipeline unwinds so a caller who hangs up mid-conversation still leaves
        a recorded outcome. The flag is what makes the second call free.
        """
        if self._finalised:
            return
        self._finalised = True

        intent = self.walker.intent or CallIntent.UNCLEAR
        status = CallStatus.COMPLETED
        error = None
        if self.tts_failures:
            # The customer answered questions they never heard. Everything
            # downstream of them kept working -- the model spoke into the void,
            # the graph advanced on their replies, and the walker arrived at an
            # intent with the same confidence as a call that went perfectly.
            #
            # That confidence is the danger, and it is why TTS is singled out.
            # An STT or LLM failure is silence in both directions: the graph
            # does not advance and the intent stays unclear on its own. Only a
            # mute agent produces a *wrong* answer rather than no answer, and
            # `declined` or `dispute` would suppress this customer for good.
            #
            # So the intent is discarded rather than trusted. Unclear leaves
            # the case open for another attempt, bounded by the attempt cap:
            # the cost of being wrong here is one more call, against never
            # calling someone who never heard us ask.
            logger.error(
                "discarding intent '%s' from call %s: %d line(s) never reached "
                "the customer",
                intent,
                self.voice_call_id,
                len(self.tts_failures),
            )
            intent = CallIntent.UNCLEAR
            status = CallStatus.FAILED
            error = f"{len(self.tts_failures)} TTS failures; first: {self.tts_failures[0]}"

        await finalise_call(
            voice_call_id=self.voice_call_id,
            recovery_case_id=self.recovery_case_id,
            result=CallResult(
                intent=intent,
                status=status,
                final_node_id=self.walker.node.id,
                transcript=self._transcript(),
                duration_seconds=await duration_since(self.started_at),
                answered_at=self.started_at,
                error=error,
                transitions=self.walker.observations_as_dicts(),
            ),
        )

    def _transcript(self) -> str | None:
        """Flatten the context. Evidence only -- never read back in."""
        lines = []
        for message in self.llm_context.get_messages():
            if message.get("role") not in ("user", "assistant"):
                continue
            content = message.get("content")
            if isinstance(content, list):
                content = " ".join(str(part) for part in content)
            text = str(content or "").strip()
            if text:
                lines.append(f"{message['role']}: {text}")
        return "\n".join(lines) or None


def call_context(*, customer_name: str, amount_paise: int, failure_reason: str,
                 language: str, company: str, subscription_halted: bool = False,
                 route: str | None = None) -> dict:
    context = {
        "company_name": company,
        "customer_name": customer_name,
        "amount_spoken": spoken_amount(amount_paise, language),
        "failure_reason": failure_reason,
        "language_hint": language_hint(language),
        "halt_note": halt_note(subscription_halted),
        "suggested_route": route,
    }
    context["_language"] = language
    # The digits, kept alongside the spoken form. `amount_spoken` is words and
    # is what the agent says; the guardrail compares figures, and needs the
    # number the words were rendered from rather than the words.
    context["_amount_paise"] = amount_paise
    return context


def _truthy(value) -> bool:
    """Read a flag that may have crossed a wire as a string.

    LiveKit carries JSON and hands back a real bool; Twilio delivers every
    custom parameter as a string, where a bare ``bool()`` turns ``"false"`` into
    True. Getting this wrong tells the agent a subscription is halted when it is
    not, which changes what it says on a live money call.
    """
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no"}
    return bool(value)


def _case_id(value) -> int | None:
    """The case id, as an int, however it arrived.

    Twilio delivers custom parameters as strings, so this crosses the wire as
    "7". Left as a string it reaches `open_call_record` and
    `send_payment_link_now` as a database parameter for an integer column, where
    asyncpg rejects it -- and `persistence.py` swallows its own failures on
    purpose, so nothing would be raised. The call would sound perfect and
    quietly never send the payment link.
    """
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("ignoring unusable recovery_case_id %r", value)
        return None


def context_from_body(body: dict) -> dict:
    """Build the call context from the runner's request body.

    The Pipecat equivalent of LiveKit job metadata: `app/recover.py` puts the
    case on the wire, this reads it off. Keys are the same on both paths so one
    dispatcher can drive either.
    """
    settings = get_settings()
    language = body.get("preferred_language") or "hi"

    context = call_context(
        customer_name=body.get("customer_name", "there"),
        amount_paise=int(body.get("amount_paise", 0)),
        failure_reason=body.get("failure_reason") or spoken_reason(None, language),
        language=language,
        company=body.get("company_name", settings.company_name),
        subscription_halted=_truthy(body.get("subscription_halted")),
        route=body.get("suggested_route"),
    )
    # Pre-rendered amounts win over our own: the dispatcher may have spoken
    # forms we cannot reconstruct from paise alone.
    if body.get("amount_spoken"):
        context["amount_spoken"] = body["amount_spoken"]
    context["_recovery_case_id"] = _case_id(body.get("recovery_case_id"))
    context["_phone"] = body.get("phone")
    return context


#: Exactly the tools the model may call. Adding one here is not optional --
#: an unlisted tool is refused at the point of use, after the model has
#: already told the customer it is doing something.
MCP_TOOLS = ["send_payment_link", "send_mandate_link", "get_case"]

#: How many rejected transitions in a row still earn the model another turn to
#: pick a legal label. Two is one honest correction plus one, past which it is
#: not reading the rejection and the customer is listening to us think.
MAX_TRANSITION_RETRIES = 2


def recovery_mcp(recovery_case_id: int | None) -> MCPClient:
    """Our own MCP server, over stdio, bound to one case.

    Runs as a child process of the agent, so it shares the environment -- the
    same DATABASE_URL and Razorpay keys -- without those ever crossing a
    network. `tools_filter` is explicit: the LLM gets exactly these and nothing
    a future tool might add by accident.

    That guard has already caught something, at the cost of a live call. A
    fourth tool was added and the flow was told to use it, but this list was
    not updated -- so Pipecat refused the call with "not in the currently
    advertised tool set" and the customer was told about a link that was never
    sent. `test_every_tool_the_flow_asks_for_is_advertised` now fails if the
    two ever drift apart again.

    The case travels in that environment rather than as a tool argument. When
    it was an argument the model had to supply an integer nobody had told it,
    and it invented one -- `send_payment_link(recovery_case_id=12345)` against
    a case numbered 1. A wrong guess that happened to land on a real case would
    have sent someone else's payment link to whoever was on this call.

    Pipecat 1.7.0 has no `tools_arguments` to bind it client-side (that landed
    later), and the environment is the seam we already own: one server process
    per call, spawned by us.
    """
    env = dict(os.environ)
    if recovery_case_id is not None:
        env["DUNNING_CASE_ID"] = str(recovery_case_id)
    else:
        # A demo run with no case. Clear any inherited value rather than let
        # the sample conversation act on whatever the last call was bound to.
        env.pop("DUNNING_CASE_ID", None)

    return MCPClient(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp_server"],
            env=env,
        ),
        tools_filter=MCP_TOOLS,
    )


def watch_tts_failures(tts, session: DunningSession) -> None:
    """Tell the session when a line never reached the customer.

    Bound to the TTS service's own `on_error` rather than the task's
    `on_pipeline_error`, which carries every processor's errors and would have
    to pick this one out by identity.
    """

    @tts.event_handler("on_error")
    async def _on_tts_error(_service, frame) -> None:
        session.note_tts_failure(str(frame.error))


async def run_call(transport, session: DunningSession) -> DunningSession:
    """Wire and run one conversation. Returns the session for its outcome."""
    # The preamble is fixed for the whole call, so it is handed to the service
    # once and never re-sent -- that is the prefix Gemini caches.
    stt, llm, tts = build_services(session.language, session.walker.preamble())

    # `transition` stays in-process: it is graph traversal, not an external
    # capability, and it must not be able to fail over a transport.
    llm.register_function("transition", session.on_transition)

    aggregators = LLMContextAggregatorPair(
        session.llm_context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=build_vad(),
            user_turn_strategies=build_turn_strategies(),
        ),
    )

    # The money tools come from our MCP server, so any MCP-speaking agent gets
    # the same guarantees rather than a second implementation of them.
    async with recovery_mcp(session.recovery_case_id) as mcp:
        mcp_tools = await mcp.register_tools(llm)
        session.llm_context.set_tools(
            ToolsSchema(
                standard_tools=(
                    session.tools().standard_tools + mcp_tools.standard_tools
                )
            )
        )
        return await _run_pipeline(session, transport, stt, llm, tts, aggregators)


async def _run_pipeline(session, transport, stt, llm, tts, aggregators) -> DunningSession:

    processors = [
        transport.input(),
        stt,
        AnnouncementFilter(),
        aggregators.user(),
        llm,
        tts,
        transport.output(),
        SpeechWatcher(session),
        aggregators.assistant(),
    ]

    # Between the model and the voice: the last point at which a sentence is
    # still text. `build_guardrail` returns None when the mode is "off", which
    # is the default, so this is a no-op unless someone turned it on.
    guardrail = build_guardrail(
        get_settings().guardrails_mode,
        expected_amount_rupees=(
            f"{session.amount_paise / 100:.2f}" if session.amount_paise else None
        ),
    )
    if guardrail is not None:
        processors.insert(processors.index(tts), guardrail)
        logger.info("guardrails active in %s mode", get_settings().guardrails_mode)

    # Backchannel wraps the list and inserts its own processors -- it is not a
    # service we place ourselves. Keep the returned list; the gate has to be
    # found inside it.
    processors = get_backchannel(session.language)(processors)
    session.backchannel_gate = next(
        (p for p in processors if isinstance(p, BackchannelProcessor)), None
    )
    # The opening node is not sensitive, but a call can be resumed or replayed
    # into one, so the policy is applied from the start rather than assumed.
    session.apply_backchannel_policy()

    pipeline = Pipeline(processors)

    watch_tts_failures(tts, session)

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        opening = session.opening_line()
        if opening is None:
            await task.queue_frames([LLMRunFrame()])
            return
        # Speak it directly and record it as ours. No LLM turn: there is no
        # customer input yet to reason about, and the round trip is half a
        # second of silence at the moment they have just said hello.
        session.note_greeting_spoken(opening)
        logger.info("opened with the rendered greeting")
        await task.queue_frames([TTSSpeakFrame(opening)])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        await task.queue_frame(EndFrame())

    async def _end_when_finished() -> None:
        await session.finished.wait()
        # Write the outcome while the closing line is still playing. It costs
        # the customer nothing, and a pipeline that dies during teardown would
        # otherwise lose the one thing the call was for.
        await session.finalise()
        await _let_the_closing_line_finish(session)
        await task.queue_frame(EndFrame())

    watcher = asyncio.create_task(_end_when_finished())
    try:
        await PipelineRunner().run(task)
    finally:
        watcher.cancel()
    return session


#: Only the transports we can actually exercise. WebRTC is the browser demo;
#: the carriers are the reason Pipecat is here at all -- all three are plain
#: websocket transports whose serializer `create_transport` fills in, so adding
#: them costs a line each rather than a SIP trunk negotiation.
#:
#: Twilio is the one to reach for first: a trial account issues a number without
#: the KYC an Indian number needs, and `create_transport` recognises Twilio's
#: handshake and builds the serializer from TWILIO_ACCOUNT_SID/AUTH_TOKEN in the
#: environment. `app/voice/telephony.py` places the call that arrives here.
#:
#: No VAD here: since 1.0 it lives on the user aggregator (see `build_vad`).
TRANSPORT_PARAMS = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=build_noise_filter(),
    ),
    "twilio": lambda: _telephony_params(),
    "plivo": lambda: _telephony_params(),
    "exotel": lambda: _telephony_params(),
}


def build_noise_filter():
    """Denoise the customer's audio before Sarvam ever sees it.

    Chosen because it is the only filter with no vendor key: RNNoise runs
    locally, so nothing about a billing call leaves our process to be cleaned
    up. Krisp and Koala are better but each needs an account, and AIC needs
    one too.

    This matters more than it would on a laptop demo. Recovery calls land on
    Indian mobile networks, and the background -- traffic, a shop, a television
    -- is what drives Sarvam to hallucinate words that then get labelled as an
    intent. Cleaning the input is cheaper than correcting a wrong branch.

    Never fatal: a noisy call still recovers money.
    """
    try:
        from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter

        return RNNoiseFilter()
    except Exception:  # noqa: BLE001 - denoising is not worth failing a call
        logger.warning("RNNoise unavailable; running without noise suppression")
        return None


def _telephony_params():
    """Imported lazily: pipecat[websocket] is not needed for the WebRTC demo."""
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

    return FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=build_noise_filter(),
    )


async def _demo_body() -> dict:
    """The body for a browser run, which arrives with none of its own.

    `DUNNING_DEMO_CASE_ID` points the demo at a real row, so a call from the
    browser exercises the whole loop -- link sent, case updated, action logged --
    rather than only the conversation. Without it we fall back to the sample,
    which writes nothing and is labelled as such in the log.
    """
    case_id = os.environ.get("DUNNING_DEMO_CASE_ID")
    if case_id:
        body = await load_call_body(int(case_id))
        if body:
            logger.info("demo call against recovery case %s", case_id)
            return body
        logger.warning("case %s not found; falling back to the sample", case_id)

    logger.warning("no runner body; using the sample call context (nothing is persisted)")
    return SAMPLE_BODY


def telephony_body(runner_args: RunnerArguments) -> dict | None:
    """The case, as the carrier delivered it on the handshake.

    Twilio carries our TwiML `<Parameter>` entries as `customParameters`, and
    Pipecat parses them onto `runner_args.call_data` while building the
    transport -- *not* onto `runner_args.body`, which only the Daily and CLI
    paths populate.

    Reading the wrong one fails silently, which is why this is its own
    function: the call still connects and the agent still talks, but it talks
    to the sample customer about the sample amount. On a real recovery call
    that is a stranger being told someone else's billing details.
    """
    call_data = getattr(runner_args, "call_data", None)
    if not call_data:
        return None
    try:
        return call_data.get("body") or None
    except Exception:  # noqa: BLE001 - a malformed handshake must not kill the call
        logger.exception("could not read the call body from the handshake")
        return None


def telephony_provider(runner_args: RunnerArguments) -> str:
    """Who carried this call, for the audit trail. Detected from the handshake."""
    return getattr(runner_args, "transport_type", None) or "pipecat"


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point. The Pipecat runner discovers this by name.

        uv run --group pipecat python -m app.voice.pipecat_agent -t webrtc

    The body carries the case, exactly as job metadata does on LiveKit.

    The transport is built first because on a telephony leg the case *arrives*
    with the transport: `create_transport` parses the carrier's handshake, and
    the customParameters are only readable afterwards. Nothing is lost by the
    reordering -- the call record is still written before any audio flows.
    """
    transport = await create_transport(runner_args, TRANSPORT_PARAMS)

    body = runner_args.body or telephony_body(runner_args) or await _demo_body()

    # Raises before any audio flows if the context is incomplete, rather than
    # reading a literal "{customer_name}" down the line.
    session = DunningSession(context_from_body(body))
    session.voice_call_id = await open_call_record(
        recovery_case_id=session.recovery_case_id,
        room_name=body.get("room_name", "pipecat"),
        dialled_number=body.get("phone"),
        provider=telephony_provider(runner_args),
    )

    # Before the pipeline runs, so the clips are on disk by the time audio flows.
    await prewarm_backchannel(session.language)

    try:
        await run_call(transport, session)
    finally:
        # A customer who hangs up mid-sentence never reaches a terminal node.
        # finalise() is idempotent, so this only fires when the graph did not.
        await session.finalise()


__all__ = ["DunningSession", "SpokenFormFilter", "bot", "build_backchannel",
           "build_llm", "build_services", "build_turn_strategies", "build_vad",
           "call_context", "context_from_body", "get_backchannel",
           "normalize_for_speech", "prewarm_backchannel", "run_call",
           "telephony_body", "telephony_provider"]


if __name__ == "__main__":
    from pipecat.runner.run import main

    # Pipecat's startup banner is drawn with box characters, and the Windows
    # console defaults to cp1252, which cannot encode them -- the runner dies
    # on its own banner before the agent ever binds a port. Same guard as
    # app/report.py uses for rupee signs.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO)
    main()
