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
import sys

from dotenv import load_dotenv
from mcp import StdioServerParameters
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.mcp_service import MCPClient
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.text.base_text_filter import BaseTextFilter

from app.config import get_settings
from app.constants import CallStatus
from app.store import utcnow
from app.voice.flow import DUNNING_FLOW, language_hint
from app.voice.intents import CallIntent
from app.voice.normalizers import (
    normalize_amounts_for_tts,
    normalize_for_hindi_tts,
    normalize_times_for_tts,
    strip_markdown,
)
from app.voice.outcomes import CallResult
from app.voice.persistence import duration_since, finalise_call, open_call_record
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


def build_services(language: str) -> tuple:
    """STT, LLM and TTS, configured exactly as the LiveKit path is."""
    settings = get_settings()

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
            model=settings.sarvam_stt_model,
            language=settings.sarvam_language,
            # Server VAD emits END_SPEECH the moment the speaker stops. Without
            # it the framework falls back to a silence heuristic, which is what
            # drove Blostem's STT p95 to 1.3s.
            vad_signals=True,
            high_vad_sensitivity=True,
        ),
        mode="transcribe",
    )

    llm = GoogleLLMService(
        model=settings.gemini_model,
        params=GoogleLLMService.InputParams(temperature=settings.llm_temperature),
    )

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        voice_id=settings.cartesia_voice,
        # Cartesia pronounces digits according to the configured language, so
        # this has to match what the normalizers produce.
        params=CartesiaTTSService.InputParams(language="hi" if language != "en" else "en"),
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
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=settings.vad_activation_threshold,
            stop_secs=settings.vad_min_silence_duration,
            start_secs=0.4,
            min_volume=0.6,
        )
    )


class DunningSession:
    """One call: the graph, the context, and the tool that moves between nodes."""

    def __init__(self, context: dict) -> None:
        self.walker = GraphWalker(DUNNING_FLOW, context)
        self.language = context.get("_language", "hi")
        self.llm_context = LLMContext()
        self.llm_context.set_messages(
            [{"role": "system", "content": self.walker.instructions()}]
        )
        self.finished = asyncio.Event()
        self.recovery_case_id: int | None = context.get("_recovery_case_id")
        self.voice_call_id: int | None = None
        self.started_at = utcnow()
        self._finalised = False

    def tools(self) -> ToolsSchema:
        """One tool, whose enum is regenerated per node.

        The model picks a label, never a destination -- an unknown label, or one
        valid elsewhere but not here, is refused.
        """
        return ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="transition",
                    description=(
                        "Move the conversation to the next stage. Call this only when "
                        "the customer's position is clear; if it is still ambiguous, "
                        "ask one short clarifying question instead."
                    ),
                    properties={
                        "label": {
                            "type": "string",
                            "enum": [e.label for e in self.walker.options],
                            "description": "\n".join(
                                f"{e.label}: {e.condition}" for e in self.walker.options
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
            logger.warning("rejected transition %r: %s", label, exc)
            await params.result_callback({"error": str(exc)})
            return

        logger.info("transition %s -> %s", label, self.walker.node.id)
        # Replacing the system message is the handoff: the model only ever sees
        # the stage it is actually in.
        self.llm_context.set_messages(
            [{"role": "system", "content": self.walker.instructions()}]
            + [m for m in self.llm_context.get_messages() if m.get("role") != "system"]
        )
        await params.result_callback({"moved_to": self.walker.node.id})

        if self.walker.finished:
            self.finished.set()

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

        await finalise_call(
            voice_call_id=self.voice_call_id,
            recovery_case_id=self.recovery_case_id,
            result=CallResult(
                intent=self.walker.intent or CallIntent.UNCLEAR,
                status=CallStatus.COMPLETED,
                final_node_id=self.walker.node.id,
                transcript=self._transcript(),
                duration_seconds=await duration_since(self.started_at),
                answered_at=self.started_at,
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
                 language: str, company: str) -> dict:
    context = {
        "company_name": company,
        "customer_name": customer_name,
        "amount_spoken": spoken_amount(amount_paise, language),
        "failure_reason": failure_reason,
        "language_hint": language_hint(language),
    }
    context["_language"] = language
    return context


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
        failure_reason=body.get("failure_reason", "बैंक ने पेमेंट अस्वीकार कर दिया"),
        language=language,
        company=body.get("company_name", settings.company_name),
    )
    # Pre-rendered amounts win over our own: the dispatcher may have spoken
    # forms we cannot reconstruct from paise alone.
    if body.get("amount_spoken"):
        context["amount_spoken"] = body["amount_spoken"]
    context["_recovery_case_id"] = body.get("recovery_case_id")
    context["_phone"] = body.get("phone")
    return context


def recovery_mcp() -> MCPClient:
    """Our own MCP server, over stdio.

    Runs as a child process of the agent, so it shares the environment -- the
    same DATABASE_URL and Razorpay keys -- without those ever crossing a
    network. `tools_filter` is explicit: the LLM gets exactly these two and
    nothing a future tool might add by accident.
    """
    return MCPClient(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp_server"],
            env=dict(os.environ),
        ),
        tools_filter=["send_payment_link", "get_case"],
    )


async def run_call(transport, session: DunningSession) -> DunningSession:
    """Wire and run one conversation. Returns the session for its outcome."""
    stt, llm, tts = build_services(session.language)

    # `transition` stays in-process: it is graph traversal, not an external
    # capability, and it must not be able to fail over a transport.
    llm.register_function("transition", session.on_transition)

    aggregators = LLMContextAggregatorPair(
        session.llm_context,
        user_params=LLMUserAggregatorParams(vad_analyzer=build_vad()),
    )

    # The money tools come from our MCP server, so any MCP-speaking agent gets
    # the same guarantees rather than a second implementation of them.
    async with recovery_mcp() as mcp:
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

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        await task.queue_frame(EndFrame())

    async def _end_when_finished() -> None:
        await session.finished.wait()
        # Write the outcome while the closing line is still playing. It costs
        # the customer nothing, and a pipeline that dies during teardown would
        # otherwise lose the one thing the call was for.
        await session.finalise()
        await asyncio.sleep(2.0)
        await task.queue_frame(EndFrame())

    watcher = asyncio.create_task(_end_when_finished())
    try:
        await PipelineRunner().run(task)
    finally:
        watcher.cancel()
    return session


#: Only the transports we can actually exercise. WebRTC is the browser demo;
#: Plivo and Exotel are the reason Pipecat is here at all -- both are plain
#: websocket transports whose serializer `create_transport` fills in, so adding
#: them costs a line each rather than a SIP trunk negotiation.
#:
#: No VAD here: since 1.0 it lives on the user aggregator (see `build_vad`).
TRANSPORT_PARAMS = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "plivo": lambda: _telephony_params(),
    "exotel": lambda: _telephony_params(),
}


def _telephony_params():
    """Imported lazily: pipecat[websocket] is not needed for the WebRTC demo."""
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

    return FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True)


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point. The Pipecat runner discovers this by name.

        uv run --group pipecat python -m app.voice.pipecat_agent -t webrtc

    The body carries the case, exactly as job metadata does on LiveKit.
    """
    body = runner_args.body or SAMPLE_BODY
    if not runner_args.body:
        logger.warning("no runner body; using the sample call context")

    # Raises before any audio flows if the context is incomplete, rather than
    # reading a literal "{customer_name}" down the line.
    session = DunningSession(context_from_body(body))
    session.voice_call_id = await open_call_record(
        recovery_case_id=session.recovery_case_id,
        room_name=body.get("room_name", "pipecat"),
        dialled_number=body.get("phone"),
    )

    transport = await create_transport(runner_args, TRANSPORT_PARAMS)
    try:
        await run_call(transport, session)
    finally:
        # A customer who hangs up mid-sentence never reaches a terminal node.
        # finalise() is idempotent, so this only fires when the graph did not.
        await session.finalise()


__all__ = ["DunningSession", "SpokenFormFilter", "bot", "build_services", "build_vad",
           "call_context", "context_from_body", "normalize_for_speech", "run_call"]


if __name__ == "__main__":
    from pipecat.runner.run import main

    logging.basicConfig(level=logging.INFO)
    main()
