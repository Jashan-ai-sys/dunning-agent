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

from dotenv import load_dotenv
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.sarvam.stt import SarvamSTTService

from app.config import get_settings
from app.voice.flow import DUNNING_FLOW, language_hint
from app.voice.normalizers import (
    normalize_amounts_for_tts,
    normalize_for_hindi_tts,
    normalize_times_for_tts,
    strip_markdown,
)
from app.voice.spoken import spoken_amount
from app.voice.walker import GraphWalker, InvalidTransition

logger = logging.getLogger(__name__)

load_dotenv()


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
    )
    return stt, llm, tts


def build_vad() -> SileroVADAnalyzer:
    """Blostem's production telephony values, mapped onto Pipecat's own names."""
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


async def run_call(transport, context: dict) -> DunningSession:
    """Wire and run one conversation. Returns the session for its outcome."""
    session = DunningSession(context)
    stt, llm, tts = build_services(session.language)

    llm.register_function("transition", session.on_transition)

    aggregators = LLMContextAggregatorPair(session.llm_context)
    session.llm_context.set_tools(session.tools())

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
        # Let the closing line play before tearing the pipeline down.
        await asyncio.sleep(2.0)
        await task.queue_frame(EndFrame())

    watcher = asyncio.create_task(_end_when_finished())
    try:
        await PipelineRunner().run(task)
    finally:
        watcher.cancel()
    return session


__all__ = ["DunningSession", "build_services", "build_vad", "call_context",
           "normalize_for_speech", "run_call"]
