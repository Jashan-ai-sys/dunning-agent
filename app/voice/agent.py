"""LiveKit voice agent that walks the dunning conversation graph.

Run as its own worker process, separate from the API:

    uv run --group voice python -m app.voice.agent dev

Deliberately thin. Every decision about where the conversation goes lives in
``walker.py`` and is unit-tested; this module only carries audio in and out and
translates one tool call into one graph transition. The blast radius of code
that cannot be tested without a telephony provider is kept as small as possible.

Stack: Sarvam STT (Indian languages, streaming), Gemini 2.5 Flash, Cartesia TTS.
"""

import asyncio
import json
import logging

from dotenv import load_dotenv
from livekit import agents, api
from livekit.agents import Agent, AgentServer, AgentSession, RunContext, function_tool
from livekit.plugins import cartesia, google, openai, sarvam, silero

from app.config import get_settings
from app.voice.flow import DUNNING_FLOW, language_hint
from app.voice.spoken import spoken_amount
from app.voice.walker import GraphWalker, InvalidTransition
from app.voice.warmup import await_warmup, warm_sarvam

logger = logging.getLogger(__name__)

# The LiveKit worker reads LIVEKIT_URL/API_KEY/API_SECRET straight from
# os.environ, and each provider plugin reads its own key the same way -- none of
# them know about our Settings object. Loading .env into the process environment
# is what makes `python -m app.voice.agent` work from a checkout.
load_dotenv()

server = AgentServer()

#: Used only when a job arrives with no metadata at all, so the agent can be
#: exercised from the LiveKit console. Never used on a dispatched recovery call.
SAMPLE_CONTEXT = {
    "customer_name": "Asha",
    "amount_spoken": "499 रुपये",
    "failure_reason": "your card had insufficient funds",
    "preferred_language": "hi",
}


class CallState:
    """Mutable result of one call, read after the session ends."""

    def __init__(self, walker: GraphWalker) -> None:
        self.walker = walker
        self.transcript: list[str] = []

    @property
    def intent(self):
        return self.walker.intent

    @property
    def final_node_id(self) -> str:
        return self.walker.node.id

    def as_transcript(self) -> str:
        return "\n".join(self.transcript)


class NodeAgent(Agent):
    """One stage of the conversation.

    Reaching a new node hands off to a new instance, so the model is only ever
    holding the instructions and branches for where it currently is.
    """

    def __init__(self, state: CallState) -> None:
        super().__init__(instructions=state.walker.instructions())
        self._state = state

    async def on_enter(self) -> None:
        await self.session.generate_reply()
        if self._state.walker.finished:
            logger.info(
                "call finished at %s with intent %s",
                self._state.final_node_id,
                self._state.intent,
            )
            # Let the closing line play out before tearing the room down.
            await self.session.drain()
            await self.session.aclose()

    @staticmethod
    def _last_user_utterance(context: RunContext) -> str | None:
        """The customer's most recent words, for the training record only.

        Never used to decide anything -- the label alone drives traversal. Read
        defensively: this is a convenience for offline prompt optimisation and
        must not be able to fail a live call.
        """
        try:
            items = getattr(context.session.history, "items", None) or []
            for item in reversed(items):
                if getattr(item, "role", None) == "user":
                    content = getattr(item, "content", None)
                    if isinstance(content, list):
                        content = " ".join(str(c) for c in content)
                    return str(content).strip() or None
        except Exception:  # noqa: BLE001 - diagnostics must never break a call
            logger.debug("could not read the last user utterance", exc_info=True)
        return None

    @function_tool
    async def transition(self, context: RunContext, label: str) -> str:
        """Move the conversation to the next stage.

        Call this only when the customer's position is clear. ``label`` must be
        one of the labels listed in your instructions.
        """
        try:
            self._state.walker.transition(label, utterance=self._last_user_utterance(context))
        except InvalidTransition as exc:
            # Hand the error back to the model rather than failing the call --
            # it can pick a valid label or ask a clarifying question.
            logger.warning("rejected transition %r: %s", label, exc)
            return str(exc)

        logger.info("transition %s -> %s", label, self._state.walker.node.id)
        return NodeAgent(self._state)


def build_llm():
    """Gemini via Vertex AI by default.

    Vertex authenticates with ADC or a service account rather than an API key,
    which keeps the LLM inside the same GCP project and IAM boundary as the rest
    of the deployment. Set GOOGLE_USE_VERTEX=false to fall back to the Gemini
    Developer API and GOOGLE_API_KEY.
    """
    settings = get_settings()

    if settings.llm_provider == "local":
        # SGLang and vLLM both speak the OpenAI protocol, so the same client
        # covers either. Speculative decoding (DSpark, n-gram, EAGLE) is a
        # server-side setting and is invisible from here.
        return openai.LLM(
            model=settings.local_llm_model,
            base_url=settings.local_llm_base_url,
            api_key=settings.local_llm_api_key,
            temperature=settings.llm_temperature,
        )

    if not settings.google_use_vertex:
        return google.LLM(
            model=settings.gemini_model, temperature=settings.llm_temperature
        )

    kwargs: dict = {
        "vertexai": True,
        "location": settings.google_cloud_location,
        "temperature": settings.llm_temperature,
    }
    if settings.google_cloud_project:
        kwargs["project"] = settings.google_cloud_project
    return google.LLM(model=settings.gemini_model, **kwargs)


def build_session() -> AgentSession:
    settings = get_settings()
    tts_kwargs = {"voice": settings.cartesia_voice} if settings.cartesia_voice else {}
    return AgentSession(
        stt=sarvam.STT(
            model=settings.sarvam_stt_model,
            language=settings.sarvam_language,
            mode="transcribe",
            flush_signal=True,
        ),
        llm=build_llm(),
        tts=cartesia.TTS(**tts_kwargs),
        vad=silero.VAD.load(
            activation_threshold=settings.vad_activation_threshold,
            min_silence_duration=settings.vad_min_silence_duration,
            min_speech_duration=settings.vad_min_speech_duration,
        ),
        turn_detection="stt",
    )


@server.rtc_session(agent_name=get_settings().livekit_agent_name)
async def dunning_session(ctx: agents.JobContext) -> None:
    """Entry point. Job metadata carries the case context and the number."""
    settings = get_settings()

    # Fire immediately and overlap with everything below. saaras has documented
    # 15-25s cold starts that swallow the opening exchange.
    warmup = asyncio.create_task(warm_sarvam())

    metadata = json.loads(ctx.job.metadata or "{}")

    if not metadata:
        # Dispatched without context -- the LiveKit console does this. Fall back
        # to a clearly-labelled sample so the agent is testable from the
        # dashboard, rather than offering to recover "Rs 0".
        logger.warning("no job metadata; using the sample call context")
        metadata = SAMPLE_CONTEXT

    call_context = {
        "company_name": metadata.get("company_name", settings.company_name),
        "customer_name": metadata.get("customer_name", "there"),
        # Pre-rendered rather than left to the model: a raw numeral plus a
        # Latin "Rs" is the script flip that makes Hindi voices stumble.
        "amount_spoken": metadata.get("amount_spoken")
        or spoken_amount(
            int(metadata.get("amount_paise", 0)), metadata.get("preferred_language")
        ),
        "failure_reason": metadata.get("failure_reason", "the bank declined it"),
        "language_hint": language_hint(metadata.get("preferred_language")),
    }
    # Raises before the phone rings if the context is incomplete, rather than
    # reading a literal "{customer_name}" down the line.
    state = CallState(GraphWalker(DUNNING_FLOW, call_context))

    session = build_session()
    # Latest useful moment: the pipeline is built, nobody has spoken yet.
    await await_warmup(warmup)
    await session.start(room=ctx.room, agent=NodeAgent(state))

    phone = metadata.get("phone")
    if phone:
        await _dial(ctx, phone, settings.livekit_sip_trunk_id)


async def _dial(ctx: agents.JobContext, phone: str, trunk_id: str) -> None:
    """Place the outbound leg. The agent is already in the room when it rings."""
    if not trunk_id:
        raise RuntimeError(
            "LIVEKIT_SIP_TRUNK_ID is not set; an outbound trunk must exist to dial PSTN"
        )
    await ctx.api.sip.create_sip_participant(
        api.CreateSIPParticipantRequest(
            room_name=ctx.room.name,
            sip_trunk_id=trunk_id,
            sip_call_to=phone,
            participant_identity="customer",
            wait_until_answered=True,
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
