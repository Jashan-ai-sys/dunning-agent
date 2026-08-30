"""The self-hosted speech and LLM path, at the seams that do not need a GPU.

This code was written and deployed in one night against models that had never
run, and it reached a live call before it had a single test. The call was
silent. What follows is the coverage that should have existed first.

Nothing here talks to Modal. What it pins is the wiring: that the switch
selects the right classes, that the vendor path is untouched when the switch is
off, and that the two shapes of failure a self-hosted service actually has --
a non-200, and an unreachable host -- surface as an ErrorFrame rather than an
exception that ends somebody's call.
"""

import pytest
from pipecat.frames.frames import ErrorFrame, TranscriptionFrame

from app.config import Settings
from app.voice.local_services import LocalSTTService, LocalTTSService, pcm_to_wav

SPEECH_URL = "https://speech.example.invalid"


# --- the WAV header SraVaani needs -----------------------------------------


def test_raw_pcm_gets_a_header_the_model_can_read():
    """SraVaani reads a file and infers the rate from its header. Handed
    headerless PCM it guesses, and it guesses wrong."""
    wav = pcm_to_wav(b"\x00\x01" * 8000, 16_000)

    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    # 16-bit mono at the rate we passed, not at some default.
    assert int.from_bytes(wav[24:28], "little") == 16_000
    assert int.from_bytes(wav[22:24], "little") == 1


def test_the_declared_rate_follows_the_audio():
    """The pipeline runs STT at 16kHz and the phone line at 8. A header that
    always said one of them would mis-describe half the calls."""
    assert int.from_bytes(pcm_to_wav(b"\x00\x01" * 100, 8_000)[24:28], "little") == 8_000


# --- the switch ------------------------------------------------------------


def _settings(**overrides) -> Settings:
    return Settings(**overrides)


def test_the_vendor_path_is_the_default():
    """Unset, none of this runs. The local stack is a deployment choice, and a
    half-configured environment must not quietly change what a customer hears.
    """
    settings = _settings()

    assert settings.local_speech_url == ""
    assert settings.llm_provider == "vertex"


def test_the_local_llm_switch_is_the_one_that_already_existed():
    """There is exactly one way to ask for a local LLM.

    A second setting meaning the same thing shadowed `local_llm_model` and
    pointed the LiveKit agent at a model name its own base URL did not serve --
    silently, because Pydantic keeps the last declaration.
    """
    settings = _settings(llm_provider="local")

    assert settings.llm_provider == "local"
    assert settings.local_llm_base_url
    assert settings.local_llm_model
    # The duplicate is gone: one declaration, one meaning.
    assert not hasattr(settings, "local_llm_url")


def test_selecting_the_local_speech_stack_builds_the_local_classes():
    from app.voice.pipecat_agent import build_local_services

    stt, _llm, tts = build_local_services("hi")

    assert isinstance(stt, LocalSTTService)
    assert isinstance(tts, LocalTTSService)


def test_the_spoken_form_filter_survives_the_swap():
    """Those rewrites turn digits and times into what a Hindi speaker says.
    They belong to the language, not to Cartesia -- VoxCPM needs them just as
    much, and losing them would have the agent read '499' as digits."""
    from app.voice.pipecat_agent import build_local_services

    _stt, _llm, tts = build_local_services("hi")

    assert tts._text_filters, "the local TTS lost the spoken-form filter"


# --- failure, which is the whole point -------------------------------------


class _Response:
    def __init__(self, status: int, payload=None, text: str = "") -> None:
        self.status = status
        self._payload = payload
        self._text = text

    async def json(self):
        return self._payload

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class _Session:
    """Stands in for aiohttp: one canned response, or an exception."""

    def __init__(self, response=None, raises: Exception | None = None) -> None:
        self._response = response
        self._raises = raises

    def post(self, *_args, **_kwargs):
        if self._raises is not None:
            raise self._raises
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


async def _frames(service, *, session, method, **kwargs):
    import app.voice.local_services as module

    original = module.aiohttp.ClientSession
    module.aiohttp.ClientSession = lambda **_: session
    try:
        return [frame async for frame in method(**kwargs)]
    finally:
        module.aiohttp.ClientSession = original


@pytest.mark.parametrize(
    "session",
    [
        _Session(response=_Response(503, text="model still loading")),
        _Session(raises=TimeoutError("cold start took longer than we waited")),
    ],
    ids=["non-200", "unreachable"],
)
async def test_a_failing_transcription_yields_an_error_not_an_exception(session):
    """A turn we could not hear must not end the call. The customer is still
    on the line and the next tick can try again."""
    stt = LocalSTTService(base_url=SPEECH_URL, sample_rate=16_000)

    frames = await _frames(stt, session=session, method=stt.run_stt, audio=b"\x00\x01" * 800)

    assert frames and all(isinstance(f, ErrorFrame) for f in frames)
    assert not any(isinstance(f, TranscriptionFrame) for f in frames)


async def test_silence_produces_no_turn_at_all():
    """An empty transcript is not a turn the customer took. Emitting one would
    start a reply to something nobody said."""
    stt = LocalSTTService(base_url=SPEECH_URL, sample_rate=16_000)
    session = _Session(response=_Response(200, payload={"text": "   "}))

    frames = await _frames(stt, session=session, method=stt.run_stt, audio=b"\x00\x01" * 800)

    assert frames == []


async def test_a_heard_utterance_becomes_a_transcription_frame():
    stt = LocalSTTService(base_url=SPEECH_URL, sample_rate=16_000)
    session = _Session(response=_Response(200, payload={"text": "हाँ जी, बताइए"}))

    frames = await _frames(stt, session=session, method=stt.run_stt, audio=b"\x00\x01" * 800)

    assert [f.text for f in frames if isinstance(f, TranscriptionFrame)] == ["हाँ जी, बताइए"]


@pytest.mark.parametrize(
    "session",
    [
        _Session(response=_Response(500, text="out of memory")),
        _Session(raises=TimeoutError("the GPU was still starting")),
    ],
    ids=["non-200", "unreachable"],
)
async def test_a_failing_synthesis_yields_an_error_not_an_exception(session):
    """This is the one that was learned the hard way.

    On a live call the speech container had scaled down, the request timed out,
    and the agent said nothing at all. An ErrorFrame is what reaches the mute
    guard, which is what stops a silent call being recorded as a successful
    one -- so raising here would lose the outcome as well as the audio.
    """
    tts = LocalTTSService(base_url=SPEECH_URL)

    frames = await _frames(
        tts, session=session, method=tts.run_tts, text="नमस्ते", context_id="ctx"
    )

    assert any(isinstance(f, ErrorFrame) for f in frames)


def test_the_source_rate_matches_what_voxcpm_actually_emits():
    """48kHz in, 8kHz on the line. Get this wrong and every reply is pitched
    wrong -- audible immediately, and not obviously a rate bug when you hear it.
    """
    from app.voice.local_services import VOXCPM_SAMPLE_RATE

    assert VOXCPM_SAMPLE_RATE == 48_000
