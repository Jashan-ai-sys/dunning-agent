"""Speech services that talk to our own GPU instead of a vendor's API.

Drop-in replacements for `SarvamSTTService` and `CartesiaTTSService`, pointed
at the Modal app in `scripts/modal_speech.py`. Everything above them -- the
graph, the walker, the policy, the outcome -- is unchanged and does not know
which pair is running.

The reason to have them at all: on a payments call the audio *is* the customer
data. "The recording never leaves your infrastructure" is a procurement answer
before it is an engineering one, and RBI localisation rules make it a real
requirement rather than a talking point. It is also insurance -- a vendor quota
ran out mid-call once already, and the agent talked to nobody for two minutes
while every log said the call was fine.

What this costs, measured rather than guessed, on an L4 with a request from
India to a US datacentre:

    STT  5.5s for one sentence
    TTS  7.7s for one sentence, generated whole

Those are not usable numbers for a phone call, which is why the TTS path here
streams. VoxCPM generates progressively, so the customer waits for the first
chunk rather than the last one; the rest arrives while they are listening to
the beginning. STT cannot do the same trick -- SraVaani transcribes a complete
utterance, so its latency is a floor until the model is swapped or served
through something faster than eager PyTorch.
"""

import io
import logging
import wave
from collections.abc import AsyncGenerator

import aiohttp
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.utils.time import time_now_iso8601

logger = logging.getLogger(__name__)

#: VoxCPM's native rate. The pipeline runs at 8kHz for a phone line, and
#: `_stream_audio_frames_from_iterator` resamples on the way through -- so this
#: has to be right or every reply is played at the wrong pitch.
VOXCPM_SAMPLE_RATE = 48_000

#: Large enough that we are not making a frame per millisecond, small enough
#: that the first one leaves quickly. The first chunk is the only one whose
#: latency a customer can hear.
CHUNK_BYTES = 8_192

#: Generous, because the thing on the other end may be loading 10GB of model.
#: 120s was not enough: a cold container takes ~264s, and giving up at 120 left
#: the agent silent on a live call while the GPU was still starting.
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=300, sock_connect=30)


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container.

    The standard library rather than a pipecat helper: `pipecat.audio.utils`
    has no `pcm_to_wav` in 1.7, and a header this simple is not worth a
    dependency that can move underneath us.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


class LocalSTTService(SegmentedSTTService):
    """SraVaani, over HTTP.

    `SegmentedSTTService` rather than `STTService` because SraVaani transcribes
    a file, not a stream: the base class buffers on VAD and hands over one
    complete utterance, which is exactly the shape the model wants. It also
    means the turn-taking decision stays where it already was -- Silero and
    SmartTurn upstream -- rather than moving into a model that has no opinion
    about it.
    """

    def __init__(self, *, base_url: str, language: str = "hi", **kwargs) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._language = language

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """One utterance in, one transcript out.

        The audio arrives as raw PCM and leaves as a WAV, because SraVaani's
        `transcribe` reads a file and infers the rate from its header. Sending
        headerless PCM makes it guess, and it guesses wrong.
        """
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        try:
            # `sample_rate` is 0 until the pipeline's StartFrame sets it, and a
            # zero here does not produce quiet audio -- `wave` refuses to write
            # a header at all, so the turn fails as "STT unreachable" and the
            # real cause is nowhere in the message. Fall back to what we were
            # constructed with, which is the rate we asked the pipeline for.
            rate = self.sample_rate or self._init_sample_rate
            wav = pcm_to_wav(audio, rate)
            async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
                async with session.post(
                    f"{self._base_url}/transcribe",
                    data=wav,
                    headers={"Content-Type": "application/octet-stream"},
                ) as response:
                    if response.status != 200:
                        detail = await response.text()
                        yield ErrorFrame(error=f"local STT {response.status}: {detail[:200]}")
                        return
                    payload = await response.json()
        except Exception as exc:  # noqa: BLE001 - a failed turn must not kill the call
            logger.exception("local STT request failed")
            yield ErrorFrame(error=f"local STT unreachable: {exc}")
            return
        finally:
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()

        text = (payload or {}).get("text", "").strip()
        if not text:
            # Silence, or a noise the model declined to interpret. Emitting an
            # empty transcript would start a turn the customer never took.
            return
        logger.debug("local STT heard: %s", text)
        yield TranscriptionFrame(text, "", time_now_iso8601(), None)


class LocalTTSService(TTSService):
    """VoxCPM, streamed.

    Generated whole, one sentence took 7.7s before a single sample reached the
    line. Streamed, the customer waits for the first chunk instead -- the rest
    is produced while they are listening to the start of the sentence. That is
    the difference between a pause and a dead line.
    """

    def __init__(self, *, base_url: str, sample_rate: int | None = None, **kwargs) -> None:
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            **kwargs,
        )
        self._base_url = base_url.rstrip("/")

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Stream 48kHz PCM from the GPU, resampled to the line on the way past."""
        logger.debug("local TTS speaking: %s", text[:80])
        try:
            await self.start_tts_usage_metrics(text)
            async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
                async with session.post(
                    f"{self._base_url}/speak", json={"text": text}
                ) as response:
                    if response.status != 200:
                        detail = await response.text()
                        yield ErrorFrame(error=f"local TTS {response.status}: {detail[:200]}")
                        return
                    async for frame in self._stream_audio_frames_from_iterator(
                        response.content.iter_chunked(CHUNK_BYTES),
                        in_sample_rate=VOXCPM_SAMPLE_RATE,
                        context_id=context_id,
                    ):
                        await self.stop_ttfb_metrics()
                        yield frame
        except Exception as exc:  # noqa: BLE001 - never break a call over one line
            logger.exception("local TTS request failed")
            yield ErrorFrame(error=f"local TTS unreachable: {exc}")
        finally:
            await self.stop_ttfb_metrics()
