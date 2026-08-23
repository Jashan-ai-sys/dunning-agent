"""Prime Sarvam before the customer speaks.

The Blostem backend documents 15-25 second cold starts on saaras that "drop the
first hello" -- the model loads while the customer is already talking, and the
opening exchange is simply lost. On a dunning call that is the worst possible
turn to lose: the customer says "hello?", hears nothing, and hangs up.

Their production fix probes Sarvam's *TTS* endpoint, on the reasoning that STT
and TTS share a subscription key and edge. We only use Sarvam for STT, so we
warm the exact model we depend on instead: a short WAV to the speech-to-text
endpoint, with the same model id the live session will open a socket for.

Three properties this must have, in order of importance:

1. **It can never fail a call.** Every error is swallowed. A cold model is a
   degraded call; a raised exception is no call at all.
2. **It is bounded.** A hung warm-up must not hold the pipeline open, so the
   wait has a hard ceiling and gives up rather than blocking.
3. **It runs in parallel.** Fired as a task at entry and awaited just before
   the session starts, so it overlaps the work we were doing anyway.
"""

import asyncio
import io
import logging
import math
import os
import struct
import wave

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

_SAMPLE_RATE = 16_000
_DURATION_SECONDS = 0.25


def _probe_wav() -> bytes:
    """A quarter-second 440 Hz tone as a WAV.

    Real audio rather than silence: some endpoints reject or short-circuit an
    empty payload, and we want the model actually loaded, not just the auth
    path exercised. The transcript is irrelevant and discarded.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(_SAMPLE_RATE)
        frames = int(_SAMPLE_RATE * _DURATION_SECONDS)
        handle.writeframes(
            b"".join(
                struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440 * i / _SAMPLE_RATE)))
                for i in range(frames)
            )
        )
    return buffer.getvalue()


#: Built once at import; the bytes never change.
PROBE_WAV = _probe_wav()


async def warm_sarvam(*, timeout: float = 8.0) -> bool:
    """Send one throwaway transcription request. Returns True if Sarvam answered.

    Never raises.
    """
    api_key = os.environ.get("SARVAM_API_KEY", "")
    if not api_key:
        logger.debug("no SARVAM_API_KEY; skipping warm-up")
        return False

    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": api_key},
                data={"model": settings.sarvam_stt_model, "mode": "transcribe"},
                files={"file": ("warmup.wav", PROBE_WAV, "audio/wav")},
            )
    except Exception as exc:  # noqa: BLE001 - a cold model beats a failed call
        logger.warning("sarvam warm-up failed: %s", exc)
        return False

    if response.status_code >= 400:
        logger.warning("sarvam warm-up returned %s", response.status_code)
        return False

    logger.info("sarvam warmed (%s)", settings.sarvam_stt_model)
    return True


async def await_warmup(task: asyncio.Task | None, *, timeout: float = 6.0) -> None:
    """Wait for a warm-up task, but never longer than ``timeout``.

    The point is to overlap the wait with pipeline construction, not to gate the
    call on it. If Sarvam is slow we proceed cold rather than leave the customer
    listening to nothing.
    """
    if task is None:
        return
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except TimeoutError:
        logger.warning("sarvam warm-up still running after %.1fs; continuing cold", timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("sarvam warm-up task errored: %s", exc)
