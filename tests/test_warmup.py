"""Sarvam warm-up. The behaviour under failure matters more than the happy path:
a cold model is a degraded call, but a raised exception is no call at all."""

import asyncio
import wave
from io import BytesIO

import pytest

from app.voice import warmup
from app.voice.warmup import PROBE_WAV, await_warmup, warm_sarvam


class FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


class FakeClient:
    """Stands in for httpx.AsyncClient."""

    def __init__(self, *, status: int = 200, raises: Exception | None = None, delay: float = 0):
        self.status = status
        self.raises = raises
        self.delay = delay
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises:
            raise self.raises
        return FakeResponse(self.status)


def install(monkeypatch, client: FakeClient) -> FakeClient:
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    monkeypatch.setattr(warmup.httpx, "AsyncClient", lambda **_: client)
    return client


# --- the probe payload -------------------------------------------------


def test_probe_is_a_valid_wav():
    """Silence can be short-circuited by an endpoint; we want the model loaded."""
    with wave.open(BytesIO(PROBE_WAV), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16_000
        assert handle.getnframes() > 0


def test_probe_is_small():
    """It runs on every call; it must not be a meaningful upload."""
    assert len(PROBE_WAV) < 50_000


# --- warm_sarvam -------------------------------------------------------


async def test_warms_with_the_model_the_session_will_use(monkeypatch):
    """Warming a different model would prime the wrong thing."""
    client = install(monkeypatch, FakeClient())
    assert await warm_sarvam() is True

    sent = client.calls[0]
    assert sent["url"] == warmup.SARVAM_STT_URL
    assert sent["data"]["model"] == "saaras:v4"
    assert sent["data"]["mode"] == "transcribe"
    assert sent["headers"]["api-subscription-key"] == "test-key"


async def test_skips_cleanly_when_no_key_is_configured(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    assert await warm_sarvam() is False


async def test_network_failure_is_swallowed(monkeypatch):
    """A cold model is a degraded call; an exception here is no call at all."""
    install(monkeypatch, FakeClient(raises=OSError("connection reset")))
    assert await warm_sarvam() is False


async def test_http_error_is_swallowed(monkeypatch):
    install(monkeypatch, FakeClient(status=503))
    assert await warm_sarvam() is False


@pytest.mark.parametrize("status", [400, 401, 429, 500])
async def test_no_status_code_raises(monkeypatch, status):
    install(monkeypatch, FakeClient(status=status))
    assert await warm_sarvam() is False


# --- await_warmup ------------------------------------------------------


async def test_await_warmup_returns_once_the_task_finishes():
    async def quick():
        return True

    await asyncio.wait_for(await_warmup(asyncio.create_task(quick())), timeout=1)


async def test_await_warmup_gives_up_rather_than_blocking():
    """A hung warm-up must not hold the pipeline open."""

    async def hang():
        await asyncio.sleep(30)

    task = asyncio.create_task(hang())
    started = asyncio.get_running_loop().time()
    await await_warmup(task, timeout=0.1)
    assert asyncio.get_running_loop().time() - started < 5
    task.cancel()


async def test_await_warmup_swallows_a_failed_task():
    async def boom():
        raise RuntimeError("sarvam exploded")

    await await_warmup(asyncio.create_task(boom()), timeout=1)


async def test_await_warmup_tolerates_no_task():
    await await_warmup(None)
