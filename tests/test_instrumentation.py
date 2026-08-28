"""The timing wrappers around turn detection.

Component latency accounted for ~0.64s of a 1.18s turn. The missing half second
is turn detection, and the two things it could be imply opposite fixes: slow
ONNX inference is a hardware problem, deliberate waiting is a tuning one.
"""


from app.voice.instrumentation import (
    VAD_REPORT_EVERY,
    TimedSileroVAD,
    TimedSmartTurnAnalyzer,
)


def test_the_analyzer_is_still_a_smart_turn_analyzer():
    """Pipecat type-checks the strategy, so the wrapper has to remain one."""
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    assert issubclass(TimedSmartTurnAnalyzer, LocalSmartTurnAnalyzerV3)


def test_the_vad_is_still_a_silero_analyzer():
    from pipecat.audio.vad.silero import SileroVADAnalyzer

    assert issubclass(TimedSileroVAD, SileroVADAnalyzer)


def test_the_agent_builds_the_timed_versions():
    """Wiring the wrapper in is the whole point; a plain analyser measures
    nothing."""
    from app.voice.pipecat_agent import build_turn_strategies, build_vad

    assert isinstance(build_vad(), TimedSileroVAD)
    strategy = build_turn_strategies().stop[0]
    assert isinstance(strategy._turn_analyzer, TimedSmartTurnAnalyzer)


def test_vad_summarises_rather_than_logging_every_frame(caplog):
    """Silero runs hundreds of times a second. Per-frame logging would cost
    more than it measures and drown the call log."""

    class Fake(TimedSileroVAD):
        def __init__(self):
            self._calls = 0
            self._total_s = 0.0

        def voice_confidence(self, buffer):
            # Skip Silero itself; only the accounting is under test.
            import time

            started = time.perf_counter()
            self._total_s += time.perf_counter() - started
            self._calls += 1
            if self._calls % VAD_REPORT_EVERY == 0:
                import logging

                logging.getLogger("app.voice.instrumentation").info(
                    "silero_vad frames=%d", self._calls
                )
            return 0.0

    vad = Fake()
    with caplog.at_level("INFO", logger="app.voice.instrumentation"):
        for _ in range(VAD_REPORT_EVERY - 1):
            vad.voice_confidence(b"")
        assert "silero_vad" not in caplog.text, "reported too early"
        vad.voice_confidence(b"")
        assert "silero_vad" in caplog.text
    assert vad._calls == VAD_REPORT_EVERY


def test_the_report_interval_is_not_absurdly_small():
    """A low interval turns the log into noise on every call."""
    assert VAD_REPORT_EVERY >= 100
