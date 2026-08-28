"""Timing for the two pieces of turn-taking nobody was measuring.

Component latency accounted for roughly 0.64s of a 1.18s turn: Vertex 0.51s,
Cartesia 0.08s, Sarvam 0.05s. The missing half second is turn detection, and
until now it was invisible -- which matters because the two things it could be
imply opposite fixes.

If SmartTurn's ONNX inference is slow, the answer is hardware: more CPU, or the
remote analyser. If inference is fast and the time is Silero waiting out
``stop_secs`` before the analyser is even asked, the answer is tuning, and
buying CPU would change nothing.

So both are timed, and the SmartTurn wrapper reports its own wall clock
alongside the model's self-reported inference time. The gap between those two
is queueing in the executor rather than compute.
"""

import logging
import time

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger = logging.getLogger(__name__)

#: Silero runs per audio frame -- hundreds of times a second. Logging each call
#: would drown the call log and cost more than it measures, so it accumulates
#: and reports a summary.
VAD_REPORT_EVERY = 500


class TimedSmartTurnAnalyzer(LocalSmartTurnAnalyzerV3):
    """SmartTurn v3, with the cost of each decision on the record."""

    async def analyze_end_of_turn(self):
        started = time.perf_counter()
        state, metrics = await super().analyze_end_of_turn()
        elapsed_ms = (time.perf_counter() - started) * 1000

        inference_ms = getattr(metrics, "inference_time_ms", None)
        queued_ms = (
            elapsed_ms - inference_ms if isinstance(inference_ms, (int, float)) else None
        )
        logger.info(
            "smart_turn state=%s wall=%.1fms inference=%s queued=%s",
            getattr(state, "name", state),
            elapsed_ms,
            f"{inference_ms:.1f}ms" if inference_ms is not None else "n/a",
            f"{queued_ms:.1f}ms" if queued_ms is not None else "n/a",
        )
        return state, metrics


class TimedSileroVAD(SileroVADAnalyzer):
    """Silero, summarised rather than logged per frame."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._calls = 0
        self._total_s = 0.0

    def voice_confidence(self, buffer) -> float:
        started = time.perf_counter()
        confidence = super().voice_confidence(buffer)
        self._total_s += time.perf_counter() - started
        self._calls += 1

        if self._calls % VAD_REPORT_EVERY == 0:
            logger.info(
                "silero_vad frames=%d mean=%.2fms total=%.0fms",
                self._calls,
                (self._total_s / self._calls) * 1000,
                self._total_s * 1000,
            )
        return confidence
