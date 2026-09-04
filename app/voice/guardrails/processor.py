"""The rail, placed in the pipeline between the model and the voice.

Where this sits is the whole design. Downstream of `llm` and upstream of `tts`
is the last point at which a sentence can still be stopped: after `tts` it is
audio, and after `transport.output()` the customer has heard it.

The cost, stated plainly
========================
A rail can only judge a finished sentence, so `block` mode holds the model's
tokens until the turn ends instead of streaming them into TTS as they arrive.
That trades away progressive synthesis: time to first byte stops being 0.058s
and becomes "however long the model took to finish". On a 0.512s p50 turn that
is the single most expensive thing in this file, and it is why `audit` exists.

    off     nothing is intercepted; identical to not installing this at all
    audit   frames stream through untouched, the turn is vetted afterwards and
            violations are logged. Zero added latency, catches nothing.
    block   the turn is held, vetted, and withheld if it violates. Safe, slow.

`audit` is the honest default for a live phone call and `block` is the honest
default for a demo of the guardrail itself. Neither is correct in the abstract,
so the mode is configuration rather than a decision made here.

Failing open
============
Every exception path lets the sentence through. This follows the same rule as
`voice/persistence.py`: losing a guarantee is bad, dropping the customer
mid-sentence because a rail raised is worse. A rail that crashes the call it
was protecting has done more damage than the sentence it was inspecting.
"""

import asyncio
import logging
import time

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.voice.guardrails.rails import DunningRails

logger = logging.getLogger(__name__)

#: A rail that has not answered by now is not going to be useful. The turn is
#: released unvetted rather than leaving the customer in silence -- see
#: "Failing open" above. Generous relative to the ~1ms the checks actually take,
#: because the budget exists for a pathological case, not the normal one.
VET_TIMEOUT_S = 0.75


class GuardrailProcessor(FrameProcessor):
    """Vets each assistant turn before it is spoken."""

    def __init__(self, guard: DunningRails, *, mode: str = "audit") -> None:
        super().__init__()
        self._guard = guard
        self._mode = mode
        self._buffer: list[LLMTextFrame] = []
        self._holding = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if self._mode == "off" or direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = []
            self._holding = self._mode == "block"
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            self._buffer.append(frame)
            if not self._holding:
                await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            await self._finish_turn(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _finish_turn(self, end_frame: Frame, direction: FrameDirection) -> None:
        spoken = "".join(f.text for f in self._buffer)
        buffered, self._buffer, holding = self._buffer, [], self._holding
        self._holding = False

        if not spoken.strip():
            await self.push_frame(end_frame, direction)
            return

        started = time.perf_counter()
        try:
            vetted, violation = await asyncio.wait_for(
                self._guard.vet(spoken), timeout=VET_TIMEOUT_S
            )
        except TimeoutError:
            logger.warning("guardrail timed out after %.0fms; releasing the turn unvetted",
                           VET_TIMEOUT_S * 1000)
            await self._release(buffered, holding, direction, end_frame)
            return
        except Exception:
            logger.exception("guardrail raised; releasing the turn unvetted")
            await self._release(buffered, holding, direction, end_frame)
            return

        elapsed_ms = (time.perf_counter() - started) * 1000

        if violation is None:
            logger.debug("guardrail passed in %.1fms", elapsed_ms)
            await self._release(buffered, holding, direction, end_frame)
            return

        if not holding:
            # audit mode: already spoken. Recording it is the whole point --
            # a violation nobody counted is indistinguishable from none.
            logger.warning(
                "guardrail violation (audit, already spoken) %s in %.1fms", violation, elapsed_ms
            )
            await self.push_frame(end_frame, direction)
            return

        logger.warning("guardrail withheld a turn: %s (%.1fms)", violation, elapsed_ms)
        await self.push_frame(LLMTextFrame(vetted), direction)
        await self.push_frame(end_frame, direction)

    async def _release(
        self,
        buffered: list[LLMTextFrame],
        holding: bool,
        direction: FrameDirection,
        end_frame: Frame,
    ) -> None:
        if holding:
            for held in buffered:
                await self.push_frame(held, direction)
        await self.push_frame(end_frame, direction)
