"""The NeMo Guardrails runtime, built once and reused for the life of a call.

Loading a `RailsConfig` and constructing `LLMRails` is not free, and doing it
per turn would put config parsing on the critical path of a phone call. One
instance is built when the guard is created and every turn runs through it.

Only output rails are enabled. `dialog: false` in `config.yml` is what keeps
NeMo out of the conversation: the graph in `flow.py` decides what happens next
and refuses illegal moves, and nothing here is allowed to disagree with it.
"""

import logging
from pathlib import Path
from typing import Any

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions

from app.voice.guardrails.checks import Violation, check_output

logger = logging.getLogger(__name__)

CONFIG_PATH = str((Path(__file__).parent / "config").resolve())

#: Output only. Input rails would inspect the customer's own words, which is
#: both useless here -- we do not act on their text, we act on a graph label --
#: and a privacy cost we have no reason to pay.
_ONLY_OUTPUT = GenerationOptions(
    rails={"input": False, "dialog": False, "retrieval": False, "output": True}
)


class DunningRails:
    """One call's guard. Not thread-safe, and does not need to be: a call is a
    single pipeline and its turns are sequential by construction.
    """

    def __init__(self, *, expected_amount_rupees: str | None = None) -> None:
        self._expected = expected_amount_rupees
        self._last: Violation | None = None

        config = RailsConfig.from_path(CONFIG_PATH)
        self._rails = LLMRails(config)
        self._rails.register_action(self._check, name="check_dunning_output")

    async def _check(self, text: str = "") -> bool:
        """The action the Colang rail calls. Truthy means withhold.

        The violation is stashed rather than returned, because Colang only
        needs the boolean -- but the caller wants to know *which* rule fired,
        and a log line saying "withheld" without saying why is not worth
        writing.
        """
        self._last = check_output(text, expected_amount_rupees=self._expected)
        return self._last is not None

    @property
    def last_violation(self) -> Violation | None:
        return self._last

    async def vet(self, text: str) -> tuple[str, Violation | None]:
        """Return the sentence that may be spoken, and what was caught.

        When the rail fires, the returned text is NeMo's refusal message from
        `rails.co`, not the model's sentence. The caller speaks whatever comes
        back -- it is already safe by the time it is returned.
        """
        self._last = None
        result: Any = await self._rails.generate_async(
            # A user turn is required for the message list to be well formed;
            # its content is never inspected because input rails are off.
            messages=[
                {"role": "user", "content": ""},
                {"role": "assistant", "content": text},
            ],
            options=_ONLY_OUTPUT,
        )

        vetted = _assistant_text(result)
        return (vetted if vetted is not None else text), self._last


def _assistant_text(result: Any) -> str | None:
    """Pull the assistant message out of whatever `generate_async` returned.

    It returns a `GenerationResponse` whose `.response` is a list of message
    dicts, but has returned a bare list in other versions. Both shapes are
    handled rather than pinned, because the failure mode of guessing wrong is
    an unvetted sentence reaching a customer.
    """
    payload = getattr(result, "response", result)

    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        return payload.get("content")
    if isinstance(payload, list):
        for message in reversed(payload):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return message.get("content")
    logger.warning("guardrails returned an unrecognised shape: %s", type(payload))
    return None
