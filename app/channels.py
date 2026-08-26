"""Outbound contact channels.

The orchestrator depends on the ``ContactChannel`` protocol, never on a concrete
implementation, so Phase 3 can drop in a LiveKit-backed channel without touching
the policy or the orchestration loop.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config import get_settings
from app.models import Customer, RecoveryCase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContactResult:
    """Outcome of *initiating* contact -- not of the conversation itself.

    The conversation outcome (intent, transcript) arrives asynchronously when
    the call ends; that is Phase 3.
    """

    channel: str
    reference: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        return {"channel": self.channel, "reference": self.reference, **self.detail}


class ContactChannel(Protocol):
    async def initiate(self, case: RecoveryCase, customer: Customer) -> ContactResult: ...


class LoggingChannel:
    """Records the intent to contact without placing a call.

    This is a placeholder, not a simulation: it exists so the policy engine and
    orchestrator can be built and tested ahead of the LiveKit integration. It is
    named ``logging`` in the audit trail precisely so nobody can mistake a run of
    this channel for evidence that a customer was actually called.
    """

    name = "logging"

    async def initiate(self, case: RecoveryCase, customer: Customer) -> ContactResult:
        logger.info(
            "would contact customer %s on case %s for %s paise",
            customer.razorpay_customer_id,
            case.id,
            case.original_amount,
        )
        return ContactResult(
            channel=self.name,
            detail={
                "language": customer.preferred_language,
                "amount": case.original_amount,
                "placed": False,
            },
        )


def build_channel() -> ContactChannel:
    """The best channel the current configuration can actually place a call on.

    Ordered by what each one needs. Twilio needs a number and a public
    websocket; LiveKit additionally needs an outbound SIP trunk negotiated with
    a carrier, which is why it sits second despite arriving first. Falling back
    to ``LoggingChannel`` is deliberate: an unconfigured deployment should keep
    working the cases and say plainly that nobody was called, rather than crash
    the worker on startup.

    Imports are local because ``dispatch`` pulls in the LiveKit SDK at module
    level, and a Twilio deployment should not have to install it.
    """
    settings = get_settings()

    if settings.twilio_configured:
        from app.voice.telephony import TwilioChannel

        return TwilioChannel()

    if settings.livekit_configured and settings.livekit_sip_trunk_id:
        from app.voice.dispatch import LiveKitChannel

        return LiveKitChannel()

    logger.warning(
        "no telephony configured; recovery cases will be worked but nobody will be called"
    )
    return LoggingChannel()
