"""Outbound contact channels.

The orchestrator depends on the ``ContactChannel`` protocol, never on a concrete
implementation, so Phase 3 can drop in a LiveKit-backed channel without touching
the policy or the orchestration loop.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

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
