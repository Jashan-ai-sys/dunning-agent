"""Call outcomes and what each one does to the recovery case.

The graph's terminal nodes resolve to exactly one of these, which is the only
thing the conversation is allowed to hand back to the orchestrator. Keeping the
mapping here -- rather than inside the agent -- means the consequences of a call
are reviewable without reading any prompt.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.constants import CaseStatus


class CallIntent(StrEnum):
    RETRY_NOW = "retry_now"
    RETRY_LATER = "retry_later"
    DECLINED = "declined"
    WRONG_NUMBER = "wrong_number"
    DISPUTE = "dispute"
    NO_ANSWER = "no_answer"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class IntentOutcome:
    """What the orchestrator does with a case after the call."""

    #: New case status, or None to leave it as-is for another attempt.
    status: CaseStatus | None
    #: Send a Razorpay payment link (Phase 4).
    send_payment_link: bool = False
    #: Never contact this number again, regardless of attempts remaining.
    suppress_contact: bool = False
    #: Needs a human, not another automated attempt.
    needs_human: bool = False


OUTCOMES: dict[CallIntent, IntentOutcome] = {
    # Wants to pay now: keep the case open until the payment actually lands.
    CallIntent.RETRY_NOW: IntentOutcome(status=None, send_payment_link=True),
    # Asked us to try later: the backoff window handles the timing.
    CallIntent.RETRY_LATER: IntentOutcome(status=None),
    # An explicit no is a stopping rule -- we do not keep calling.
    CallIntent.DECLINED: IntentOutcome(status=CaseStatus.DECLINED, suppress_contact=True),
    # Reached the wrong person: stop immediately. Continuing would be both a
    # compliance problem and a privacy one.
    CallIntent.WRONG_NUMBER: IntentOutcome(
        status=CaseStatus.STOPPED, suppress_contact=True
    ),
    # Disputes the charge: no bot should argue about money.
    CallIntent.DISPUTE: IntentOutcome(
        status=CaseStatus.STOPPED, suppress_contact=True, needs_human=True
    ),
    # Nobody picked up, or the intent never became clear: leave the case open
    # and let the attempt cap bound it.
    CallIntent.NO_ANSWER: IntentOutcome(status=None),
    CallIntent.UNCLEAR: IntentOutcome(status=None),
}


def outcome_for(intent: CallIntent) -> IntentOutcome:
    return OUTCOMES[intent]


#: Intents that must never lead to another automated call.
TERMINAL_INTENTS = frozenset(
    intent for intent, outcome in OUTCOMES.items() if outcome.suppress_contact
)
