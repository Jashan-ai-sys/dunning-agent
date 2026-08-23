"""Apply the result of a call back onto the recovery case.

This is the seam between the conversation and the money. It runs after the call
ends, whatever transport placed it, so the rules here are testable without a
telephony provider in the loop.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import ActionType, CallStatus
from app.models import Customer, RecoveryCase, VoiceCall
from app.payment_links import create_recovery_link
from app.store import log_action, utcnow
from app.voice.intents import CallIntent, outcome_for

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CallResult:
    """What the agent observed. Transport-agnostic on purpose."""

    intent: CallIntent
    status: CallStatus = CallStatus.COMPLETED
    final_node_id: str | None = None
    transcript: str | None = None
    duration_seconds: int | None = None
    answered_at: datetime | None = None
    error: str | None = None
    #: Labelled graph turns, from GraphWalker.observations_as_dicts().
    transitions: list[dict] | None = None


async def apply_call_result(
    session: AsyncSession,
    case: RecoveryCase,
    call: VoiceCall,
    result: CallResult,
    *,
    customer: Customer | None = None,
    client=None,
) -> None:
    """Record the call and move the case according to the detected intent.

    Intent decides the case's fate; the transcript is evidence, never input.
    Keeping it that way means a persuasive customer cannot talk the system into
    a state the policy does not allow.
    """
    call.status = result.status
    call.detected_intent = result.intent
    call.final_node_id = result.final_node_id
    call.transcript = result.transcript
    call.transitions = result.transitions or None
    call.duration_seconds = result.duration_seconds
    call.answered_at = result.answered_at
    call.error = result.error
    call.ended_at = utcnow()

    outcome = outcome_for(result.intent)

    if outcome.status is not None:
        case.status = outcome.status

    if outcome.suppress_contact:
        # Close the door regardless of attempts remaining: an explicit no, a
        # wrong number and a disputed charge must all end the calling.
        case.attempt_count = case.max_attempts

    await log_action(
        session,
        case,
        ActionType.VOICE_CALL,
        {
            "voice_call_id": call.id,
            "intent": str(result.intent),
            "status": str(result.status),
            "final_node": result.final_node_id,
            "duration_seconds": result.duration_seconds,
            "suppressed_contact": outcome.suppress_contact,
            "needs_human": outcome.needs_human,
            "payment_link_due": outcome.send_payment_link,
        },
    )

    if outcome.status is not None:
        await log_action(
            session,
            case,
            ActionType.STOPPED,
            {"reason": f"call_intent_{result.intent}", "needs_human": outcome.needs_human},
        )

    if outcome.needs_human:
        logger.warning(
            "case %s needs human review after a disputed charge on call %s", case.id, call.id
        )

    if outcome.send_payment_link and customer is not None and client is not None:
        # Failing to send the link must not undo the rest of the call record --
        # the customer still said yes, and the next tick can retry the link.
        try:
            await create_recovery_link(session, case, customer, client)
        except Exception:
            logger.exception("could not create a payment link for case %s", case.id)
