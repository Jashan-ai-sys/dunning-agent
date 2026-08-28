"""Apply the result of a call back onto the recovery case.

This is the seam between the conversation and the money. It runs after the call
ends, whatever transport placed it, so the rules here are testable without a
telephony provider in the loop.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
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

    @property
    def intent_sends_link(self) -> bool:
        """Whether this outcome should create a payment link.

        Lets the caller skip building a Razorpay client for a call that ended in
        "declined" and will never touch the API.
        """
        return outcome_for(self.intent).send_payment_link


async def apply_call_result(
    session: AsyncSession,
    case: RecoveryCase,
    call: VoiceCall,
    result: CallResult,
    *,
    customer: Customer | None = None,
    client=None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> None:
    """Record the call and move the case according to the detected intent.

    Intent decides the case's fate; the transcript is evidence, never input.
    Keeping it that way means a persuasive customer cannot talk the system into
    a state the policy does not allow.

    ``now`` moves the bookkeeping clocks -- the promise-to-pay deadline and the
    suppression timestamp -- so a simulated batch can run on simulated time.
    The call's own timestamps are always real, because a call either happened
    or it did not.
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

    settings = settings or get_settings()
    now = now or utcnow()
    outcome = outcome_for(result.intent)

    if outcome.promises_payment:
        # A promise starts a clock, not a status. Whether it was kept is read
        # later from recovered_at against this deadline, so nothing has to
        # remember to close it out.
        #
        # ``now`` is injectable so a simulated batch can run its promises on
        # simulated time. Stamping real time here while the batch advances by
        # simulated days would make every promise unkeepable-in-principle and
        # kept-in-practice, which is a 100% success rate that means nothing.
        promised_at = now
        case.promised_at = promised_at
        case.promise_due_at = promised_at + timedelta(hours=settings.promise_window_hours)
        await log_action(
            session,
            case,
            ActionType.PROMISE_MADE,
            {
                "voice_call_id": call.id,
                "intent": str(result.intent),
                "amount": case.original_amount,
                "due_at": case.promise_due_at.isoformat(),
            },
        )

    if outcome.status is not None:
        case.status = outcome.status

    if outcome.suppress_contact:
        # Close the door regardless of attempts remaining: an explicit no, a
        # wrong number and a disputed charge must all end the calling.
        case.attempt_count = case.max_attempts
        if customer is not None:
            # Burning the attempt budget only closes *this* case. The refusal
            # was from a person, and their next failed charge opens a new case
            # with a fresh budget, so the flag has to live on the customer.
            #
            # A wrong number is the exception, and the distinction matters: the
            # stranger who answered is not this customer. Banning the person
            # would also block an email payment link that has nothing to do
            # with the bad phone, and they still owe the money -- so mark the
            # number, not the human.
            if result.intent is CallIntent.WRONG_NUMBER:
                customer.phone_is_wrong = True
            else:
                customer.do_not_contact = True
                customer.do_not_contact_reason = str(result.intent)
                customer.do_not_contact_at = now or utcnow()
        else:
            # Worth shouting about: the case has no customer row, so the only
            # suppression we managed is case-scoped and the number could be
            # dialled again from another case.
            logger.error(
                "case %s asked to suppress contact but has no customer record; "
                "suppression is case-scoped only",
                case.id,
            )

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
