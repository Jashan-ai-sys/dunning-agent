"""The recovery orchestrator: one tick of the work loop.

Reads actionable cases, asks the policy what to do with each, and executes that
decision. Every decision that changes a case is written to ``recovery_actions``,
so the audit trail explains not just what happened but why.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.channels import ContactChannel
from app.config import Settings, get_settings
from app.constants import ActionType, CaseStatus
from app.mandate import charge_mandate
from app.models import Customer, RecoveryCase
from app.payment_links import create_recovery_link
from app.policy import Action, StopReason, decide
from app.priority import score_expression
from app.razorpay.client import RazorpayClient
from app.store import log_action, utcnow

logger = logging.getLogger(__name__)


@dataclass
class TickResult:
    considered: int = 0
    #: Calls placed.
    contacted: int = 0
    #: Payment links sent without spending a call -- the cheap intervention.
    linked: int = 0
    #: Mandates re-charged without the customer doing anything.
    charged: int = 0
    stopped: int = 0
    #: Of the stopped, how many were closed because *we* are misconfigured.
    needs_human: int = 0
    waiting: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "considered": self.considered,
            "contacted": self.contacted,
            "linked": self.linked,
            "charged": self.charged,
            "stopped": self.stopped,
            "needs_human": self.needs_human,
            "waiting": self.waiting,
            "failed": self.failed,
        }


async def _claim_actionable_cases(
    session: AsyncSession, limit: int, *, now: datetime
) -> list[RecoveryCase]:
    """Lock a batch of open cases for this worker.

    ``SKIP LOCKED`` means a second worker takes the next batch instead of
    blocking, so the loop scales horizontally without double-calling anyone.
    Ordered by :func:`app.priority.score` -- a weighted combination of the
    tier (broken mandate, then a customer who was actually trying to pay, then
    everything else), the size of the debt, and whether Razorpay has given up
    retrying. A large enough debt can outrank a higher tier; a marginal one
    cannot. Age breaks ties, oldest first, so nothing starves.

    Cases the policy has parked are skipped until their ``next_eligible_at``.
    They are still OPEN and still among the oldest rows, so without this filter
    they sort to the front of every batch, and once there are more of them than
    ``worker_batch_size`` nothing newer is ever claimed again.
    """
    result = await session.execute(
        select(RecoveryCase)
        .where(RecoveryCase.status.in_([CaseStatus.OPEN, CaseStatus.IN_PROGRESS]))
        .where(
            or_(
                RecoveryCase.next_eligible_at.is_(None),
                RecoveryCase.next_eligible_at <= now,
            )
        )
        .order_by(score_expression().desc(), RecoveryCase.created_at)
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    return list(result.scalars())


async def _customer_for(session: AsyncSession, case: RecoveryCase) -> Customer | None:
    if not case.razorpay_customer_id:
        return None
    result = await session.execute(
        select(Customer).where(Customer.razorpay_customer_id == case.razorpay_customer_id)
    )
    return result.scalar_one_or_none()


async def run_once(
    session: AsyncSession,
    channel: ContactChannel,
    *,
    now: datetime | None = None,
    settings: Settings | None = None,
    client: RazorpayClient | None = None,
) -> TickResult:
    """Process one batch of cases. Safe to call concurrently.

    ``client`` is injectable for the same reason ``channel`` is: the link
    intervention talks to Razorpay, and a test of the loop should not. Building
    one is free -- it only reads settings, and opens no connection until a
    request is actually made.
    """
    settings = settings or get_settings()
    now = now or utcnow()
    client = client or RazorpayClient()
    result = TickResult()

    for case in await _claim_actionable_cases(session, settings.worker_batch_size, now=now):
        result.considered += 1
        customer = await _customer_for(session, case)
        decision = decide(case, customer, now=now, settings=settings)

        if decision.action is Action.WAIT:
            # Not logged: a waiting case is re-evaluated every tick, and writing
            # a row each time would bury the meaningful entries. The parking
            # time is a column, not an audit entry, for the same reason.
            case.next_eligible_at = decision.retry_after
            result.waiting += 1
            continue

        await log_action(session, case, ActionType.POLICY_DECISION, decision.as_metadata())
        case.next_eligible_at = None

        if decision.action is Action.STOP:
            case.status = CaseStatus.STOPPED
            await log_action(session, case, ActionType.STOPPED, {"reason": decision.reason})
            result.stopped += 1
            if decision.reason == StopReason.NEEDS_HUMAN:
                # A configuration failure is ours and fixable, so closing the
                # case quietly would bury recoverable money among cases that
                # simply ran out of attempts. Count it and say so.
                result.needs_human += 1
                logger.warning(
                    "case %s stopped for human review: %s failure (%s)",
                    case.id,
                    decision.root_cause,
                    case.failure_reason,
                )
            continue

        if decision.action is Action.RETRY_MANDATE:
            # Not a spent attempt on failure: nothing reached the customer, so
            # the next tick falls through to a payment link instead.
            delivered = await _retry_mandate(
                session, case, customer, client, now=now, settings=settings
            )
            _record_delivery(case, customer, result, delivered, "charged", now=now)
            continue

        if decision.action is Action.LINK:
            delivered = await _send_link(
                session, case, customer, client, now=now, settings=settings
            )
            _record_delivery(case, customer, result, delivered, "linked", now=now)
            continue

        delivered = await _contact(session, case, customer, channel, now=now)
        _record_delivery(case, customer, result, delivered, "contacted", now=now)

    await session.commit()
    return result


def _record_delivery(
    case: RecoveryCase,
    customer: Customer,
    result: TickResult,
    delivered: bool,
    counter: str,
    *,
    now: datetime,
) -> None:
    """Tally one delivery attempt and track consecutive our-side failures.

    ``delivery_failures`` is separate from ``attempt_count`` on purpose: a
    telephony outage must not spend the customer's contact budget. But it has
    to be counted somewhere, or a case whose delivery can never succeed is
    retried every backoff window forever without ever reaching the attempt cap.

    ``last_contacted_at`` moves on the *customer*, not the case, and only when
    something actually reached them. Within a tick that is what stops a person
    with several open debts being rung once per debt: the policy re-reads the
    same customer row for the next case and sees the cooldown.
    """
    if delivered:
        case.delivery_failures = 0
        customer.last_contacted_at = now
        setattr(result, counter, getattr(result, counter) + 1)
    else:
        case.delivery_failures = (case.delivery_failures or 0) + 1
        result.failed += 1


async def _retry_mandate(
    session: AsyncSession,
    case: RecoveryCase,
    customer: Customer,
    client: RazorpayClient,
    *,
    now: datetime,
    settings: Settings,
) -> bool:
    """Charge an existing mandate instead of asking the customer for anything.

    ``last_attempt_at`` moves either way so a broken integration backs off, but
    ``attempt_count`` only moves on a charge that Razorpay accepted -- a
    customer with no chargeable mandate has not used up one of their contacts,
    because nothing contacted them.
    """
    case.status = CaseStatus.IN_PROGRESS
    case.last_attempt_at = now

    try:
        charged = await charge_mandate(
            session, case, customer, client, settings=settings, now_epoch=int(now.timestamp())
        )
    except Exception as exc:  # noqa: BLE001 - Razorpay is external; never kill the tick
        logger.exception("mandate retry failed for case %s", case.id)
        await log_action(
            session, case, ActionType.MANDATE_RETRIED, {"error": repr(exc), "charged": False}
        )
        return False

    if charged:
        case.attempt_count += 1
    return charged


async def _send_link(
    session: AsyncSession,
    case: RecoveryCase,
    customer: Customer,
    client: RazorpayClient,
    *,
    now: datetime,
    settings: Settings,
) -> bool:
    """Send the payment link, without placing a call.

    Counts as a contact attempt, and moves the same two fields ``_contact``
    does: a link is cheaper than a call but it is still a message to a
    customer, so it consumes the same budget and respects the same backoff.
    """
    case.status = CaseStatus.IN_PROGRESS
    case.last_attempt_at = now

    try:
        await create_recovery_link(session, case, customer, client, settings=settings, now=now)
    except Exception as exc:  # noqa: BLE001 - Razorpay is external; never kill the tick
        logger.exception("payment link failed for case %s", case.id)
        await log_action(
            session, case, ActionType.PAYMENT_LINK_CREATED, {"error": repr(exc), "created": False}
        )
        return False

    # create_recovery_link writes the audit row itself, on both the create and
    # the resend path, so there is exactly one entry per attempt either way.
    case.attempt_count += 1
    return True


async def _contact(
    session: AsyncSession,
    case: RecoveryCase,
    customer: Customer,
    channel: ContactChannel,
    *,
    now: datetime,
) -> bool:
    """Initiate contact and record the outcome.

    ``last_attempt_at`` moves even when initiation fails, so a broken channel
    backs off instead of hot-looping. ``attempt_count`` only moves on success,
    so an outage on our side does not burn the customer's attempt budget.
    """
    case.status = CaseStatus.IN_PROGRESS
    case.last_attempt_at = now

    try:
        outcome = await channel.initiate(case, customer)
    except Exception as exc:  # noqa: BLE001 - the channel is external; never kill the tick
        logger.exception("contact failed for case %s", case.id)
        await log_action(
            session, case, ActionType.VOICE_CALL, {"error": repr(exc), "placed": False}
        )
        return False

    case.attempt_count += 1
    await log_action(session, case, ActionType.VOICE_CALL, outcome.as_metadata())
    return True
