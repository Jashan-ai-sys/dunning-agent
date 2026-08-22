"""The recovery orchestrator: one tick of the work loop.

Reads actionable cases, asks the policy what to do with each, and executes that
decision. Every decision that changes a case is written to ``recovery_actions``,
so the audit trail explains not just what happened but why.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.channels import ContactChannel
from app.config import Settings, get_settings
from app.constants import ActionType, CaseStatus
from app.models import Customer, RecoveryCase
from app.policy import Action, decide
from app.store import log_action, utcnow

logger = logging.getLogger(__name__)


@dataclass
class TickResult:
    considered: int = 0
    contacted: int = 0
    stopped: int = 0
    waiting: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "considered": self.considered,
            "contacted": self.contacted,
            "stopped": self.stopped,
            "waiting": self.waiting,
            "failed": self.failed,
        }


async def _claim_actionable_cases(session: AsyncSession, limit: int) -> list[RecoveryCase]:
    """Lock a batch of open cases for this worker.

    ``SKIP LOCKED`` means a second worker takes the next batch instead of
    blocking, so the loop scales horizontally without double-calling anyone.
    Cases Razorpay has already given up on (``halted_at``) are worked first.
    """
    result = await session.execute(
        select(RecoveryCase)
        .where(RecoveryCase.status.in_([CaseStatus.OPEN, CaseStatus.IN_PROGRESS]))
        .order_by(RecoveryCase.halted_at.nulls_last(), RecoveryCase.created_at)
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
) -> TickResult:
    """Process one batch of cases. Safe to call concurrently."""
    settings = settings or get_settings()
    now = now or utcnow()
    result = TickResult()

    for case in await _claim_actionable_cases(session, settings.worker_batch_size):
        result.considered += 1
        customer = await _customer_for(session, case)
        decision = decide(case, customer, now=now, settings=settings)

        if decision.action is Action.WAIT:
            # Not logged: a waiting case is re-evaluated every tick, and writing
            # a row each time would bury the meaningful entries.
            result.waiting += 1
            continue

        await log_action(session, case, ActionType.POLICY_DECISION, decision.as_metadata())

        if decision.action is Action.STOP:
            case.status = CaseStatus.STOPPED
            await log_action(session, case, ActionType.STOPPED, {"reason": decision.reason})
            result.stopped += 1
            continue

        contacted = await _contact(session, case, customer, channel, now=now)
        if contacted:
            result.contacted += 1
        else:
            result.failed += 1

    await session.commit()
    return result


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
