"""Idempotent persistence helpers shared by the webhook handlers and (from
Phase 2) the recovery orchestrator.

Every write here is safe to replay: Razorpay delivers webhooks at-least-once, so
a handler may run twice for the same event.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import ActionType, CaseSource, CaseStatus
from app.models import Customer, Payment, RecoveryAction, RecoveryCase, Subscription
from app.priority import tier_from


def utcnow() -> datetime:
    return datetime.now(UTC)


async def upsert_customer(session: AsyncSession, entity: dict[str, Any]) -> None:
    """Insert or refresh a customer from a Razorpay customer entity."""
    values = {
        "razorpay_customer_id": entity["id"],
        "name": entity.get("name"),
        "phone": entity.get("contact"),
        "email": entity.get("email"),
    }
    stmt = insert(Customer).values(**values)
    # preferred_language is ours, not Razorpay's -- never overwrite it on refresh.
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Customer.razorpay_customer_id],
            set_={k: v for k, v in values.items() if k != "razorpay_customer_id"},
        )
    )


async def upsert_subscription(session: AsyncSession, entity: dict[str, Any]) -> None:
    values = {
        "razorpay_subscription_id": entity["id"],
        "razorpay_customer_id": entity.get("customer_id"),
        "razorpay_plan_id": entity.get("plan_id"),
        "status": entity.get("status"),
    }
    stmt = insert(Subscription).values(**values)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Subscription.razorpay_subscription_id],
            set_={k: v for k, v in values.items() if k != "razorpay_subscription_id"},
        )
    )


async def upsert_payment(
    session: AsyncSession,
    entity: dict[str, Any],
    *,
    subscription_id: str | None = None,
    customer_id: str | None = None,
) -> None:
    """Persist a payment entity from a payment.* or subscription.charged webhook.

    ``subscription_id``/``customer_id`` are passed in because a payment entity
    does not carry them -- they are resolved from the invoice.
    """
    values = {
        "razorpay_payment_id": entity["id"],
        "razorpay_invoice_id": entity.get("invoice_id"),
        "razorpay_order_id": entity.get("order_id"),
        "razorpay_subscription_id": subscription_id,
        "razorpay_customer_id": customer_id,
        "amount": entity.get("amount", 0),
        "currency": entity.get("currency", "INR"),
        "status": entity.get("status", "failed"),
        "method": entity.get("method"),
        "error_code": entity.get("error_code"),
        "error_description": entity.get("error_description"),
        "error_reason": entity.get("error_reason"),
        "error_source": entity.get("error_source"),
        "error_step": entity.get("error_step"),
        "notes": entity.get("notes") or None,
    }
    stmt = insert(Payment).values(**values)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Payment.razorpay_payment_id],
            set_={k: v for k, v in values.items() if k != "razorpay_payment_id"},
        )
    )


async def get_open_cases_for_subscription(
    session: AsyncSession, subscription_id: str
) -> list[RecoveryCase]:
    """Cases still worth acting on -- excludes recovered/declined/stopped."""
    result = await session.execute(
        select(RecoveryCase)
        .where(RecoveryCase.razorpay_subscription_id == subscription_id)
        .where(RecoveryCase.status.in_([CaseStatus.OPEN, CaseStatus.IN_PROGRESS]))
        .order_by(RecoveryCase.created_at)
    )
    return list(result.scalars())


async def get_open_case_for_invoice(
    session: AsyncSession, invoice_id: str
) -> RecoveryCase | None:
    """An unfinished case already chasing this invoice, if there is one.

    Cases are keyed on ``razorpay_payment_id``, which is right for idempotency
    -- the same webhook delivered twice must not open two cases. It is wrong
    for *debt* identity: every retry against one invoice is a new payment id,
    so a subscription that fails authorisation four times opens four cases for
    one Rs 499 debt, each with its own untouched contact budget.

    The invoice is the debt. This is how a repeat failure finds it.
    """
    result = await session.execute(
        select(RecoveryCase)
        .where(RecoveryCase.razorpay_invoice_id == invoice_id)
        .where(RecoveryCase.status.in_([CaseStatus.OPEN, CaseStatus.IN_PROGRESS]))
        .order_by(RecoveryCase.created_at)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def open_recovery_case(
    session: AsyncSession,
    *,
    payment: dict[str, Any],
    subscription_id: str | None,
    customer_id: str | None,
    source: str = CaseSource.RAZORPAY,
    priority_tier: int | None = None,
    next_eligible_at: datetime | None = None,
) -> RecoveryCase | None:
    """Open a case for a failed charge. Returns None if one already exists.

    Insert-then-check rather than check-then-insert: two concurrent deliveries of
    the same event would both clear a prior SELECT, and the loser would then die
    on the unique constraint. Letting Postgres arbitrate makes the duplicate a
    quiet no-op instead.
    """
    stmt = (
        insert(RecoveryCase)
        .values(
            razorpay_payment_id=payment["id"],
            razorpay_invoice_id=payment.get("invoice_id"),
            razorpay_subscription_id=subscription_id,
            razorpay_customer_id=customer_id,
            original_amount=payment.get("amount", 0),
            currency=payment.get("currency", "INR"),
            failure_code=payment.get("error_code"),
            failure_reason=payment.get("error_description") or payment.get("error_reason"),
            failure_source=payment.get("error_source"),
            failure_reason_code=payment.get("error_reason"),
            failure_step=payment.get("error_step"),
            razorpay_order_id=payment.get("order_id"),
            priority_tier=priority_tier
            if priority_tier is not None
            else tier_from(
                payment.get("error_source"),
                payment.get("error_reason"),
                payment.get("error_step"),
            ),
            status=CaseStatus.OPEN,
            source=source,
            next_eligible_at=next_eligible_at,
        )
        .on_conflict_do_nothing(index_elements=[RecoveryCase.razorpay_payment_id])
        .returning(RecoveryCase.id)
    )
    case_id = (await session.execute(stmt)).scalar_one_or_none()
    if case_id is None:
        return None
    return await session.get(RecoveryCase, case_id)


async def mark_recovered(
    session: AsyncSession,
    case: RecoveryCase,
    *,
    payment_id: str,
    amount: int,
) -> bool:
    """Close a case as recovered. Returns False if it was already closed."""
    if case.status == CaseStatus.RECOVERED:
        return False
    case.status = CaseStatus.RECOVERED
    case.recovered_payment_id = payment_id
    case.recovered_amount = amount
    case.recovered_at = utcnow()
    return True


async def log_action(
    session: AsyncSession,
    case: RecoveryCase,
    action_type: ActionType,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append one immutable row to the case's audit trail."""
    session.add(
        RecoveryAction(
            recovery_case_id=case.id,
            action_type=action_type,
            metadata_json=metadata,
        )
    )
