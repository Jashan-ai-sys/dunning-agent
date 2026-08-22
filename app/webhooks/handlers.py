"""Razorpay event handlers.

Note on event names: Razorpay has no ``subscription.charged.failed`` event. A
failed recurring charge surfaces as ``payment.failed`` (the attempt) plus
``subscription.pending`` (Razorpay is retrying) and eventually
``subscription.halted`` (Razorpay gave up). Those three are our revenue-at-risk
triggers; ``subscription.charged``, ``payment.captured`` and
``payment_link.paid`` close the loop.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import ActionType
from app.models import Customer, RecoveryCase, Subscription
from app.payment_links import case_id_from_reference
from app.razorpay.client import RazorpayClient
from app.store import (
    get_open_cases_for_subscription,
    log_action,
    mark_recovered,
    open_recovery_case,
    upsert_customer,
    upsert_payment,
    upsert_subscription,
    utcnow,
)

logger = logging.getLogger(__name__)

Handler = Callable[[AsyncSession, dict[str, Any], RazorpayClient], Awaitable[None]]


def _note(entity: dict[str, Any], key: str) -> Any:
    """Read one note off an entity.

    Razorpay sends ``notes`` as an object when set and, per its own docs, as an
    empty *list* when not -- so this cannot assume a dict.
    """
    notes = entity.get("notes")
    return notes.get(key) if isinstance(notes, dict) else None


async def _resolve_invoice_context(
    client: RazorpayClient, invoice_id: str
) -> tuple[str | None, str | None]:
    """A payment entity carries no customer_id or subscription_id, only an
    invoice_id. The invoice is the join table."""
    invoice = await client.fetch_invoice(invoice_id)
    return invoice.get("subscription_id"), invoice.get("customer_id")


async def _ensure_customer(session: AsyncSession, client: RazorpayClient, customer_id: str) -> None:
    known = await session.execute(
        select(Customer.id).where(Customer.razorpay_customer_id == customer_id)
    )
    if known.scalar_one_or_none() is not None:
        return
    await upsert_customer(session, await client.fetch_customer(customer_id))


async def _ensure_subscription(
    session: AsyncSession, client: RazorpayClient, subscription_id: str
) -> None:
    known = await session.execute(
        select(Subscription.id).where(Subscription.razorpay_subscription_id == subscription_id)
    )
    if known.scalar_one_or_none() is not None:
        return
    await upsert_subscription(session, await client.fetch_subscription(subscription_id))


async def handle_payment_failed(
    session: AsyncSession, event: dict[str, Any], client: RazorpayClient
) -> None:
    """A charge attempt failed. If it belongs to a subscription, open a case."""
    payment = event["payload"]["payment"]["entity"]
    invoice_id = payment.get("invoice_id")

    subscription_id: str | None = None
    customer_id: str | None = None
    if invoice_id:
        subscription_id, customer_id = await _resolve_invoice_context(client, invoice_id)

    await upsert_payment(session, payment, subscription_id=subscription_id, customer_id=customer_id)

    if not subscription_id:
        # One-off checkout failure, not a subscription charge. Recorded for the
        # payment-degradation view; out of scope for subscription recovery.
        logger.info("payment %s failed without subscription linkage", payment["id"])
        return

    if customer_id:
        await _ensure_customer(session, client, customer_id)
    await _ensure_subscription(session, client, subscription_id)

    case = await open_recovery_case(
        session, payment=payment, subscription_id=subscription_id, customer_id=customer_id
    )
    if case is None:
        logger.info("case already open for payment %s (replayed webhook)", payment["id"])
        return

    await log_action(
        session,
        case,
        ActionType.CASE_OPENED,
        {
            "razorpay_event_id": event.get("razorpay_event_id"),
            "error_code": payment.get("error_code"),
            "error_description": payment.get("error_description"),
            "error_source": payment.get("error_source"),
            "error_step": payment.get("error_step"),
            "method": payment.get("method"),
            "amount": payment.get("amount"),
        },
    )


async def handle_subscription_pending(
    session: AsyncSession, event: dict[str, Any], client: RazorpayClient
) -> None:
    """Razorpay is retrying this subscription on its own. Record the signal."""
    subscription = event["payload"]["subscription"]["entity"]
    await upsert_subscription(session, subscription)

    cases = await get_open_cases_for_subscription(session, subscription["id"])
    if not cases:
        # We saw the subscription go pending without ever seeing the failed
        # payment. Phase 2's reconciler backfills these from the invoice list.
        logger.warning("subscription %s is pending with no open recovery case", subscription["id"])
        return

    for case in cases:
        await log_action(
            session,
            case,
            ActionType.SUBSCRIPTION_PENDING,
            {
                "razorpay_event_id": event.get("razorpay_event_id"),
                "auth_attempts": subscription.get("auth_attempts"),
                "status": subscription.get("status"),
            },
        )


async def handle_subscription_halted(
    session: AsyncSession, event: dict[str, Any], client: RazorpayClient
) -> None:
    """Razorpay exhausted its retries. This is the hard escalation trigger."""
    subscription = event["payload"]["subscription"]["entity"]
    await upsert_subscription(session, subscription)

    cases = await get_open_cases_for_subscription(session, subscription["id"])
    if not cases:
        logger.warning("subscription %s halted with no open recovery case", subscription["id"])
        return

    for case in cases:
        case.halted_at = utcnow()
        await log_action(
            session,
            case,
            ActionType.SUBSCRIPTION_HALTED,
            {
                "razorpay_event_id": event.get("razorpay_event_id"),
                "auth_attempts": subscription.get("auth_attempts"),
            },
        )


async def handle_subscription_charged(
    session: AsyncSession, event: dict[str, Any], client: RazorpayClient
) -> None:
    """A recurring charge succeeded, so any open case for it is now recovered."""
    subscription = event["payload"]["subscription"]["entity"]
    payment = event["payload"]["payment"]["entity"]

    await upsert_subscription(session, subscription)
    await upsert_payment(
        session,
        payment,
        subscription_id=subscription["id"],
        customer_id=subscription.get("customer_id"),
    )
    await _close_cases_for_subscription(session, subscription["id"], payment, event)


async def handle_payment_captured(
    session: AsyncSession, event: dict[str, Any], client: RazorpayClient
) -> None:
    """A payment succeeded. Attribute it to a case if we can.

    Recovery payment links carry ``notes.recovery_case_id`` (Phase 4), which is
    the precise attribution path. Otherwise fall back to the invoice linkage.
    """
    payment = event["payload"]["payment"]["entity"]
    case_id = _note(payment, "recovery_case_id")

    invoice_id = payment.get("invoice_id")
    subscription_id: str | None = None
    customer_id: str | None = None
    if invoice_id:
        subscription_id, customer_id = await _resolve_invoice_context(client, invoice_id)

    await upsert_payment(session, payment, subscription_id=subscription_id, customer_id=customer_id)

    if case_id is not None:
        case = await _get_case_by_id(session, case_id)
        if case is None:
            logger.warning("payment %s references unknown case %s", payment["id"], case_id)
            return
        await _record_recovery(session, case, payment, event)
        return

    if subscription_id:
        await _close_cases_for_subscription(session, subscription_id, payment, event)


async def handle_payment_link_paid(
    session: AsyncSession, event: dict[str, Any], client: RazorpayClient
) -> None:
    """A recovery payment link was paid.

    The most reliable attribution path: the link entity carries our own
    ``reference_id``, a first-class field, rather than depending on notes
    surviving the hop onto the payment entity.
    """
    link = event["payload"]["payment_link"]["entity"]
    payment = event["payload"]["payment"]["entity"]

    case_id = case_id_from_reference(link.get("reference_id")) or _note(
        link, "recovery_case_id"
    )

    await upsert_payment(session, payment)

    case = await _get_case_by_id(session, case_id)
    if case is None:
        logger.info("payment link %s is not one of ours; ignoring", link.get("id"))
        return

    await _record_recovery(session, case, payment, event)


async def _get_case_by_id(session: AsyncSession, case_id: Any) -> RecoveryCase | None:
    try:
        return await session.get(RecoveryCase, int(case_id))
    except (TypeError, ValueError):
        return None


async def _close_cases_for_subscription(
    session: AsyncSession,
    subscription_id: str,
    payment: dict[str, Any],
    event: dict[str, Any],
) -> None:
    for case in await get_open_cases_for_subscription(session, subscription_id):
        await _record_recovery(session, case, payment, event)


async def _record_recovery(
    session: AsyncSession, case: RecoveryCase, payment: dict[str, Any], event: dict[str, Any]
) -> None:
    if not await mark_recovered(
        session, case, payment_id=payment["id"], amount=payment.get("amount", 0)
    ):
        return
    await log_action(
        session,
        case,
        ActionType.PAYMENT_CAPTURED,
        {
            "razorpay_event_id": event.get("razorpay_event_id"),
            "razorpay_payment_id": payment["id"],
            "amount": payment.get("amount"),
            "method": payment.get("method"),
        },
    )


EVENT_HANDLERS: dict[str, Handler] = {
    "payment.failed": handle_payment_failed,
    "payment.captured": handle_payment_captured,
    "subscription.pending": handle_subscription_pending,
    "subscription.halted": handle_subscription_halted,
    "subscription.charged": handle_subscription_charged,
    "payment_link.paid": handle_payment_link_paid,
}
