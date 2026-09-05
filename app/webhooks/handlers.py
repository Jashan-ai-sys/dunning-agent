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
from datetime import timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.constants import ActionType, CaseSource, CaseStatus
from app.models import Customer, RecoveryCase, Subscription
from app.payment_links import case_id_from_reference
from app.priority import Priority
from app.razorpay.client import RazorpayClient
from app.store import (
    get_open_case_for_invoice,
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
        # One-off checkout failure. Not a subscription charge, but not nothing
        # either: somebody reached the payment page, tried, and it broke under
        # them. That is the abandoned-checkout case.
        await _open_checkout_case(session, payment, event)
        return

    if customer_id:
        await _ensure_customer(session, client, customer_id)
    await _ensure_subscription(session, client, subscription_id)

    # One debt, one case. Every retry against the same invoice arrives with a
    # fresh payment id, so keying only on that opens a new case -- and a new
    # contact budget -- per attempt. A repeat failure is more evidence about a
    # debt we are already chasing, not a second debt.
    if invoice_id:
        existing = await get_open_case_for_invoice(session, invoice_id)
        if existing is not None and existing.razorpay_payment_id == payment["id"]:
            # The same attempt delivered twice, not a new one. Razorpay retries
            # deliveries, and our own replay sweep re-dispatches; recording a
            # "repeat failure" for either would invent an attempt that never
            # happened. open_recovery_case is idempotent on the payment id, so
            # falling through is safe -- but it would also log CASE_OPENED
            # again, which is why this returns instead.
            logger.info("payment %s replayed; case %s already open", payment["id"], existing.id)
            return
        if existing is not None:
            await log_action(
                session,
                existing,
                ActionType.REPEAT_FAILURE,
                {
                    "razorpay_event_id": event.get("razorpay_event_id"),
                    "razorpay_payment_id": payment["id"],
                    "error_reason": payment.get("error_reason"),
                    "error_source": payment.get("error_source"),
                    "error_step": payment.get("error_step"),
                },
            )
            logger.info(
                "payment %s is another failure on invoice %s; recorded against case %s",
                payment["id"],
                invoice_id,
                existing.id,
            )
            return

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


async def _open_checkout_case(
    session: AsyncSession, payment: dict[str, Any], event: dict[str, Any]
) -> None:
    """Open a case for a checkout somebody attempted and walked away from.

    Two things make this different from a failed subscription charge, and both
    are handled here rather than in the policy:

    * **It is parked on arrival.** A customer who retries with another card two
      minutes later has not abandoned anything, and chasing them mid-purchase
      would be absurd. The case sits until ``checkout_grace_minutes`` is up; if
      they pay in the meantime, ``order.paid`` closes it and nobody is
      contacted at all.
    * **It is tier 3 whatever the card said.** The failure fields would often
      score this as an attempted payment, but the relationship is weaker than a
      live subscriber's -- one purchase attempt, not a standing mandate -- so it
      queues behind them.

    We can only work it if we already know the person. The order entity carries
    no customer at all, and the payment entity carries a bare email and phone
    with no Razorpay customer id, so inventing a customer record for a stranger
    is a product and consent decision rather than a technical one. Unmatched
    attempts are logged and counted, not quietly dropped.
    """
    settings = get_settings()
    customer = await _match_known_customer(session, payment)
    if customer is None:
        logger.info(
            "checkout payment %s failed but the payer is not a known customer; "
            "detected, not actionable",
            payment["id"],
        )
        return

    case = await open_recovery_case(
        session,
        payment=payment,
        subscription_id=None,
        customer_id=customer.razorpay_customer_id,
        source=CaseSource.CHECKOUT,
        priority_tier=Priority.CHECKOUT_ABANDONED,
        next_eligible_at=utcnow() + timedelta(minutes=settings.checkout_grace_minutes),
    )
    if case is None:
        logger.info("checkout case already open for payment %s", payment["id"])
        return

    await log_action(
        session,
        case,
        ActionType.CHECKOUT_ABANDONED,
        {
            "razorpay_event_id": event.get("razorpay_event_id"),
            "razorpay_order_id": payment.get("order_id"),
            "amount": payment.get("amount"),
            "method": payment.get("method"),
            "error_reason": payment.get("error_reason"),
            "grace_minutes": settings.checkout_grace_minutes,
        },
    )


async def _match_known_customer(
    session: AsyncSession, payment: dict[str, Any]
) -> Customer | None:
    """Find the payer among customers we already have a relationship with.

    Matched on the contact details Razorpay puts on the payment entity, since a
    one-off checkout carries no ``customer_id``.
    """
    contact = payment.get("contact")
    email = payment.get("email")
    filters = []
    if contact:
        filters.append(Customer.phone == contact)
    if email:
        filters.append(Customer.email == email)
    if not filters:
        return None

    result = await session.execute(select(Customer).where(or_(*filters)).limit(1))
    return result.scalar_one_or_none()


async def handle_order_paid(
    session: AsyncSession, event: dict[str, Any], client: RazorpayClient
) -> None:
    """The order behind an abandoned checkout was paid after all.

    Closes the case whether we contacted them or not -- including during the
    grace window, which is the common and desirable outcome: they retried on
    their own and the agent never had to ring.
    """
    order = event["payload"]["order"]["entity"]
    payment = event["payload"]["payment"]["entity"]

    await upsert_payment(session, payment)

    result = await session.execute(
        select(RecoveryCase)
        .where(RecoveryCase.razorpay_order_id == order["id"])
        .where(RecoveryCase.status.in_([CaseStatus.OPEN, CaseStatus.IN_PROGRESS]))
    )
    for case in result.scalars():
        await _record_recovery(session, case, payment, event)


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



async def handle_payment_authorized(
    session: AsyncSession, event: dict[str, Any], client: RazorpayClient
) -> None:
    """A payment succeeded but has not been captured.

    Subscribed to for one signal in particular: a zero-amount eMandate
    registration. That is not a debt and must never open a recovery case --
    nothing is owed, the customer paid nothing, and there is nothing to chase.
    It is recorded and nothing else happens here.

    What makes it worth recording is what it means when the *next* event does
    not arrive. A mandate registration that authorises and is never followed by
    ``subscription.authenticated`` has failed silently: the money side looks
    perfect, and every future charge on that subscription is already broken.
    Razorpay signals that by omission -- there is no "mandate failed" webhook --
    so the only way to catch it is to notice the absence, which needs a sweep
    rather than a handler. See `Known gaps`.
    """
    payment = event["payload"]["payment"]["entity"]
    await upsert_payment(session, payment)

    if payment.get("amount") == 0 and payment.get("method") == "emandate":
        logger.info(
            "mandate registration authorised: payment %s token %s -- expecting "
            "subscription.authenticated to follow",
            payment["id"],
            payment.get("token_id"),
        )
        return

    logger.info("payment %s authorised (not captured); recorded, no case", payment["id"])


async def handle_subscription_authenticated(
    session: AsyncSession, event: dict[str, Any], client: RazorpayClient
) -> None:
    """The mandate registered and the subscription is live.

    The counterpart to the zero-amount authorisation above: this is the event
    whose *absence* means a silent mandate failure. Recording it is what makes
    that absence detectable later.

    It also attaches the customer. Until a subscription authenticates there is
    no customer on it, which is why a failed authorisation opens a case with no
    contact details and the policy correctly stops it.
    """
    subscription = event["payload"]["subscription"]["entity"]
    await upsert_subscription(session, subscription)

    customer_id = subscription.get("customer_id")
    if customer_id:
        await _ensure_customer(session, client, customer_id)

    logger.info(
        "subscription %s authenticated (customer %s); mandate is live",
        subscription["id"],
        customer_id,
    )



EVENT_HANDLERS: dict[str, Handler] = {
    "payment.failed": handle_payment_failed,
    "payment.authorized": handle_payment_authorized,
    "payment.captured": handle_payment_captured,
    "subscription.authenticated": handle_subscription_authenticated,
    "subscription.pending": handle_subscription_pending,
    "subscription.halted": handle_subscription_halted,
    "subscription.charged": handle_subscription_charged,
    "payment_link.paid": handle_payment_link_paid,
    "order.paid": handle_order_paid,
}
