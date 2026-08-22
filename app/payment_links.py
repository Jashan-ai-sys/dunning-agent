"""Razorpay Payment Links as the recovery instrument.

Money never moves over the phone. When a customer says they want to pay, they
get a link -- which means the card details never touch us, the agent, or the
call recording.

Attribution is deliberately belt-and-braces. Every link carries both:

* ``reference_id = recovery-{case.id}`` -- a first-class field on the link
  entity, and the reliable key, and
* ``notes.recovery_case_id`` -- which Razorpay propagates onto the payment,
  though the docs show it arriving as an empty list in at least one case.

Either one is enough to credit the right case, so a single failure mode cannot
silently lose a recovery from the batch metrics.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.constants import ActionType
from app.models import Customer, RecoveryCase
from app.razorpay.client import RazorpayClient
from app.store import log_action, utcnow

logger = logging.getLogger(__name__)

REFERENCE_PREFIX = "recovery-"


@dataclass(frozen=True)
class PaymentLink:
    id: str
    short_url: str
    reference_id: str


def reference_id_for(case: RecoveryCase) -> str:
    """Razorpay caps reference_id at 40 characters; this is well inside it."""
    return f"{REFERENCE_PREFIX}{case.id}"


def case_id_from_reference(reference_id: str | None) -> int | None:
    """Inverse of :func:`reference_id_for`. None if it is not one of ours."""
    if not reference_id or not reference_id.startswith(REFERENCE_PREFIX):
        return None
    try:
        return int(reference_id[len(REFERENCE_PREFIX) :])
    except ValueError:
        return None


def build_payload(
    case: RecoveryCase, customer: Customer, *, settings: Settings, expire_at: int
) -> dict:
    payload: dict = {
        "amount": case.original_amount,
        "currency": case.currency,
        "description": f"Payment for your {settings.company_name} subscription",
        "reference_id": reference_id_for(case),
        "expire_by": expire_at,
        "reminder_enable": True,
        "notes": {
            "recovery_case_id": str(case.id),
            "failed_payment_id": case.razorpay_payment_id,
        },
        "notify": {"sms": bool(customer.phone), "email": bool(customer.email)},
    }
    contact = {}
    if customer.name:
        contact["name"] = customer.name
    if customer.phone:
        contact["contact"] = customer.phone
    if customer.email:
        contact["email"] = customer.email
    if contact:
        payload["customer"] = contact
    return payload


async def create_recovery_link(
    session: AsyncSession,
    case: RecoveryCase,
    customer: Customer,
    client: RazorpayClient,
    *,
    settings: Settings | None = None,
) -> PaymentLink | None:
    """Create and record a payment link for a case.

    Returns the existing link untouched if one has already been sent: a customer
    who agreed to pay twice should not receive two different links for the same
    debt, which would risk charging them twice.
    """
    settings = settings or get_settings()

    if case.payment_link_id and case.payment_link_url:
        logger.info("case %s already has payment link %s", case.id, case.payment_link_id)
        return PaymentLink(
            id=case.payment_link_id,
            short_url=case.payment_link_url,
            reference_id=reference_id_for(case),
        )

    expire_at = int(
        (utcnow() + timedelta(hours=settings.payment_link_expiry_hours)).timestamp()
    )
    payload = build_payload(case, customer, settings=settings, expire_at=expire_at)
    entity = await client.create_payment_link(payload)

    link = PaymentLink(
        id=entity["id"],
        short_url=entity["short_url"],
        reference_id=entity.get("reference_id") or reference_id_for(case),
    )
    case.payment_link_id = link.id
    case.payment_link_url = link.short_url
    case.payment_link_sent_at = utcnow()

    await log_action(
        session,
        case,
        ActionType.PAYMENT_LINK_CREATED,
        {
            "payment_link_id": link.id,
            "short_url": link.short_url,
            "reference_id": link.reference_id,
            "amount": case.original_amount,
            "expire_by": expire_at,
            "notified_sms": bool(customer.phone),
        },
    )
    logger.info("created payment link %s for case %s", link.id, case.id)
    return link
