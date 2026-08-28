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
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.constants import ActionType
from app.models import Customer, RecoveryCase
from app.razorpay.client import RazorpayClient
from app.store import log_action, utcnow

logger = logging.getLogger(__name__)

REFERENCE_PREFIX = "recovery-"

#: Razorpay rejects a reference_id longer than this.
MAX_REFERENCE_CHARS = 40


@dataclass(frozen=True)
class PaymentLink:
    id: str
    short_url: str
    reference_id: str


def reference_id_for(case: RecoveryCase, *, settings: Settings | None = None) -> str:
    """The attribution key we put on every payment link.

    Razorpay enforces uniqueness on reference_id across the whole account, and
    forever. Two things follow from that, and both are real 400s we have hit.

    Keyed on the case alone, the second link a case ever needs is rejected:

        payment link with given reference_id: recovery-1 already exists

    hence the attempt suffix. And case ids restart at 1 on a fresh database, so
    a redeploy against a new instance collides with links the account has held
    since the last one -- hence the namespace, which is empty by default and
    reproduces the old format exactly.

    The namespace goes on the *end* deliberately: the case id stays the first
    segment, so every reference ever issued, with or without one, still parses
    back to its case.
    """
    settings = settings or get_settings()
    # `or 0` rather than the raw value: a case built in memory and not yet
    # flushed has attempt_count None, and "recovery-7-None" would be sent to
    # Razorpay as a real, permanently-reserved reference.
    reference = f"{REFERENCE_PREFIX}{case.id}-{case.attempt_count or 0}"

    namespace = settings.payment_reference_namespace.strip()
    if namespace:
        reference = f"{reference}-{namespace}"
    # Truncating from the end costs the namespace; truncating from the front
    # would cost the case id, which is the one part that must survive.
    return reference[:MAX_REFERENCE_CHARS]


def case_id_from_reference(reference_id: str | None) -> int | None:
    """Inverse of :func:`reference_id_for`. None if it is not one of ours.

    Reads only the case segment, so links written before the attempt suffix
    existed (``recovery-1``) still attribute correctly -- those payments are
    real money and must not stop reconciling because the format moved on.
    """
    if not reference_id or not reference_id.startswith(REFERENCE_PREFIX):
        return None
    try:
        return int(reference_id[len(REFERENCE_PREFIX) :].split("-")[0])
    except (ValueError, IndexError):
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


async def _notify_existing_link(
    case: RecoveryCase, customer: Customer, client: RazorpayClient
) -> list[str]:
    """Re-send a link the case already has, over whatever we can reach.

    Never raises. A failed re-notification is not a reason to break a call that
    is otherwise going well: the customer still has the original message and
    the agent reads the URL out loud anyway.

    Returns the media it actually reached them on, which may be empty. The
    caller writes that into the audit trail rather than assuming success -- an
    entry claiming a link was resent when nothing left the building is worse
    than no entry at all.
    """
    delivered: list[str] = []
    media = [("sms", customer.phone), ("email", customer.email)]
    for medium, contact in media:
        if not contact:
            continue
        try:
            await client.notify_payment_link(case.payment_link_id, medium)
        except Exception:  # noqa: BLE001 - never break a call over a resend
            logger.exception(
                "could not re-send payment link %s by %s", case.payment_link_id, medium
            )
        else:
            logger.info("re-sent payment link %s by %s", case.payment_link_id, medium)
            delivered.append(medium)
    return delivered


async def create_recovery_link(
    session: AsyncSession,
    case: RecoveryCase,
    customer: Customer,
    client: RazorpayClient,
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> PaymentLink:
    """Create and record a payment link for a case.

    Returns the existing link untouched if one has already been sent: a customer
    who agreed to pay twice should not receive two different links for the same
    debt, which would risk charging them twice.
    """
    settings = settings or get_settings()
    now = now or utcnow()

    if case.payment_link_id and case.payment_link_url:
        logger.info("case %s already has payment link %s", case.id, case.payment_link_id)
        # Deliver it again. Razorpay notifies once, at creation, so without this
        # a customer called a second time hears the agent say it is sending a
        # link and receives nothing. Re-notifying is the right half to repeat:
        # a second *link* would risk charging them twice, a second SMS only
        # tells them again what they just asked to be told.
        resent_via = await _notify_existing_link(case, customer, client)
        # Log the resend. Callers count on every return from here leaving an
        # audit row -- the orchestrator burns an attempt on the strength of it,
        # and an attempt with nothing beside it in the trail is unexplainable.
        # ``resent_via`` is what actually went out, empty list included: the
        # trail has to be able to show an attempt that reached nobody.
        await log_action(
            session,
            case,
            ActionType.PAYMENT_LINK_CREATED,
            {
                "payment_link_id": case.payment_link_id,
                "short_url": case.payment_link_url,
                "created": False,
                "resent_via": resent_via,
            },
        )
        return PaymentLink(
            id=case.payment_link_id,
            short_url=case.payment_link_url,
            reference_id=reference_id_for(case),
        )

    expire_at = int(
        (now + timedelta(hours=settings.payment_link_expiry_hours)).timestamp()
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
    case.payment_link_sent_at = now

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
