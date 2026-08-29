"""Charging a mandate the customer already authorised.

The "mandate retry sequencer" half of Track 03. Razorpay retries a failed
subscription charge on its own; what it cannot do is retry at a moment *we*
have reason to believe will work. A customer whose card simply had no money in
it does not need to re-enter any details -- the mandate is still valid, and the
right recovery is to charge it again rather than to send them a link.

Two deliberate constraints, because this is the only path in the service that
moves money without the customer touching anything:

* it is **off by default** (``mandate_retry_enabled``), so a deployment opts in
  rather than discovers it, and
* it only ever runs where the diagnosis says the *instrument* is fine and the
  *funds* were not. A revoked mandate or a dead card is not retried, and a
  gateway failure is not the customer's to pay for twice.

A charge and a payment link are alternatives, never both on the same attempt:
two live ways to pay one debt is how a customer gets charged twice.
"""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.constants import ActionType
from app.models import Customer, RecoveryCase
from app.payment_links import reference_id_for
from app.razorpay.client import RazorpayClient
from app.store import log_action

logger = logging.getLogger(__name__)


def usable_token(tokens: list[dict[str, Any]], *, amount: int, now_epoch: int) -> dict | None:
    """The first token that can actually be charged for ``amount``.

    Every condition here is a way the charge would otherwise fail at Razorpay,
    and one of them -- ``max_amount`` -- would fail *after* we had already told
    the customer we were taking the money. E-mandates carry a ceiling agreed at
    authorisation; a plan that has since been upgraded can sit above it.

    Returns None rather than raising: no chargeable mandate is an ordinary
    outcome, and the caller falls back to a payment link.
    """
    candidates = []
    for token in tokens:
        if not token.get("recurring"):
            continue
        # Cards report confirmation status; e-mandate tokens often omit it.
        status = (token.get("recurring_details") or {}).get("status")
        if status is not None and status != "confirmed":
            continue
        expires = token.get("expired_at")
        if expires is not None and expires <= now_epoch:
            continue
        ceiling = token.get("max_amount")
        if ceiling is not None and ceiling < amount:
            continue
        candidates.append(token)

    if not candidates:
        return None
    # Most recently used first: the instrument the customer actually transacts
    # on is likelier to work than one they abandoned.
    return max(candidates, key=lambda t: t.get("used_at") or 0)


async def charge_mandate(
    session: AsyncSession,
    case: RecoveryCase,
    customer: Customer,
    client: RazorpayClient,
    *,
    settings: Settings,
    now_epoch: int,
) -> bool:
    """Re-charge the customer's existing mandate for this case.

    Returns False on any failure, having logged why. The caller must then treat
    the attempt as not spent, so the case falls back to a payment link on the
    next tick rather than being left with nothing.
    """
    # Razorpay requires both email and contact on a recurring charge and
    # rejects the request without them. A number we were told is somebody
    # else's is not a contact for this customer, whatever the column says.
    reachable = customer.phone and not customer.phone_is_wrong
    if not (customer.razorpay_customer_id and customer.email and reachable):
        await _log_failure(session, case, "missing customer details for a recurring charge")
        return False

    tokens = await client.fetch_customer_tokens(customer.razorpay_customer_id)
    token = usable_token(
        tokens.get("items") or [], amount=case.original_amount, now_epoch=now_epoch
    )
    if token is None:
        await _log_failure(session, case, "no chargeable mandate on file")
        return False

    order = await client.create_order(
        {
            "amount": case.original_amount,
            "currency": case.currency,
            # Unique per attempt, for the same reason the payment link's
            # reference_id is: Razorpay holds these forever.
            "receipt": reference_id_for(case),
            "notes": {"recovery_case_id": str(case.id)},
        }
    )

    payment = await client.create_recurring_payment(
        {
            "email": customer.email,
            "contact": customer.phone,
            "amount": case.original_amount,
            "currency": case.currency,
            "order_id": order["id"],
            "customer_id": customer.razorpay_customer_id,
            "token": token["id"],
            "recurring": True,
            "description": f"Retrying your {settings.company_name} subscription payment",
            "notes": {"recovery_case_id": str(case.id)},
        }
    )

    await log_action(
        session,
        case,
        ActionType.MANDATE_RETRIED,
        {
            "razorpay_payment_id": payment.get("razorpay_payment_id"),
            "razorpay_order_id": order["id"],
            "token_id": token["id"],
            "amount": case.original_amount,
        },
    )
    logger.info("re-charged mandate %s for case %s", token["id"], case.id)
    return True


async def _log_failure(session: AsyncSession, case: RecoveryCase, reason: str) -> None:
    logger.info("mandate retry skipped for case %s: %s", case.id, reason)
    await log_action(
        session, case, ActionType.MANDATE_RETRIED, {"charged": False, "reason": reason}
    )


async def send_mandate_link(
    session: AsyncSession,
    case: RecoveryCase,
    customer: Customer,
    client: RazorpayClient,
    *,
    settings: Settings,
    sms=None,
) -> str | None:
    """Send the customer the link that re-authorises their mandate.

    A payment link settles what is owed. It does not bring a revoked mandate
    back, so the next cycle fails in exactly the same way -- which is the whole
    reason ``CUSTOMER_INSTRUMENT`` earns a call rather than a link. Until now
    the agent could diagnose that and then had no tool to act on it: asked for
    the mandate link on a real call, the only thing it could send was the
    payment link again.

    The link is the subscription's own ``short_url``, fetched live rather than
    stored: it is Razorpay's hosted authorisation page and they own its
    lifetime.

    Delivery is ours. Razorpay notifies for payment links and has no equivalent
    for subscriptions, so this goes out over the same Twilio number that placed
    the call.

    Returns the URL when it was sent, None otherwise. The agent is instructed
    never to claim it sent a link it did not, so None must stay distinguishable
    from success.
    """
    if not case.razorpay_subscription_id:
        await _log_failure(session, case, "no subscription to re-authorise")
        return None
    if not customer.phone or customer.phone_is_wrong:
        await _log_failure(session, case, "no usable number to send a mandate link to")
        return None

    subscription = await client.fetch_subscription(case.razorpay_subscription_id)
    url = subscription.get("short_url")
    if not url:
        await _log_failure(session, case, "subscription carries no authorisation link")
        return None

    body = (
        f"{settings.company_name}: aapka auto-pay mandate dobara set karne ke liye "
        f"is link par jaayein - {url}"
    )
    sender = sms or _default_sms
    sid = await sender(customer.phone, body)
    if sid is None:
        await _log_failure(session, case, "could not send the mandate link")
        return None

    await log_action(
        session,
        case,
        ActionType.MANDATE_LINK_SENT,
        {
            "short_url": url,
            "razorpay_subscription_id": case.razorpay_subscription_id,
            "subscription_status": subscription.get("status"),
            "message_sid": sid,
        },
    )
    logger.info("sent mandate link for case %s", case.id)
    return url


async def _default_sms(to: str, body: str) -> str | None:
    """Imported lazily: the telephony module pulls in Twilio settings that the
    webhook service has no reason to load.

    The module function, not the channel. TwilioChannel refuses to construct
    without a stream URL, which a service that only receives calls does not
    have -- and sending a text does not need one.
    """
    from app.voice.telephony import send_sms

    return await send_sms(to, body)
