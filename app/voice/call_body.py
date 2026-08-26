"""What the agent is told about the call it is about to make.

One shape, two transports. LiveKit carries it as JSON job metadata and Pipecat
as the runner's request body, but the keys are identical -- so a case dispatched
either way produces the same conversation, and a bug in how the amount is spoken
is a bug in one place.

Kept out of ``dispatch.py`` on purpose: that module imports the LiveKit SDK at
module level, and the Pipecat path must not drag it in to learn a customer's
name.
"""

import logging
from typing import Any

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models import Customer, RecoveryCase
from app.voice.reasons import spoken_reason
from app.voice.routes import suggested_route
from app.voice.spoken import spoken_amount

logger = logging.getLogger(__name__)


def call_body(case: RecoveryCase, customer: Customer, *, company_name: str) -> dict[str, Any]:
    """Everything the agent needs to open the call, and nothing it does not.

    Deliberately no card details, no payment link and no full history: the agent
    cannot leak what it was never given.
    """
    return {
        "recovery_case_id": case.id,
        "phone": customer.phone,
        "customer_name": customer.name or "there",
        "preferred_language": customer.preferred_language,
        "company_name": company_name,
        # Already in spoken form -- "5 लाख रुपये", not "500000". Rendering it
        # here rather than in the agent keeps a raw numeral and a Latin "Rs"
        # from ever reaching a Hindi voice.
        "amount_spoken": spoken_amount(case.original_amount, customer.preferred_language),
        "amount_paise": case.original_amount,
        # Phrased in the call's language, not Razorpay's. The raw string is
        # English and would land inside a Devanagari sentence.
        "failure_reason": spoken_reason(case.failure_reason, customer.preferred_language),
        # Razorpay has given up retrying this one, so the subscription itself is
        # at risk rather than just this charge. The agent needs to know: the
        # right thing to say changes, and so does the urgency.
        "subscription_halted": case.halted_at is not None,
        # Decided by rule, not by the model: the remedy for a given failure code
        # is the same every time, and it is the one thing the agent tells the
        # customer to actually do.
        "suggested_route": suggested_route(
            case.failure_reason, halted=case.halted_at is not None
        ),
    }


async def load_call_body(recovery_case_id: int) -> dict[str, Any] | None:
    """Build the body for a case straight from the database.

    Returns None rather than raising, for *any* reason it cannot produce a body
    -- unknown case, missing customer, or a database that is simply not there.
    The caller falls back to a sample and the agent still talks.

    That last case is the one that matters. This runs before the transport
    exists, so an exception here does not degrade the call, it prevents it:
    the caller never gets a session, and a customer who picked up hears
    silence. A recovery agent that cannot read its own database has lost the
    ability to *personalise* a call, not the ability to make one.
    """
    try:
        async with SessionLocal() as session:
            case = await session.get(RecoveryCase, recovery_case_id)
            if case is None:
                logger.warning("no recovery case %s", recovery_case_id)
                return None

            customer = None
            if case.razorpay_customer_id:
                customer = (
                    await session.execute(
                        select(Customer).where(
                            Customer.razorpay_customer_id == case.razorpay_customer_id
                        )
                    )
                ).scalar_one_or_none()

            if customer is None:
                logger.warning("case %s has no customer record", recovery_case_id)
                return None

            return call_body(case, customer, company_name=get_settings().company_name)
    except Exception:  # noqa: BLE001 - never let the database silence the agent
        logger.exception("could not load recovery case %s", recovery_case_id)
        return None


__all__ = ["call_body", "load_call_body"]
