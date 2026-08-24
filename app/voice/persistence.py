"""Write what happened on a call back into the database.

The agent process is separate from the API, so it owns its own sessions. This
module is the only place it touches Postgres, which keeps ``agent.py`` about
audio and nothing else.

Two rules, both load-bearing:

1. **A database failure must never break a live call.** Every entry point
   swallows its exceptions. Losing the record of a call is bad; dropping the
   customer mid-sentence because a connection pool hiccuped is worse.
2. **A demo run must not write anything.** When the job metadata carries no
   ``recovery_case_id`` -- the LiveKit console, or ``demo_call`` without
   ``--case`` -- every function here is a no-op. Nothing invents a case.
"""

import logging

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Customer, RecoveryCase, VoiceCall
from app.razorpay.client import RazorpayClient
from app.store import utcnow
from app.voice.outcomes import CallResult, apply_call_result

logger = logging.getLogger(__name__)


async def open_call_record(
    *, recovery_case_id: int | None, room_name: str, dialled_number: str | None = None
) -> int | None:
    """Record that a call started. Returns the voice_calls id, or None.

    Written at the start rather than the end so an abandoned or crashed call
    still leaves a trace -- a call that vanished is itself a finding.
    """
    if not recovery_case_id:
        logger.info("no recovery_case_id in metadata; not recording this call")
        return None

    try:
        async with SessionLocal() as session:
            case = await session.get(RecoveryCase, int(recovery_case_id))
            if case is None:
                logger.warning("recovery case %s does not exist", recovery_case_id)
                return None

            call = VoiceCall(
                recovery_case_id=case.id,
                provider="livekit",
                room_name=room_name,
                dialled_number=dialled_number,
                status="initiated",
            )
            session.add(call)
            await session.commit()
            await session.refresh(call)
            logger.info("opened voice_call %s for case %s", call.id, case.id)
            return call.id
    except Exception:  # noqa: BLE001 - never break the call
        logger.exception("could not open a call record")
        return None


async def finalise_call(
    *,
    voice_call_id: int | None,
    recovery_case_id: int | None,
    result: CallResult,
    _client=None,
) -> None:
    """Apply the call's outcome to its case, and send a link if one is due.

    This is where the conversation finally touches money: ``apply_call_result``
    moves the case according to the detected intent and, for ``retry_now``,
    creates the Razorpay payment link.
    """
    if not voice_call_id or not recovery_case_id:
        logger.info(
            "call finished with intent %s (not persisted: demo run)", result.intent
        )
        return

    try:
        async with SessionLocal() as session:
            call = await session.get(VoiceCall, voice_call_id)
            case = await session.get(RecoveryCase, int(recovery_case_id))
            if call is None or case is None:
                logger.warning("call %s or case %s vanished", voice_call_id, recovery_case_id)
                return

            customer = None
            if case.razorpay_customer_id:
                customer = (
                    await session.execute(
                        select(Customer).where(
                            Customer.razorpay_customer_id == case.razorpay_customer_id
                        )
                    )
                ).scalar_one_or_none()

            await apply_call_result(
                session,
                case,
                call,
                result,
                customer=customer,
                # Only build a Razorpay client when a link is actually due, so a
                # call that ends in "declined" never touches the API. `_client`
                # is the test seam.
                client=(
                    _client
                    if _client is not None
                    else (RazorpayClient() if result.intent_sends_link else None)
                ),
            )
            await session.commit()
            logger.info(
                "case %s updated from call %s: intent=%s status=%s",
                case.id,
                call.id,
                result.intent,
                case.status,
            )
    except Exception:  # noqa: BLE001 - never break the call
        logger.exception("could not persist the call outcome")


async def duration_since(started_at) -> int | None:
    """Whole seconds since ``started_at``, or None if it was never set."""
    if started_at is None:
        return None
    return max(0, int((utcnow() - started_at).total_seconds()))
