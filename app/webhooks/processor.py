"""Two-stage webhook intake.

Stage 1 (synchronous, in the request): persist the verified envelope. This is a
single INSERT, so we can return 200 well inside Razorpay's timeout.

Stage 2 (background): dispatch to a handler. If it crashes, the envelope is
already durable and ``processed_at IS NULL`` leaves it on the replay queue --
Razorpay's own at-least-once redelivery is a backstop, not the only safety net.
"""

import logging
import traceback
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionLocal
from app.models import WebhookEvent
from app.razorpay.client import RazorpayClient
from app.store import utcnow
from app.webhooks.handlers import EVENT_HANDLERS

logger = logging.getLogger(__name__)


async def record_event(
    session: AsyncSession,
    *,
    razorpay_event_id: str,
    event_name: str,
    payload: dict[str, Any],
) -> WebhookEvent | None:
    """Persist the envelope. Returns None if this event was already recorded."""
    event = WebhookEvent(
        razorpay_event_id=razorpay_event_id,
        event=event_name,
        payload=payload,
        signature_verified=True,
    )
    session.add(event)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return None
    await session.refresh(event)
    return event


async def process_event(webhook_event_pk: int) -> None:
    """Dispatch one recorded event to its handler.

    Runs outside the request, so it owns its session.
    """
    async with SessionLocal() as session:
        event = await session.get(WebhookEvent, webhook_event_pk)
        if event is None:
            logger.error("webhook event %s vanished before processing", webhook_event_pk)
            return
        if event.processed_at is not None:
            return

        handler = EVENT_HANDLERS.get(event.event)
        if handler is None:
            # Subscribed to an event we do not act on yet. Mark it done so it
            # does not sit on the replay queue forever.
            event.processed_at = utcnow()
            await session.commit()
            return

        # Handlers read the event id for the audit trail; it lives in a header,
        # not the body, so splice it in.
        payload = {**event.payload, "razorpay_event_id": event.razorpay_event_id}
        try:
            await handler(session, payload, RazorpayClient())
            event.processed_at = utcnow()
            event.processing_error = None
            await session.commit()
        except Exception:
            logger.exception("handler for %s failed on event %s", event.event, event.id)
            await session.rollback()
            await _record_failure(webhook_event_pk, traceback.format_exc(limit=5))


async def _record_failure(webhook_event_pk: int, error: str) -> None:
    """Record the error on a clean session, since the original was rolled back."""
    async with SessionLocal() as session:
        event = await session.get(WebhookEvent, webhook_event_pk)
        if event is None:
            return
        event.processing_error = error
        await session.commit()
