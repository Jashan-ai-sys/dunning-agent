"""Two-stage webhook intake.

Stage 1 (synchronous, in the request): persist the verified envelope. This is a
single INSERT, so we can return 200 well inside Razorpay's timeout.

Stage 2 (background): dispatch to a handler. If it crashes, the envelope is
already durable and ``processed_at IS NULL`` leaves it on the replay queue --
Razorpay's own at-least-once redelivery is a backstop, not the only safety net.

Stage 3 (eventually): give up. A queue that only ever retries does not become
resilient, it jams. ``replay_unprocessed`` reads the oldest unprocessed
envelopes first, so one that fails deterministically sits at the front of every
sweep; once there are more of those than the sweep's limit, nothing newer is
replayed again. And because ``record_event`` dedupes on the event id, Razorpay
redelivering a poison event changes nothing -- the sweep is the only retry path
there is. So a failure is either retried or dead-lettered, decided by whether
it could plausibly succeed next time.
"""

import logging
import traceback
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app import cache
from app.config import Settings, get_settings
from app.db import SessionLocal
from app.models import WebhookEvent
from app.razorpay.client import RazorpayClient, RazorpayError
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
    # Ask the cache first. Razorpay redelivers, and our own sweep re-dispatches,
    # so duplicates are ordinary traffic rather than an edge case -- and every
    # one of them currently costs an INSERT that is guaranteed to fail.
    #
    # The cache can only ever say "definitely seen". A miss falls through to
    # the unique constraint, which is what actually guarantees uniqueness; if
    # Redis is down or lying, the database still refuses the duplicate.
    settings = get_settings()
    if await cache.seen(
        f"webhook:{razorpay_event_id}", settings.redis_event_ttl_seconds
    ):
        logger.info("event %s already seen (cache)", razorpay_event_id)
        return None

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
    except Exception:
        # The claim is only honest if it is released when the write it guarded
        # did not happen. Leaving it set would make a retry of a *failed*
        # insert look like a duplicate and drop the event entirely.
        await cache.forget(f"webhook:{razorpay_event_id}")
        raise
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
        except Exception as exc:  # noqa: BLE001 - classified below, never swallowed
            logger.exception("handler for %s failed on event %s", event.event, event.id)
            await session.rollback()
            await _record_failure(
                webhook_event_pk,
                traceback.format_exc(limit=5),
                retryable=is_retryable(exc),
            )


def is_retryable(exc: BaseException) -> bool:
    """Could this failure plausibly succeed on another attempt?

    A 4xx from Razorpay is a statement about the request, not about the moment;
    replaying it produces the identical 4xx forever. Rate limiting and request
    timeouts are the exceptions -- 4xx by number, transient in fact. Anything
    we cannot classify counts as retryable, because dead-lettering a
    recoverable payment event is the worse mistake of the two.
    """
    if isinstance(exc, RazorpayError):
        if exc.status_code in (408, 429):
            return True
        return not 400 <= exc.status_code < 500
    return True


async def replay_unprocessed(limit: int = 50, older_than_seconds: int = 60) -> int:
    """Reprocess envelopes whose handler never completed.

    Covers the case Razorpay's own redelivery does not: we returned 200, so it
    considers the event delivered, but our handler then died (bad API response,
    process restart mid-task). The age cutoff keeps this from racing the
    background task that is still legitimately in flight.

    Dead-lettered envelopes are excluded. Without that they are the oldest
    unprocessed rows in every sweep and, past ``limit`` of them, starve every
    newer event out of the queue permanently.

    Returns the number of events re-dispatched.
    """
    cutoff = utcnow() - timedelta(seconds=older_than_seconds)
    async with SessionLocal() as session:
        result = await session.execute(
            select(WebhookEvent.id)
            .where(WebhookEvent.processed_at.is_(None))
            .where(WebhookEvent.dead_at.is_(None))
            .where(WebhookEvent.created_at < cutoff)
            .order_by(WebhookEvent.id)
            .limit(limit)
        )
        event_ids = list(result.scalars())

        buried = await session.execute(
            select(func.count(WebhookEvent.id)).where(WebhookEvent.dead_at.isnot(None))
        )
        dead_letters = buried.scalar_one()

    for event_id in event_ids:
        await process_event(event_id)
    if event_ids:
        logger.info("replayed %d unprocessed webhook events", len(event_ids))
    if dead_letters:
        # Loud on purpose: each one is an event we accepted from Razorpay and
        # never acted on, and nothing will retry it without a human.
        logger.warning("%d webhook events are dead-lettered and need review", dead_letters)
    return len(event_ids)


async def requeue_dead(limit: int = 50) -> int:
    """Put dead-lettered envelopes back on the queue.

    The other half of a dead-letter queue, and the reason burying an event is
    not the same as dropping it: once the bug that killed them is fixed the
    events are still here and still worth processing. The attempt count is
    cleared too, so a requeued event gets a full budget rather than dying again
    on its first failure.
    """
    async with SessionLocal() as session:
        result = await session.execute(
            select(WebhookEvent)
            .where(WebhookEvent.dead_at.isnot(None))
            .order_by(WebhookEvent.id)
            .limit(limit)
        )
        events = list(result.scalars())
        for event in events:
            event.dead_at = None
            event.attempt_count = 0
        await session.commit()

    if events:
        logger.info("requeued %d dead-lettered webhook events", len(events))
    return len(events)


async def _record_failure(
    webhook_event_pk: int,
    error: str,
    *,
    retryable: bool,
    settings: Settings | None = None,
) -> None:
    """Record the error on a clean session, since the original was rolled back.

    Also decides whether this envelope is finished. A failure that cannot
    succeed on a retry is buried immediately rather than after N identical
    attempts: there is nothing to learn from repeating it.
    """
    settings = settings or get_settings()
    async with SessionLocal() as session:
        event = await session.get(WebhookEvent, webhook_event_pk)
        if event is None:
            return
        event.attempt_count += 1
        event.processing_error = error
        if not retryable or event.attempt_count >= settings.webhook_max_attempts:
            event.dead_at = utcnow()
            logger.error(
                "webhook event %s (%s) dead-lettered after %d attempt(s); retryable=%s",
                event.id,
                event.event,
                event.attempt_count,
                retryable,
            )
        await session.commit()
