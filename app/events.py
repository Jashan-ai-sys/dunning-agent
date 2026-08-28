"""Push delivery for webhook events.

The service has always worked on a five-minute cron: the handler stores the
envelope, and a sweep picks it up on the next tick. That is up to 300 seconds
before anything acts on a failed payment. Nothing about dunning needs
sub-minute response, but nothing about it wants five minutes either.

So the envelope is also announced on Pub/Sub, and a push subscription calls us
back within about a second.

Three things follow from Postgres remaining the source of truth.

**The message carries an id, not a payload.** The envelope is already durable
and already verified; shipping the body again would put the same fact in two
places and invite them to disagree.

**Publishing is best effort.** A failure is logged and swallowed, because the
cron sweep still finds the row. Push is an accelerator on top of a mechanism
that already worked; making the request fail when the accelerator does would
trade a slow path for no path.

**Delivery is at-least-once, and that is fine.** ``process_event`` already
returns early on an envelope with ``processed_at`` set, so a redelivery
racing the sweep is a no-op rather than a double dispatch.
"""

import logging

from app.config import get_settings

logger = logging.getLogger(__name__)

_publisher = None
_unavailable = False


def _client():
    """The shared publisher, or None when no topic is configured."""
    global _publisher, _unavailable
    if _unavailable:
        return None
    if _publisher is not None:
        return _publisher
    if not get_settings().pubsub_topic.strip():
        _unavailable = True
        return None
    try:
        from google.cloud import pubsub_v1

        _publisher = pubsub_v1.PublisherClient()
    except Exception:  # noqa: BLE001 - the cron sweep is the fallback
        logger.exception("could not build the Pub/Sub publisher; cron only")
        _unavailable = True
        return None
    return _publisher


def reset_for_tests() -> None:
    global _publisher, _unavailable
    _publisher, _unavailable = None, False


async def announce_event(webhook_event_pk: int, event_name: str) -> bool:
    """Tell any subscriber that an envelope is ready to process.

    Called after the row is committed, never before: a subscriber that arrives
    first would look for a row that does not exist yet and mark a perfectly
    good event as missing.

    Returns whether it was published. False is not an error -- it means this
    event will be handled by the sweep instead, a few minutes later.
    """
    client = _client()
    if client is None:
        return False

    settings = get_settings()
    try:
        future = client.publish(
            settings.pubsub_topic,
            str(webhook_event_pk).encode("utf-8"),
            event=event_name,
        )
        # Publishing is a background thread in this client; resolving the
        # future here keeps "published" honest rather than optimistic.
        future.result(timeout=settings.pubsub_publish_timeout_seconds)
    except Exception:  # noqa: BLE001 - the sweep still has it
        logger.warning(
            "could not announce webhook event %s; leaving it to the sweep",
            webhook_event_pk,
            exc_info=True,
        )
        return False

    logger.info("announced webhook event %s (%s)", webhook_event_pk, event_name)
    return True
