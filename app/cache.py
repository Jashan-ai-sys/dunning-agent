"""A shared cache, and what it is allowed to decide.

Redis is a *fast path*, never the authority. Postgres stays the source of
truth for every fact in this system, and losing Redis has to degrade
performance rather than correctness -- a cache that can silently change an
outcome is a second database with none of the guarantees.

That principle shows up concretely in each user:

* Webhook dedupe still relies on the unique constraint. Redis only saves the
  round trip when it already knows the answer; a miss falls through and the
  database decides.
* The contact cooldown still reads ``customers.last_contacted_at``. Redis is
  consulted first because with several workers on several machines the row
  lock only serialises within one transaction, not across the fleet.

Unconfigured is the normal case, not an error. Every call degrades to a miss
when ``REDIS_URL`` is empty, so a deployment without Redis behaves exactly as
this service did before it existed.
"""

import logging

from app.config import get_settings

logger = logging.getLogger(__name__)

#: Module-level so one pool is shared. None means "not yet built"; False means
#: "tried and unavailable", which is remembered so a dead Redis is not dialled
#: on every event.
_client: object | None = None
_unavailable = False


def _redis():
    """The shared client, or None when Redis is not configured or not up."""
    global _client, _unavailable
    if _unavailable:
        return None
    if _client is not None:
        return _client

    url = get_settings().redis_url.strip()
    if not url:
        _unavailable = True
        return None

    try:
        import redis.asyncio as redis_asyncio

        _client = redis_asyncio.from_url(
            url,
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
        )
    except Exception:  # noqa: BLE001 - an unreachable cache must not be fatal
        logger.exception("could not build the Redis client; running without a cache")
        _unavailable = True
        return None
    return _client


def reset_for_tests() -> None:
    """Drop the memoised client so a test can swap the configuration."""
    global _client, _unavailable
    _client, _unavailable = None, False


async def claim(key: str, ttl_seconds: int) -> bool:
    """Claim ``key`` for ``ttl_seconds``. True if we got it, False if taken.

    ``SET key 1 NX EX ttl`` -- atomic across every worker, which is the point:
    a Postgres row lock serialises within one transaction, not across a fleet.

    Returns True when Redis is unavailable. The caller must treat a claim as
    permission to *check* the durable rule, never as the rule itself, so
    failing open degrades to the behaviour we already had rather than
    silently skipping work.
    """
    client = _redis()
    if client is None:
        return True
    try:
        return bool(await client.set(key, "1", nx=True, ex=ttl_seconds))
    except Exception:  # noqa: BLE001 - never fail a call over the cache
        logger.warning("Redis claim failed for %s; falling through", key, exc_info=True)
        return True


async def seen(key: str, ttl_seconds: int) -> bool:
    """True if ``key`` was already recorded. Records it either way.

    The inverse of :func:`claim`, named for how the caller reads: a webhook
    asks "have I seen this event id", not "may I have this lock".
    """
    return not await claim(key, ttl_seconds)


async def forget(key: str) -> None:
    """Drop a key. Used when the durable write a claim guarded did not happen."""
    client = _redis()
    if client is None:
        return
    try:
        await client.delete(key)
    except Exception:  # noqa: BLE001
        logger.warning("Redis delete failed for %s", key, exc_info=True)


async def seen_only(key: str) -> bool:
    """True if ``key`` exists, without creating it.

    :func:`seen` records as it checks, which is right for dedupe -- the first
    caller should win. It is wrong for reading a cooldown another worker set:
    checking whether someone was recently called must not itself count as
    calling them.
    """
    client = _redis()
    if client is None:
        return False
    try:
        return bool(await client.exists(key))
    except Exception:  # noqa: BLE001 - an unreadable cache stops nothing
        logger.warning("Redis read failed for %s", key, exc_info=True)
        return False
