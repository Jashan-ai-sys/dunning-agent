"""Background worker.

Runs two jobs on a fixed interval:

1. The replay sweep, which re-dispatches webhook envelopes whose handler died.
2. An orchestrator tick, which works the open recovery cases.

Run it alongside the API:  uv run python -m app.worker
"""

import asyncio
import logging

from app.channels import LoggingChannel
from app.config import get_settings
from app.db import SessionLocal, engine
from app.orchestrator import run_once
from app.webhooks.processor import replay_unprocessed

logger = logging.getLogger(__name__)


async def tick() -> None:
    """One full pass. Errors are logged, never fatal -- the loop must survive a
    bad batch or a transient database blip."""
    try:
        await replay_unprocessed()
    except Exception:
        logger.exception("replay sweep failed")

    try:
        async with SessionLocal() as session:
            result = await run_once(session, LoggingChannel())
        if result.considered:
            logger.info("orchestrator tick %s", result.as_dict())
    except Exception:
        logger.exception("orchestrator tick failed")


async def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("worker started, interval %ss", settings.worker_interval_seconds)
    try:
        while True:
            await tick()
            await asyncio.sleep(settings.worker_interval_seconds)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("worker stopped")
