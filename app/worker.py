"""Background worker.

Runs two jobs on a fixed interval:

1. The replay sweep, which re-dispatches webhook envelopes whose handler died.
2. An orchestrator tick, which works the open recovery cases.

Run it alongside the API:  uv run python -m app.worker
"""

import argparse
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


async def main(once: bool = False) -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        if once:
            # Cloud Run Jobs invoke the container once per schedule tick, so the
            # loop lives in Cloud Scheduler rather than in this process.
            logger.info("worker running a single tick")
            await tick()
            return
        logger.info("worker started, interval %ss", settings.worker_interval_seconds)
        while True:
            await tick()
            await asyncio.sleep(settings.worker_interval_seconds)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="run one tick and exit (for Cloud Run Jobs / cron)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(once=args.once))
    except KeyboardInterrupt:
        logger.info("worker stopped")
