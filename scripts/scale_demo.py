"""Show what the queue does when you add workers.

    uv run python -m scripts.scale_demo --cases 200 --workers 3
    uv run python -m scripts.scale_demo --cases 200 --workers 1   # the baseline

Seeds a batch of open recovery cases, runs N orchestrator ticks at once against
the real database, and reports the two numbers that matter: how long it took,
and whether anybody was contacted twice.

Nothing here is a mock. It is the real `run_once`, the real policy, the real
priority ordering and the real `FOR UPDATE SKIP LOCKED` claim. What is faked is
only the telephony -- no call is placed and no money moves -- and every case it
creates is stamped ``source='seed'``, which is a column and not a naming
convention, so simulated work can never be reported as recovered revenue.

The point it demonstrates is not throughput for its own sake. Adding workers is
easy; adding workers *without ringing the same customer twice* is the part that
needs the database to guarantee it. Run it with `--workers 1` first and then
with 3, and the interesting line is not the wall time -- it is that
`contacted twice` stays at zero.
"""

import argparse
import asyncio
import time

from sqlalchemy import delete, func, select

from app.channels import ContactResult
from app.config import get_settings
from app.constants import ActionType, CaseSource, CaseStatus
from app.db import SessionLocal, engine
from app.models import Customer, RecoveryAction, RecoveryCase
from app.orchestrator import run_once
from app.store import utcnow

PREFIX = "scale-demo"


class SilentChannel:
    """Counts contacts instead of placing them.

    A real Twilio call per case would make this a test of Twilio's rate limits
    rather than of the queue, and would cost money to run twice.
    """

    name = "scale-demo"

    def __init__(self) -> None:
        self.contacted: list[int] = []

    async def initiate(self, case: RecoveryCase, customer: Customer) -> ContactResult:
        self.contacted.append(case.id)
        return ContactResult(channel=self.name, reference=f"demo_{case.id}")


class SilentRazorpay:
    """The payment-link half of the ladder, without the API."""

    async def fetch_customer_tokens(self, customer_id: str) -> dict:
        return {"items": []}

    async def create_payment_link(self, payload: dict) -> dict:
        return {
            "id": f"plink_{payload.get('reference_id', 'x')}",
            "short_url": "https://rzp.io/demo",
            "reference_id": payload.get("reference_id"),
        }

    async def notify_payment_link(self, link_id: str, medium: str) -> dict:
        return {"success": True}


async def clear() -> None:
    """Remove everything a previous run left behind."""
    async with SessionLocal() as session:
        cases = (
            await session.execute(
                select(RecoveryCase.id).where(
                    RecoveryCase.razorpay_payment_id.like(f"{PREFIX}%")
                )
            )
        ).scalars().all()
        if cases:
            await session.execute(
                delete(RecoveryAction).where(RecoveryAction.recovery_case_id.in_(cases))
            )
        await session.execute(
            delete(RecoveryCase).where(RecoveryCase.razorpay_payment_id.like(f"{PREFIX}%"))
        )
        await session.execute(
            delete(Customer).where(Customer.razorpay_customer_id.like(f"{PREFIX}%"))
        )
        await session.commit()


async def seed(count: int) -> None:
    """One open case per customer, at varying amounts.

    Distinct customers because a shared one would be held by the contact
    cooldown -- correctly, but it would mean the demo measured the policy
    rather than the queue.

    `attempt_count=1` starts each case past the payment-link rung of the
    ladder, so every one reaches the contact step and a double-claim would show
    up as what it really is: a second phone call.
    """
    async with SessionLocal() as session:
        for i in range(count):
            session.add(
                Customer(
                    razorpay_customer_id=f"{PREFIX}-cust-{i}",
                    phone=f"+9198{i:08d}",
                    name=f"Demo Customer {i}",
                )
            )
            session.add(
                RecoveryCase(
                    razorpay_payment_id=f"{PREFIX}-pay-{i}",
                    razorpay_customer_id=f"{PREFIX}-cust-{i}",
                    razorpay_subscription_id=f"{PREFIX}-sub-{i}",
                    original_amount=10_000 + (i * 137) % 490_000,
                    status=CaseStatus.OPEN,
                    source=CaseSource.SEED,
                    attempt_count=1,
                    last_attempt_at=utcnow().replace(year=2020),
                )
            )
        await session.commit()


async def one_worker(batch: int) -> list[int]:
    """A single tick in its own session, which is what makes it its own worker."""
    channel = SilentChannel()
    async with SessionLocal() as session:
        await run_once(session, channel, client=SilentRazorpay(), settings=_settings(batch))
    return channel.contacted


def _settings(batch: int):
    settings = get_settings()
    return settings.model_copy(
        update={
            "worker_batch_size": batch,
            # The window is a real rule and the right one on a live call, but a
            # demo that only works between 9 and 9 is a demo that fails on
            # stage. Widened here and nowhere else -- this process never dials.
            "contact_window_start_hour": 0,
            "contact_window_end_hour": 24,
        }
    )


async def duplicates() -> list[tuple[int, int]]:
    """Cases with more than one contact recorded. The number that matters."""
    async with SessionLocal() as session:
        rows = await session.execute(
            select(RecoveryAction.recovery_case_id, func.count())
            .join(RecoveryCase, RecoveryCase.id == RecoveryAction.recovery_case_id)
            .where(RecoveryCase.razorpay_payment_id.like(f"{PREFIX}%"))
            .where(RecoveryAction.action_type == ActionType.VOICE_CALL)
            .group_by(RecoveryAction.recovery_case_id)
            .having(func.count() > 1)
        )
        return list(rows.all())


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--clear", action="store_true", help="clean up and exit")
    args = parser.parse_args()

    if args.clear:
        await clear()
        print("cleared")
        await engine.dispose()
        return

    await clear()
    await seed(args.cases)
    # Ceiling, not floor: with 200 cases and 3 workers, flooring gives a batch
    # of 66 and the last two cases are never claimed by anyone. A demo that
    # quietly drops work is worse than no demo.
    per_worker = max(1, -(-args.cases // args.workers))

    print(f"\n  {args.cases} open cases, {args.workers} worker(s), batch {per_worker} each")
    print("  " + "-" * 56)

    started = time.perf_counter()
    batches = await asyncio.gather(*(one_worker(per_worker) for _ in range(args.workers)))
    elapsed = time.perf_counter() - started

    contacted = [case_id for batch in batches for case_id in batch]
    dupes = await duplicates()

    for i, batch in enumerate(batches):
        print(f"  worker {i + 1}: {len(batch):>4} contacted")
    print("  " + "-" * 56)
    print(f"  total contacted   {len(contacted)}")
    print(f"  distinct cases    {len(set(contacted))}")
    print(f"  contacted twice   {len(dupes)}")
    print(f"  wall time         {elapsed:.2f}s")
    if contacted:
        print(f"  per case          {elapsed / len(contacted) * 1000:.0f}ms")

    print()
    if dupes:
        # Loud, because this is the failure the whole design exists to prevent.
        print("  FAIL: these cases were contacted more than once:")
        for case_id, n in dupes:
            print(f"    case {case_id}: {n} times")
    else:
        print(f"  OK: {args.workers} workers shared one queue, nobody contacted twice.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
