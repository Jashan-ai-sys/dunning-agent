"""Seed a synthetic batch of failed subscription charges, then drive it through
the real recovery pipeline.

    uv run python -m scripts.seed_batch --count 50
    uv run python -m scripts.seed_batch --count 50 --simulate
    uv run python -m scripts.seed_batch --clear

Every row it creates is stamped ``source='seed'``. That flag is in the schema,
not a naming convention, so simulated recovery can never be reported as real
money -- ``app.report`` separates the two and refuses to blend them.

What is real here and what is not:

* real -- the policy engine, the orchestrator, the stopping rules, the contact
  window, the attempt caps and the audit trail. Seeded cases go through exactly
  the same code as a live webhook would.
* simulated -- the failures themselves, and (with ``--simulate``) the customer's
  response to the call. No money moves and no call is placed.
"""

import argparse
import asyncio
import random
import sys
from datetime import timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import delete, select

from app.channels import ContactResult
from app.constants import ActionType, CallStatus, CaseStatus
from app.db import SessionLocal, engine
from app.metrics import compute_metrics, format_report, rupees
from app.models import Customer, RecoveryCase, Subscription, VoiceCall
from app.orchestrator import run_once
from app.store import log_action, utcnow
from app.voice.intents import CallIntent
from app.voice.outcomes import CallResult, apply_call_result

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = "seed"

# Plans a real Indian SaaS/OTT/D2C mix would bill, in paise.
PLAN_AMOUNTS = [4_900, 9_900, 19_900, 29_900, 49_900, 99_900, 249_900, 4_000]

# Failure reasons weighted roughly as Razorpay reports them: most recurring
# failures are bank-side, not the customer's fault.
FAILURE_MODES = [
    ("BAD_REQUEST_ERROR", "Your card has insufficient funds.", 34),
    ("GATEWAY_ERROR", "The bank could not process the request.", 22),
    ("BAD_REQUEST_ERROR", "Your card has expired.", 14),
    ("BAD_REQUEST_ERROR", "The payment mandate was revoked by the customer.", 12),
    ("BAD_REQUEST_ERROR", "Transaction limit exceeded for this card.", 10),
    ("SERVER_ERROR", "The issuing bank is temporarily unavailable.", 8),
]

NAMES = [
    "Asha Rao", "Vikram Nair", "Priya Sharma", "Rahul Mehta", "Sneha Iyer",
    "Arjun Reddy", "Kavya Menon", "Imran Sheikh", "Neha Gupta", "Rohit Das",
    "Ananya Bose", "Karthik Subramanian", "Farhan Qureshi", "Meera Joshi",
    "Sanjay Patel", "Divya Krishnan", "Aditya Kulkarni", "Ritu Chawla",
]
LANGUAGES = ["hinglish"] * 6 + ["hi"] * 3 + ["en"] * 2

# How customers respond when reached. Deliberately not flattering: most calls
# do not end in an immediate payment.
INTENT_MIX = (
    [CallIntent.RETRY_NOW] * 30
    + [CallIntent.RETRY_LATER] * 20
    + [CallIntent.NO_ANSWER] * 22
    + [CallIntent.DECLINED] * 12
    + [CallIntent.UNCLEAR] * 8
    + [CallIntent.WRONG_NUMBER] * 5
    + [CallIntent.DISPUTE] * 3
)

# Of those who asked for a link, how many actually pay it.
LINK_CONVERSION = 0.62


class SeedChannel:
    """Stands in for telephony during a simulated batch.

    Named ``seed`` in the audit trail so no run of this can be mistaken for a
    real call.
    """

    name = "seed"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.contacted: list[int] = []

    async def initiate(self, case, customer) -> ContactResult:
        self.contacted.append(case.id)
        return ContactResult(channel=self.name, reference=f"seed-room-{case.id}",
                             detail={"placed": False, "simulated": True})


async def clear(session) -> int:
    result = await session.execute(
        delete(RecoveryCase).where(RecoveryCase.source == SEED).returning(RecoveryCase.id)
    )
    removed = len(list(result.scalars()))
    await session.execute(delete(Customer).where(Customer.razorpay_customer_id.like("cust_seed_%")))
    await session.execute(
        delete(Subscription).where(Subscription.razorpay_subscription_id.like("sub_seed_%"))
    )
    await session.commit()
    return removed


async def seed(session, count: int, rng: random.Random) -> int:
    now = utcnow()
    weights = [w for *_, w in FAILURE_MODES]

    for i in range(count):
        customer_id = f"cust_seed_{i}"
        session.add(
            Customer(
                razorpay_customer_id=customer_id,
                name=rng.choice(NAMES),
                # Reserved test range: these can never route to a real person.
                phone=f"+9199999{i:05d}",
                email=f"seed{i}@example.com",
                preferred_language=rng.choice(LANGUAGES),
            )
        )
        subscription_id = f"sub_seed_{i}"
        amount = rng.choice(PLAN_AMOUNTS)
        session.add(
            Subscription(
                razorpay_subscription_id=subscription_id,
                razorpay_customer_id=customer_id,
                plan_amount=amount,
                status="pending",
            )
        )
        code, description, _ = rng.choices(FAILURE_MODES, weights=weights, k=1)[0]
        case = RecoveryCase(
            razorpay_payment_id=f"pay_seed_{i}",
            razorpay_subscription_id=subscription_id,
            razorpay_customer_id=customer_id,
            original_amount=amount,
            failure_code=code,
            failure_reason=description,
            status=CaseStatus.OPEN,
            source=SEED,
            # Spread over the past week so the batch looks like a real backlog.
            created_at=now - timedelta(hours=rng.randint(1, 168)),
        )
        session.add(case)
        await session.flush()
        await log_action(
            session, case, ActionType.CASE_OPENED,
            {"source": SEED, "error_code": code, "amount": amount},
        )

    await session.commit()
    return count


async def simulate(session, rng: random.Random, days: int) -> dict[str, int]:
    """Run the real orchestrator once per simulated day, then apply a simulated
    customer response to whoever it decided to call.

    Ticking across days rather than once is what lets the bounded-workflow rules
    actually show: the 24h backoff spaces the attempts out, and the attempt cap
    closes cases that never convert. Each tick is taken at 14:00 IST, inside the
    permitted calling window -- run at 2am the policy would correctly refuse to
    call anyone and the batch would never move.
    """
    tally: dict[str, int] = {}
    base = utcnow().astimezone(ZoneInfo("Asia/Kolkata")).replace(
        hour=14, minute=0, second=0, microsecond=0
    )

    for day in range(days):
        now = base + timedelta(days=day)
        channel = SeedChannel(rng)
        result = await run_once(session, channel, now=now)
        print(f"  day {day + 1}: {result.as_dict()}")

        cases = (
            await session.execute(
                select(RecoveryCase)
                .where(RecoveryCase.source == SEED)
                .where(RecoveryCase.id.in_(channel.contacted))
            )
        ).scalars().all()

        for case in cases:
            customer = (
                await session.execute(
                    select(Customer).where(
                        Customer.razorpay_customer_id == case.razorpay_customer_id
                    )
                )
            ).scalar_one_or_none()

            intent = rng.choice(INTENT_MIX)
            tally[intent] = tally.get(intent, 0) + 1

            call = VoiceCall(
                recovery_case_id=case.id,
                provider=SEED,
                room_name=f"seed-room-{case.id}",
                status=CallStatus.COMPLETED if intent is not CallIntent.NO_ANSWER
                else CallStatus.NO_ANSWER,
                duration_seconds=rng.randint(18, 95),
            )
            session.add(call)
            await session.flush()
            await apply_call_result(
                session, case, call,
                CallResult(intent=intent, final_node_id=str(intent),
                           duration_seconds=call.duration_seconds),
                customer=customer, client=None,
            )

            # Those who asked for a link: some of them actually pay it. This is the
            # only place money "moves", and it only ever touches source='seed' rows.
            if intent is CallIntent.RETRY_NOW and rng.random() < LINK_CONVERSION:
                case.status = CaseStatus.RECOVERED
                case.recovered_payment_id = f"pay_seed_recovered_{case.id}"
                case.recovered_amount = case.original_amount
                case.recovered_at = utcnow()
                await log_action(
                    session, case, ActionType.PAYMENT_CAPTURED,
                    {"source": SEED, "amount": case.original_amount, "simulated": True},
                )

    await session.commit()
    return tally


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--simulate", action="store_true",
                        help="run the orchestrator and apply simulated call outcomes")
    parser.add_argument("--days", type=int, default=4,
                        help="simulated days of orchestrator ticks")
    parser.add_argument("--clear", action="store_true", help="delete seeded rows and exit")
    parser.add_argument("--seed", type=int, default=7, help="rng seed, for reproducible demos")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    try:
        async with SessionLocal() as session:
            if args.clear:
                print(f"removed {await clear(session)} seeded cases")
                return

            await clear(session)
            created = await seed(session, args.count, rng)
            print(f"seeded {created} failed charges (source='{SEED}')")

            if args.simulate:
                print(f"simulating {args.days} days of ticks:")
                tally = await simulate(session, rng, args.days)
                print("simulated call outcomes:")
                for intent, n in sorted(tally.items()):
                    print(f"  {intent:<14} {n:>3}")

            print()
            metrics = await compute_metrics(session, source=SEED)
            print(format_report(metrics))
            print()
            print(f"NOTE: {rupees(metrics.amount_recovered)} of simulated recovery. "
                  "No money moved and no call was placed.")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
