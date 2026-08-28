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
# (error_code, error_source, error_reason, description, weight). The source and
# reason are what app.diagnosis reads, so a seeded batch exercises the real
# root-cause routing rather than falling through to "unknown".
FAILURE_MODES = [
    ("BAD_REQUEST_ERROR", "issuer", "insufficient_funds",
     "Your card has insufficient funds.", 34),
    ("GATEWAY_ERROR", "gateway", None,
     "The bank could not process the request.", 22),
    ("BAD_REQUEST_ERROR", "customer", "card_expired",
     "Your card has expired.", 14),
    ("BAD_REQUEST_ERROR", "customer", "mandate_revoked",
     "The payment mandate was revoked by the customer.", 12),
    # A reason the diagnosis table has never seen: it must degrade to the
    # source rather than to "unknown".
    ("BAD_REQUEST_ERROR", "issuer", "transaction_limit_exceeded",
     "Transaction limit exceeded for this card.", 10),
    ("SERVER_ERROR", "bank", None,
     "The issuing bank is temporarily unavailable.", 8),
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

# Of those who asked for a link on a call, how many actually pay it.
LINK_CONVERSION = 0.62

# Of those who were sent a link without being spoken to, how many pay it. Much
# lower, which is the whole reason the ladder escalates to a call at all.
COLD_LINK_CONVERSION = 0.18


class SeedRazorpay:
    """Stands in for the payment-link API during a simulated batch.

    The cheap intervention talks to Razorpay, so without this a seeded run
    would create *real* payment links on the merchant account against
    synthetic debts. Every id it returns is prefixed ``plink_seed_`` so a link
    from a simulation can never be mistaken for one a customer was sent.
    """

    def __init__(self) -> None:
        self.created: list[dict] = []

    async def create_payment_link(self, payload: dict) -> dict:
        self.created.append(payload)
        n = len(self.created)
        return {
            "id": f"plink_seed_{n}",
            "short_url": f"https://example.invalid/seed/{n}",
            "reference_id": payload["reference_id"],
        }

    async def notify_payment_link(self, link_id: str, medium: str) -> dict:
        return {"success": True}


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
        code, error_source, reason_code, description, _ = rng.choices(
            FAILURE_MODES, weights=weights, k=1
        )[0]
        case = RecoveryCase(
            razorpay_payment_id=f"pay_seed_{i}",
            razorpay_subscription_id=subscription_id,
            razorpay_customer_id=customer_id,
            original_amount=amount,
            failure_code=code,
            failure_reason=description,
            failure_source=error_source,
            failure_reason_code=reason_code,
            status=CaseStatus.OPEN,
            source=SEED,
            # Spread over the past week so the batch looks like a real backlog.
            created_at=now - timedelta(hours=rng.randint(1, 168)),
            # Razorpay has already given up on some of them. The rest are
            # still inside its retry sequence and the policy defers to it --
            # bounded by bank_retry_grace_hours, so they are worked eventually
            # either way; this just makes the batch show both paths.
            halted_at=now - timedelta(hours=rng.randint(1, 48)) if rng.random() < 0.45
            else None,
        )
        session.add(case)
        await session.flush()
        await log_action(
            session, case, ActionType.CASE_OPENED,
            {
                "source": SEED,
                "error_code": code,
                "error_source": error_source,
                "error_reason": reason_code,
                "amount": amount,
            },
        )

    await session.commit()
    return count


async def _settle_unprompted_links(session, rng: random.Random, now) -> int:
    """Pay off some of the cases holding a link that nobody has called about.

    Deliberately a lower rate than ``LINK_CONVERSION``: a link somebody asked
    for on a call converts far better than one that arrived unannounced.
    """
    cases = (
        await session.execute(
            select(RecoveryCase)
            .where(RecoveryCase.source == SEED)
            .where(RecoveryCase.status == CaseStatus.IN_PROGRESS)
            .where(RecoveryCase.payment_link_id.isnot(None))
            .where(RecoveryCase.attempt_count == 1)
        )
    ).scalars().all()

    paid = 0
    for case in cases:
        if rng.random() >= COLD_LINK_CONVERSION:
            continue
        case.status = CaseStatus.RECOVERED
        case.recovered_payment_id = f"pay_seed_link_{case.id}"
        case.recovered_amount = case.original_amount
        case.recovered_at = now
        await log_action(
            session, case, ActionType.PAYMENT_CAPTURED,
            {"source": SEED, "amount": case.original_amount, "simulated": True,
             "via": "payment_link"},
        )
        paid += 1
    return paid


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
    # Run the window so it *ends* today rather than starting today. Simulating
    # forward put every promise deadline in the future, which meant no promise
    # could ever be scored as broken and the report read 100% kept -- a number
    # that says nothing. Ending today lets the early days' promises resolve.
    base = (utcnow() - timedelta(days=days - 1)).astimezone(ZoneInfo("Asia/Kolkata")).replace(
        hour=14, minute=0, second=0, microsecond=0
    )

    for day in range(days):
        now = base + timedelta(days=day)
        channel = SeedChannel(rng)
        result = await run_once(session, channel, now=now, client=SeedRazorpay())
        print(f"  day {day + 1}: {result.as_dict()}")

        # Some people simply pay the link. Modelling this matters: the ladder
        # sends a link before it spends a call, so a simulation that only ever
        # converts after a conversation would make the cheap intervention look
        # worthless when it is the one doing most of the work.
        await _settle_unprompted_links(session, rng, now)

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
                customer=customer, client=None, now=now,
            )

            # Those who asked for a link: some of them actually pay it. This is the
            # only place money "moves", and it only ever touches source='seed' rows.
            if intent is CallIntent.RETRY_NOW and rng.random() < LINK_CONVERSION:
                case.status = CaseStatus.RECOVERED
                case.recovered_payment_id = f"pay_seed_recovered_{case.id}"
                case.recovered_amount = case.original_amount
                # Simulated time, not real time. Paying "now" against a promise
                # stamped with the same clock is what makes the kept/broken
                # split mean anything; the ones who never pay drift past their
                # deadline as the days advance and show up as broken.
                case.recovered_at = min(
                    now + timedelta(hours=rng.randint(1, 60)), utcnow()
                )
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
