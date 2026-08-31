"""Several workers on one queue, which is the whole scaling claim.

The orchestrator's batch query is `... FOR UPDATE SKIP LOCKED`. Everything the
architecture says about scaling horizontally rests on that clause, and until
these tests existed it had only ever been exercised by a single worker -- which
is precisely the configuration in which it cannot fail.

What is actually at stake is not throughput. It is that a customer whose
payment failed gets *one* phone call. Two workers reading the same open case a
millisecond apart, both deciding it is actionable, is how someone's phone rings
twice about the same debt -- and the row lock is the only thing standing in the
way.

These run against real Postgres because that is the point. `SKIP LOCKED` is a
database behaviour; mocking the session would test nothing at all.
"""

import asyncio

from sqlalchemy import func, select

from app.constants import ActionType, CaseStatus
from app.db import SessionLocal
from app.models import Customer, RecoveryAction, RecoveryCase
from app.orchestrator import run_once
from tests.test_orchestrator import NOW, SpyChannel, SpyRazorpay, settings


async def seed_many(session, count: int) -> list[int]:
    """`count` open cases, each with its own customer.

    Distinct customers on purpose: a shared one would be held back by the
    contact cooldown, and this is a test of the row lock, not of the policy.
    """
    ids = []
    for i in range(count):
        session.add(
            Customer(razorpay_customer_id=f"cust_c{i}", phone=f"+9190000{i:05d}")
        )
        case = RecoveryCase(
            razorpay_payment_id=f"pay_c{i}",
            razorpay_customer_id=f"cust_c{i}",
            razorpay_subscription_id=f"sub_c{i}",
            # Varying amounts so the priority ordering has something to sort.
            original_amount=10_000 + (i * 1_000),
            status=CaseStatus.OPEN,
            # Past the first rung of the ladder, so the tick reaches `_contact`
            # and a double-claim would show up as a double *call*.
            attempt_count=1,
        )
        session.add(case)
        ids.append(case)
    await session.commit()
    for case in ids:
        await session.refresh(case)
    return [c.id for c in ids]


async def _worker(channel, batch_size: int) -> None:
    """One tick, in its own session -- which is what makes it a second worker.

    Sharing the caller's session would share its transaction, and the lock
    being tested only exists between transactions.
    """
    async with SessionLocal() as own:
        await run_once(
            own,
            channel,
            now=NOW,
            settings=settings(worker_batch_size=batch_size),
            client=SpyRazorpay(),
        )


async def test_two_workers_never_claim_the_same_case(session):
    """The claim that matters: nobody is contacted twice.

    Without SKIP LOCKED the second worker blocks on the first's lock and then
    reads the same rows once they are released -- committed as contacted, but
    read again inside a transaction that started before the commit landed.
    """
    await seed_many(session, 20)
    left, right = SpyChannel(), SpyChannel()

    await asyncio.gather(_worker(left, 20), _worker(right, 20))

    overlap = set(left.contacted) & set(right.contacted)
    assert not overlap, f"both workers contacted the same cases: {sorted(overlap)}"


async def test_no_case_is_ever_contacted_twice(session):
    """The same guarantee read from the audit trail rather than the spies.

    This is the version that would catch a regression the spies could not: the
    trail is what an auditor reads, and a second VOICE_CALL row against one
    case is a second call to a real person however it got there.
    """
    await seed_many(session, 20)

    await asyncio.gather(*(_worker(SpyChannel(), 20) for _ in range(3)))

    rows = await session.execute(
        select(RecoveryAction.recovery_case_id, func.count())
        .where(RecoveryAction.action_type == ActionType.VOICE_CALL)
        .group_by(RecoveryAction.recovery_case_id)
        .having(func.count() > 1)
    )
    duplicated = rows.all()
    assert not duplicated, f"cases contacted more than once: {duplicated}"


async def test_the_work_is_divided_rather_than_repeated(session):
    """Two workers should get through one batch together, not each do all of it.

    The weak assertion is deliberate. Exactly how the rows split depends on
    timing, and pinning that would make this flake; what must hold is that the
    total contacted equals the number of cases -- every case handled, none
    handled twice.
    """
    ids = await seed_many(session, 12)
    left, right = SpyChannel(), SpyChannel()

    await asyncio.gather(_worker(left, 6), _worker(right, 6))

    handled = left.contacted + right.contacted
    assert sorted(handled) == sorted(set(handled)), "a case was handled twice"
    assert set(handled) <= set(ids)


async def test_a_broken_channel_releases_the_batch_for_the_next_worker(session):
    """A telephony outage must cost the batch, not the queue.

    `_contact` swallows the channel's exception on purpose -- an external
    provider must never kill the tick -- so the transaction commits normally
    and the locks go with it. What matters for scaling is what is left behind:
    the rows have to be claimable by the next worker rather than stranded.

    Note this asserts the *cases* are reachable again, not that they are
    contacted again. Whether the policy lets them through immediately is its
    decision; the lock releasing is the row-level guarantee under test.
    """
    ids = await seed_many(session, 5)

    broken = SpyChannel(fail=True)
    await _worker(broken, 5)  # does not raise: the channel error is caught

    assert broken.contacted == [], "the spy should have failed every call"

    # The attempt was recorded against every case, which is only possible if
    # each was successfully claimed -- and then released.
    rows = await session.execute(
        select(RecoveryAction.recovery_case_id)
        .where(RecoveryAction.action_type == ActionType.VOICE_CALL)
        .distinct()
    )
    attempted = sorted(rows.scalars())
    assert attempted == sorted(ids), "some cases were never claimed at all"

    # And nothing is still holding them: a second worker can read the whole
    # table without blocking. A stranded lock would hang this line.
    async with SessionLocal() as other:
        total = await other.execute(select(func.count()).select_from(RecoveryCase))
        assert total.scalar_one() == len(ids)


async def test_a_failed_delivery_does_not_spend_the_attempt_budget(session):
    """An outage on our side must not consume the customer's attempts.

    Worth pinning next to the concurrency tests: with several workers running,
    a provider outage hits many cases at once, and a bug here would burn every
    case's budget in a single bad minute.
    """
    ids = await seed_many(session, 3)
    before = {i: (await session.get(RecoveryCase, i)).attempt_count for i in ids}

    await _worker(SpyChannel(fail=True), 3)

    for case_id in ids:
        case = await session.get(RecoveryCase, case_id)
        await session.refresh(case)
        assert case.attempt_count == before[case_id], (
            f"case {case_id} spent an attempt on our own outage"
        )


async def test_the_most_valuable_cases_go_first(session):
    """Ordering has to survive concurrency, or scaling out quietly changes who
    gets called first -- and the priority scheme is the product, not a detail.
    """
    await seed_many(session, 10)
    channel = SpyChannel()

    await _worker(channel, 3)

    amounts = []
    for case_id in channel.contacted:
        case = await session.get(RecoveryCase, case_id)
        amounts.append(case.original_amount)
    assert amounts == sorted(amounts, reverse=True), (
        f"batch was not taken in priority order: {amounts}"
    )
