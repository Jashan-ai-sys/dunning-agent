"""Orchestrator behaviour against a real schema."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.channels import ContactResult
from app.config import Settings
from app.constants import ActionType, CaseStatus
from app.models import Customer, RecoveryAction, RecoveryCase
from app.orchestrator import run_once

NOW = datetime(2026, 8, 20, 8, 30, tzinfo=UTC)  # 14:00 IST, inside the window


def settings(**overrides) -> Settings:
    base = {
        "min_recoverable_amount_paise": 5_000,
        "retry_backoff_hours": 24,
        "contact_window_start_hour": 9,
        "contact_window_end_hour": 21,
        "contact_timezone": "Asia/Kolkata",
        "worker_batch_size": 50,
    }
    return Settings(**{**base, **overrides})


class SpyChannel:
    """Records who it was asked to contact."""

    name = "spy"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.contacted: list[int] = []

    async def initiate(self, case: RecoveryCase, customer: Customer) -> ContactResult:
        if self.fail:
            raise RuntimeError("telephony provider unavailable")
        self.contacted.append(case.id)
        return ContactResult(channel=self.name, reference="room_1")


async def seed(session, *, case_kwargs=None, customer_kwargs=None) -> RecoveryCase:
    customer = Customer(
        **{
            "razorpay_customer_id": "cust_1",
            "phone": "+919000000000",
            **(customer_kwargs or {}),
        }
    )
    session.add(customer)
    case = RecoveryCase(
        **{
            "razorpay_payment_id": "pay_1",
            "razorpay_customer_id": customer.razorpay_customer_id,
            "razorpay_subscription_id": "sub_1",
            "original_amount": 49_900,
            "status": CaseStatus.OPEN,
            **(case_kwargs or {}),
        }
    )
    session.add(case)
    await session.commit()
    await session.refresh(case)
    return case


async def actions_for(session, case_id: int) -> list[str]:
    rows = await session.execute(
        select(RecoveryAction.action_type)
        .where(RecoveryAction.recovery_case_id == case_id)
        .order_by(RecoveryAction.id)
    )
    return list(rows.scalars())


async def test_eligible_case_is_contacted_and_recorded(session):
    case = await seed(session)
    channel = SpyChannel()

    result = await run_once(session, channel, now=NOW, settings=settings())

    assert result.as_dict()["contacted"] == 1
    assert channel.contacted == [case.id]

    await session.refresh(case)
    assert case.status == CaseStatus.IN_PROGRESS
    assert case.attempt_count == 1
    assert case.last_attempt_at is not None
    assert await actions_for(session, case.id) == [
        ActionType.POLICY_DECISION,
        ActionType.VOICE_CALL,
    ]


async def test_backoff_prevents_a_second_call_in_the_same_day(session):
    """The stopping-rule guarantee: one tick per minute must not mean one call
    per minute."""
    case = await seed(session)
    channel = SpyChannel()

    await run_once(session, channel, now=NOW, settings=settings())
    await run_once(session, channel, now=NOW + timedelta(minutes=1), settings=settings())

    await session.refresh(case)
    assert case.attempt_count == 1
    assert len(channel.contacted) == 1


async def test_case_is_stopped_once_attempts_are_exhausted(session):
    case = await seed(session, case_kwargs={"attempt_count": 3, "max_attempts": 3})
    channel = SpyChannel()

    result = await run_once(session, channel, now=NOW, settings=settings())

    assert result.as_dict()["stopped"] == 1
    assert channel.contacted == []
    await session.refresh(case)
    assert case.status == CaseStatus.STOPPED
    assert await actions_for(session, case.id) == [
        ActionType.POLICY_DECISION,
        ActionType.STOPPED,
    ]


async def test_stopped_case_is_not_picked_up_again(session):
    """STOP is permanent -- the bounded-workflow requirement."""
    await seed(session, case_kwargs={"attempt_count": 3, "max_attempts": 3})
    channel = SpyChannel()

    await run_once(session, channel, now=NOW, settings=settings())
    second = await run_once(session, channel, now=NOW, settings=settings())

    assert second.considered == 0


async def test_waiting_case_writes_no_audit_noise(session):
    """A case outside the contact window is re-evaluated every tick; it must not
    append a row each time."""
    case = await seed(session)
    channel = SpyChannel()
    midnight_ist = datetime(2026, 8, 20, 20, 0, tzinfo=UTC)

    for _ in range(3):
        result = await run_once(session, channel, now=midnight_ist, settings=settings())

    assert result.as_dict()["waiting"] == 1
    assert channel.contacted == []
    assert await actions_for(session, case.id) == []
    await session.refresh(case)
    assert case.status == CaseStatus.OPEN


async def test_channel_failure_does_not_burn_an_attempt(session):
    """An outage on our side must not consume the customer's attempt budget,
    but must still back off rather than hot-loop."""
    case = await seed(session)
    channel = SpyChannel(fail=True)

    result = await run_once(session, channel, now=NOW, settings=settings())

    assert result.as_dict()["failed"] == 1
    await session.refresh(case)
    assert case.attempt_count == 0
    assert case.last_attempt_at is not None
    assert await actions_for(session, case.id) == [
        ActionType.POLICY_DECISION,
        ActionType.VOICE_CALL,
    ]


async def test_channel_failure_does_not_abort_the_batch(session):
    """One bad case must not stop the other cases in the tick."""
    await seed(session)
    await seed(
        session,
        case_kwargs={"razorpay_payment_id": "pay_2"},
        customer_kwargs={"razorpay_customer_id": "cust_2"},
    )
    channel = SpyChannel(fail=True)

    result = await run_once(session, channel, now=NOW, settings=settings())

    assert result.considered == 2
    assert result.as_dict()["failed"] == 2


async def test_case_without_a_reachable_customer_is_stopped(session):
    case = await seed(session, customer_kwargs={"phone": None})
    channel = SpyChannel()

    await run_once(session, channel, now=NOW, settings=settings())

    await session.refresh(case)
    assert case.status == CaseStatus.STOPPED
    assert channel.contacted == []


async def test_halted_cases_are_worked_first(session):
    """Razorpay has given up on a halted subscription, so it is the most urgent
    thing in the queue."""
    await seed(session, case_kwargs={"razorpay_payment_id": "pay_old"})
    halted = await seed(
        session,
        case_kwargs={
            "razorpay_payment_id": "pay_halted",
            "halted_at": NOW - timedelta(hours=1),
        },
        customer_kwargs={"razorpay_customer_id": "cust_2"},
    )
    channel = SpyChannel()

    await run_once(session, channel, now=NOW, settings=settings(worker_batch_size=50))

    assert channel.contacted[0] == halted.id


async def test_batch_size_bounds_the_tick(session):
    for i in range(3):
        await seed(
            session,
            case_kwargs={"razorpay_payment_id": f"pay_{i}"},
            customer_kwargs={"razorpay_customer_id": f"cust_{i}"},
        )
    channel = SpyChannel()

    result = await run_once(session, channel, now=NOW, settings=settings(worker_batch_size=2))

    assert result.considered == 2


@pytest.mark.parametrize("amount", [4_999, 1])
async def test_small_debts_are_stopped_not_called(session, amount):
    case = await seed(session, case_kwargs={"original_amount": amount})
    channel = SpyChannel()

    await run_once(session, channel, now=NOW, settings=settings())

    await session.refresh(case)
    assert case.status == CaseStatus.STOPPED
    assert channel.contacted == []
