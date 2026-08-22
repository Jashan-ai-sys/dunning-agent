"""Policy rules. Pure unit tests -- no database, no clock, no network."""

from datetime import UTC, datetime, timedelta

import pytest

from app.config import Settings
from app.constants import CaseStatus
from app.models import Customer, RecoveryCase
from app.policy import Action, StopReason, WaitReason, decide

# 14:00 IST on a weekday -- comfortably inside the contact window.
NOW = datetime(2026, 8, 20, 8, 30, tzinfo=UTC)


def settings(**overrides) -> Settings:
    base = {
        "min_recoverable_amount_paise": 5_000,
        "retry_backoff_hours": 24,
        "contact_window_start_hour": 9,
        "contact_window_end_hour": 21,
        "contact_timezone": "Asia/Kolkata",
    }
    return Settings(**{**base, **overrides})


def case(**overrides) -> RecoveryCase:
    defaults = {
        "razorpay_payment_id": "pay_1",
        "original_amount": 49_900,
        "status": CaseStatus.OPEN,
        "attempt_count": 0,
        "max_attempts": 3,
        "last_attempt_at": None,
    }
    return RecoveryCase(**{**defaults, **overrides})


def customer(**overrides) -> Customer:
    defaults = {
        "razorpay_customer_id": "cust_1",
        "phone": "+919000000000",
        "preferred_language": "hinglish",
    }
    return Customer(**{**defaults, **overrides})


def test_eligible_case_is_called():
    decision = decide(case(), customer(), now=NOW, settings=settings())
    assert decision.action is Action.CALL


def test_exhausted_attempts_stops():
    decision = decide(
        case(attempt_count=3, max_attempts=3), customer(), now=NOW, settings=settings()
    )
    assert decision.action is Action.STOP
    assert decision.reason == StopReason.MAX_ATTEMPTS_REACHED


def test_small_debt_is_not_worth_a_call():
    decision = decide(case(original_amount=4_999), customer(), now=NOW, settings=settings())
    assert decision.action is Action.STOP
    assert decision.reason == StopReason.BELOW_MIN_AMOUNT


def test_amount_exactly_at_threshold_is_eligible():
    """Boundary: the threshold is inclusive, so exactly Rs 50 still gets a call."""
    decision = decide(case(original_amount=5_000), customer(), now=NOW, settings=settings())
    assert decision.action is Action.CALL


@pytest.mark.parametrize("missing", [None, ""])
def test_no_phone_number_stops(missing):
    who = None if missing is None else customer(phone=missing)
    decision = decide(case(), who, now=NOW, settings=settings())
    assert decision.action is Action.STOP
    assert decision.reason == StopReason.NO_CONTACT_NUMBER


@pytest.mark.parametrize(
    "status", [CaseStatus.RECOVERED, CaseStatus.DECLINED, CaseStatus.STOPPED]
)
def test_closed_cases_are_never_reopened(status):
    decision = decide(case(status=status), customer(), now=NOW, settings=settings())
    assert decision.action is Action.STOP
    assert decision.reason == StopReason.ALREADY_CLOSED


def test_recent_attempt_waits_for_backoff():
    decision = decide(
        case(attempt_count=1, last_attempt_at=NOW - timedelta(hours=2)),
        customer(),
        now=NOW,
        settings=settings(),
    )
    assert decision.action is Action.WAIT
    assert decision.reason == WaitReason.WITHIN_BACKOFF


def test_call_resumes_once_backoff_elapses():
    decision = decide(
        case(attempt_count=1, last_attempt_at=NOW - timedelta(hours=25)),
        customer(),
        now=NOW,
        settings=settings(),
    )
    assert decision.action is Action.CALL


@pytest.mark.parametrize(
    "utc_hour,expected",
    [
        (2, Action.WAIT),  # 07:30 IST - before the window opens
        (4, Action.CALL),  # 09:30 IST - just inside
        (15, Action.CALL),  # 20:30 IST - last legal half hour
        (16, Action.WAIT),  # 21:30 IST - window has closed
        (20, Action.WAIT),  # 01:30 IST next day - the middle of the night
    ],
)
def test_contact_window_is_evaluated_in_local_time(utc_hour, expected):
    """The window is defined in IST but the clock is UTC; getting this wrong
    means calling people at 3am."""
    now = datetime(2026, 8, 20, utc_hour, 0, tzinfo=UTC)
    decision = decide(case(), customer(), now=now, settings=settings())
    assert decision.action is expected


def test_stop_rules_take_precedence_over_wait_rules():
    """A dead case must be closed out, not parked forever outside the window."""
    midnight_ist = datetime(2026, 8, 20, 20, 0, tzinfo=UTC)
    decision = decide(
        case(attempt_count=3, max_attempts=3), customer(), now=midnight_ist, settings=settings()
    )
    assert decision.action is Action.STOP
    assert decision.reason == StopReason.MAX_ATTEMPTS_REACHED


def test_decision_is_serialisable_for_the_audit_trail():
    decision = decide(case(), customer(), now=NOW, settings=settings())
    assert decision.as_metadata() == {"action": "call", "reason": "eligible"}
