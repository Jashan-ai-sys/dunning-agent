"""Policy rules. Pure unit tests -- no database, no clock, no network."""

from datetime import UTC, datetime, timedelta

import pytest

from app.config import Settings
from app.constants import CaseStatus
from app.diagnosis import RootCause
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


def test_a_fresh_case_opens_with_the_cheap_intervention():
    """Nobody has been sent anything yet, so a link is enough. A call is what
    the *second* attempt costs."""
    decision = decide(case(), customer(), now=NOW, settings=settings())
    assert decision.action is Action.LINK


def test_a_second_attempt_escalates_to_a_call():
    """attempt_count 1 means a link has already gone out and been ignored."""
    decision = decide(case(attempt_count=1), customer(), now=NOW, settings=settings())
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
    """Boundary: the threshold is inclusive, so exactly Rs 50 is still worked."""
    decision = decide(case(original_amount=5_000), customer(), now=NOW, settings=settings())
    assert decision.action is Action.LINK


@pytest.mark.parametrize("missing", [None, ""])
def test_a_customer_we_cannot_reach_at_all_stops(missing):
    who = None if missing is None else customer(phone=missing)
    decision = decide(case(), who, now=NOW, settings=settings())
    assert decision.action is Action.STOP
    assert decision.reason == StopReason.NO_CONTACT_DETAILS


def test_an_email_only_customer_is_still_worth_a_link():
    """A payment link goes by email too. Stopping here because there is no
    phone would abandon money we can still collect."""
    decision = decide(
        case(), customer(phone=None, email="a@example.com"), now=NOW, settings=settings()
    )
    assert decision.action is Action.LINK


def test_an_email_only_customer_is_never_escalated_to_a_call():
    """There is nothing to dial, so the second attempt is another link rather
    than a CALL the orchestrator could not place."""
    decision = decide(
        case(attempt_count=1),
        customer(phone=None, email="a@example.com"),
        now=NOW,
        settings=settings(),
    )
    assert decision.action is Action.LINK


def test_a_wrong_number_is_not_dialled_again_but_email_still_works():
    """Somebody else answered that phone. The customer still owes the money."""
    who = customer(phone="+919000000000", phone_is_wrong=True, email="a@example.com")
    decision = decide(case(attempt_count=1), who, now=NOW, settings=settings())
    assert decision.action is Action.LINK


def test_a_wrong_number_with_no_email_stops():
    who = customer(phone="+919000000000", phone_is_wrong=True)
    decision = decide(case(), who, now=NOW, settings=settings())
    assert decision.action is Action.STOP
    assert decision.reason == StopReason.NO_CONTACT_DETAILS


@pytest.mark.parametrize(
    "status", [CaseStatus.RECOVERED, CaseStatus.DECLINED, CaseStatus.STOPPED]
)
def test_closed_cases_are_never_reopened(status):
    decision = decide(case(status=status), customer(), now=NOW, settings=settings())
    assert decision.action is Action.STOP
    assert decision.reason == StopReason.ALREADY_CLOSED


def test_suppressed_customer_is_never_called():
    decision = decide(case(), customer(do_not_contact=True), now=NOW, settings=settings())
    assert decision.action is Action.STOP
    assert decision.reason == StopReason.DO_NOT_CONTACT


def test_suppression_holds_across_a_brand_new_case():
    """The bug this rule exists for.

    Burning ``attempt_count`` only closes the case that was on the call. A
    second failed charge from the same person opens a *fresh* case with a full
    attempt budget, and nothing case-scoped would stop us dialling a number we
    were already told was wrong.
    """
    decision = decide(
        case(razorpay_payment_id="pay_2", attempt_count=0, status=CaseStatus.OPEN),
        customer(do_not_contact=True),
        now=NOW,
        settings=settings(),
    )
    assert decision.action is Action.STOP
    assert decision.reason == StopReason.DO_NOT_CONTACT


def test_suppression_outranks_the_attempt_budget_in_the_audit_trail():
    """Both stop the case; only one of them explains itself honestly."""
    decision = decide(
        case(attempt_count=3, max_attempts=3),
        customer(do_not_contact=True),
        now=NOW,
        settings=settings(),
    )
    assert decision.reason == StopReason.DO_NOT_CONTACT


def test_unflushed_customer_is_not_treated_as_suppressed():
    """A Customer built in memory has ``do_not_contact`` None, not False -- the
    column default only lands at INSERT. The rule must read that as 'not
    suppressed' rather than skipping every call in a unit test."""
    who = customer()
    assert who.do_not_contact is None
    assert decide(case(), who, now=NOW, settings=settings()).action is not Action.STOP


def test_recent_attempt_waits_for_backoff():
    decision = decide(
        case(attempt_count=1, last_attempt_at=NOW - timedelta(hours=2)),
        customer(),
        now=NOW,
        settings=settings(),
    )
    assert decision.action is Action.WAIT
    assert decision.reason == WaitReason.WITHIN_BACKOFF


def test_contact_resumes_once_backoff_elapses():
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
        (4, Action.LINK),  # 09:30 IST - just inside
        (15, Action.LINK),  # 20:30 IST - last legal half hour
        (16, Action.WAIT),  # 21:30 IST - window has closed
        (20, Action.WAIT),  # 01:30 IST next day - the middle of the night
    ],
)
def test_contact_window_is_evaluated_in_local_time(utc_hour, expected):
    """The window is defined in IST but the clock is UTC; getting this wrong
    means calling people at 3am.

    The window binds the cheap intervention too: a payment link is still an SMS
    landing on somebody's phone.
    """
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
    decision = decide(
        case(failure_source="issuer", failure_reason_code="insufficient_funds"),
        customer(),
        now=NOW,
        settings=settings(),
    )
    assert decision.as_metadata() == {
        "action": "link",
        "reason": "eligible",
        "root_cause": "customer_funds",
    }


def test_every_decision_carries_the_root_cause_even_when_it_did_not_drive_it():
    """A stop that had nothing to do with the diagnosis still has to say what
    was wrong with the payment -- that is the audit trail's job."""
    decision = decide(
        case(status=CaseStatus.RECOVERED, failure_reason_code="card_expired"),
        customer(),
        now=NOW,
        settings=settings(),
    )
    assert decision.reason == StopReason.ALREADY_CLOSED
    assert decision.root_cause is RootCause.CUSTOMER_INSTRUMENT


# --- Root cause drives the intervention -----------------------------------


def test_a_dead_instrument_earns_a_call_on_the_first_attempt():
    """An expired card is the case a link cannot close on its own.

    The link would recover today's money and leave the mandate broken, so the
    same charge fails again next cycle. That is worth a conversation
    immediately rather than after a wasted link.
    """
    decision = decide(
        case(failure_source="customer", failure_reason_code="card_expired"),
        customer(),
        now=NOW,
        settings=settings(),
    )
    assert decision.action is Action.CALL
    assert decision.root_cause is RootCause.CUSTOMER_INSTRUMENT


def test_insufficient_funds_still_opens_with_a_link():
    """Nothing is wrong with the instrument, so paying is all it takes."""
    decision = decide(
        case(failure_source="issuer", failure_reason_code="insufficient_funds"),
        customer(),
        now=NOW,
        settings=settings(),
    )
    assert decision.action is Action.LINK


@pytest.mark.parametrize("source", ["gateway", "internal"])
def test_our_own_outage_never_reaches_the_customer(source):
    """Calling somebody about a failure that was ours is useless and rude."""
    decision = decide(
        case(failure_source=source), customer(), now=NOW, settings=settings()
    )
    assert decision.action is Action.WAIT
    assert decision.reason == WaitReason.AWAITING_BANK_RETRY


def test_a_bank_decline_waits_for_razorpay_to_finish_retrying():
    decision = decide(
        case(failure_source="issuer"), customer(), now=NOW, settings=settings()
    )
    assert decision.action is Action.WAIT
    assert decision.reason == WaitReason.AWAITING_BANK_RETRY


def test_once_razorpay_halts_the_debt_is_ours_to_chase():
    """halted_at means Razorpay has given up its own retries. Waiting for a
    retry that will never come would park the case forever."""
    decision = decide(
        case(failure_source="issuer", halted_at=NOW - timedelta(hours=1)),
        customer(),
        now=NOW,
        settings=settings(),
    )
    assert decision.action is Action.LINK


def test_our_misconfiguration_goes_to_a_human_not_the_customer():
    decision = decide(
        case(failure_source="business"), customer(), now=NOW, settings=settings()
    )
    assert decision.action is Action.STOP
    assert decision.reason == StopReason.NEEDS_HUMAN


def test_an_undiagnosable_case_is_still_worked():
    """No failure fields must not mean no recovery -- it means the ladder runs
    without a diagnosis to sharpen it."""
    decision = decide(case(), customer(), now=NOW, settings=settings())
    assert decision.root_cause is RootCause.UNKNOWN
    assert decision.action is Action.LINK


# --- The bank-retry wait must end -----------------------------------------


def test_deferring_to_the_bank_expires_even_if_halted_never_arrives():
    """The regression this rule exists for.

    ``subscription.halted`` is a webhook, and webhooks go missing -- the
    subscription is cancelled rather than halted, the event is not subscribed,
    the delivery is lost past the replay window. Waiting on it unconditionally
    made a second permanent terminal state: the case never moved, never burned
    an attempt, and never reached the attempt cap. Worse, it stayed OPEN and
    oldest, so it sat at the front of every batch and crowded newer cases out.
    """
    old_case = case(
        failure_source="issuer",
        halted_at=None,
        created_at=NOW - timedelta(hours=80),
    )
    decision = decide(old_case, customer(), now=NOW, settings=settings())
    assert decision.action is Action.LINK


def test_the_bank_still_gets_the_first_go_inside_the_grace_window():
    fresh = case(failure_source="issuer", created_at=NOW - timedelta(hours=2))
    decision = decide(fresh, customer(), now=NOW, settings=settings())
    assert decision.action is Action.WAIT
    assert decision.reason == WaitReason.AWAITING_BANK_RETRY


def test_the_grace_window_is_configurable():
    aged = case(failure_source="gateway", created_at=NOW - timedelta(hours=10))
    assert decide(aged, customer(), now=NOW, settings=settings()).action is Action.WAIT
    assert (
        decide(
            aged, customer(), now=NOW, settings=settings(bank_retry_grace_hours=6)
        ).action
        is Action.LINK
    )


def test_a_case_with_no_created_at_is_treated_as_brand_new():
    """An unflushed case has no created_at; the age subtraction must not crash."""
    decision = decide(
        case(failure_source="issuer", created_at=None),
        customer(),
        now=NOW,
        settings=settings(),
    )
    assert decision.action is Action.WAIT


# --- Mandate retry (opt-in) -----------------------------------------------


def test_mandate_retry_is_off_unless_switched_on():
    """It takes money without the customer doing anything, so a deployment
    opts in rather than inherits it."""
    decision = decide(
        case(failure_source="issuer", failure_reason_code="insufficient_funds"),
        customer(),
        now=NOW,
        settings=settings(),
    )
    assert decision.action is Action.LINK


def test_missing_funds_re_charges_the_mandate_when_enabled():
    """The instrument is fine; only the money was absent. Charging the mandate
    they already authorised beats asking them to pay all over again."""
    decision = decide(
        case(failure_source="issuer", failure_reason_code="insufficient_funds"),
        customer(),
        now=NOW,
        settings=settings(mandate_retry_enabled=True),
    )
    assert decision.action is Action.RETRY_MANDATE


def test_a_dead_instrument_is_never_re_charged():
    """Retrying a revoked mandate or an expired card cannot work, and doing it
    silently is worse than doing it uselessly."""
    decision = decide(
        case(failure_source="customer", failure_reason_code="mandate_revoked"),
        customer(),
        now=NOW,
        settings=settings(mandate_retry_enabled=True),
    )
    assert decision.action is Action.CALL


def test_a_second_attempt_does_not_re_charge():
    """One silent charge per case. After that they hear from us."""
    decision = decide(
        case(
            failure_source="issuer",
            failure_reason_code="insufficient_funds",
            attempt_count=1,
        ),
        customer(),
        now=NOW,
        settings=settings(mandate_retry_enabled=True),
    )
    assert decision.action is Action.CALL


# --- One person, several debts ---------------------------------------------


def test_a_recently_contacted_customer_is_not_rung_again():
    """The rule that was missing.

    Attempt budgets and backoff are per case, which assumed one case per
    customer. Four cases opened against one subscription in two hours here,
    each with a full untouched budget -- so the same person would have been
    called four times in one tick.
    """
    decision = decide(
        case(razorpay_payment_id="pay_second_debt"),
        customer(last_contacted_at=NOW - timedelta(hours=2)),
        now=NOW,
        settings=settings(),
    )
    assert decision.action is Action.WAIT
    assert decision.reason == WaitReason.CUSTOMER_RECENTLY_CONTACTED


def test_the_cooldown_expires():
    decision = decide(
        case(),
        customer(last_contacted_at=NOW - timedelta(hours=25)),
        now=NOW,
        settings=settings(),
    )
    assert decision.action is not Action.WAIT


def test_the_cooldown_is_configurable():
    who = customer(last_contacted_at=NOW - timedelta(hours=4))
    assert decide(case(), who, now=NOW, settings=settings()).action is Action.WAIT
    assert (
        decide(case(), who, now=NOW, settings=settings(customer_contact_cooldown_hours=1)).action
        is not Action.WAIT
    )


def test_a_customer_never_contacted_is_not_held_back():
    who = customer()
    assert who.last_contacted_at is None
    assert decide(case(), who, now=NOW, settings=settings()).action is not Action.WAIT


def test_the_cooldown_parks_the_case_until_it_lifts():
    """A parked case must say when it is worth reading again, or it occupies
    the batch on every tick."""
    contacted = NOW - timedelta(hours=2)
    decision = decide(
        case(), customer(last_contacted_at=contacted), now=NOW, settings=settings()
    )
    assert decision.retry_after == contacted + timedelta(hours=24)


def test_a_stop_still_outranks_the_cooldown():
    """Cooldown is a WAIT. A case that should be closed must still close, or it
    sits in the queue forever behind a person we keep not calling."""
    decision = decide(
        case(attempt_count=3, max_attempts=3),
        customer(last_contacted_at=NOW - timedelta(minutes=5)),
        now=NOW,
        settings=settings(),
    )
    assert decision.action is Action.STOP
    assert decision.reason == StopReason.MAX_ATTEMPTS_REACHED
