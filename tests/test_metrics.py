"""Batch metrics. These are the numbers judges read, so the definitions matter
more than the SQL."""

import pytest

from app.constants import CallStatus, CaseStatus
from app.metrics import BatchMetrics, compute_metrics, format_report, rupees
from app.models import RecoveryCase, VoiceCall


async def add_case(session, payment_id, amount, status=CaseStatus.OPEN, **kwargs):
    case = RecoveryCase(
        razorpay_payment_id=payment_id,
        razorpay_subscription_id="sub_1",
        razorpay_customer_id="cust_1",
        original_amount=amount,
        status=status,
        **kwargs,
    )
    session.add(case)
    await session.flush()
    return case


# --- formatting --------------------------------------------------------


@pytest.mark.parametrize(
    "paise,expected",
    [
        (0, "₹0.00"),
        (None, "₹0.00"),
        (4_999, "₹49.99"),
        (100_000, "₹1,000.00"),
        (10_000_000, "₹1,00,000.00"),
        (1_000_000_000, "₹1,00,00,000.00"),
    ],
)
def test_rupees_uses_indian_grouping(paise, expected):
    """Lakh/crore grouping, not thousands -- this is read by Indian judges."""
    assert rupees(paise) == expected


# --- rate definitions --------------------------------------------------


def test_rates_are_zero_when_nothing_is_at_risk():
    """Never divide by zero on an empty batch."""
    metrics = BatchMetrics()
    assert metrics.recovery_rate_by_amount == 0.0
    assert metrics.recovery_rate_by_count == 0.0


def test_amount_rate_and_count_rate_can_disagree():
    """One big recovery beats several small ones; the headline is by amount."""
    metrics = BatchMetrics(
        total_cases=4, recovered_cases=1, amount_at_risk=100_000, amount_recovered=90_000
    )
    assert metrics.recovery_rate_by_amount == 0.9
    assert metrics.recovery_rate_by_count == 0.25


# --- aggregation -------------------------------------------------------


async def test_empty_batch_reports_zeroes(session):
    metrics = await compute_metrics(session)
    assert metrics.total_cases == 0
    assert metrics.amount_at_risk == 0
    assert metrics.recovery_rate_by_amount == 0.0


async def test_totals_and_headline_rate(session):
    await add_case(session, "pay_1", 50_000, CaseStatus.RECOVERED, recovered_amount=50_000)
    await add_case(session, "pay_2", 30_000, CaseStatus.OPEN)
    await add_case(session, "pay_3", 20_000, CaseStatus.DECLINED)
    await session.commit()

    metrics = await compute_metrics(session)

    assert metrics.total_cases == 3
    assert metrics.recovered_cases == 1
    assert metrics.amount_at_risk == 100_000
    assert metrics.amount_recovered == 50_000
    assert metrics.amount_outstanding == 50_000
    assert metrics.recovery_rate_by_amount == 0.5


async def test_stopped_cases_stay_in_the_denominator(session):
    """Excluding cases the policy declined to work would flatter the rate."""
    await add_case(session, "pay_1", 50_000, CaseStatus.RECOVERED, recovered_amount=50_000)
    await add_case(session, "pay_2", 50_000, CaseStatus.STOPPED)
    await session.commit()

    metrics = await compute_metrics(session)

    assert metrics.total_cases == 2
    assert metrics.amount_at_risk == 100_000
    assert metrics.recovery_rate_by_amount == 0.5


async def test_a_promise_to_pay_is_not_a_recovery(session):
    """An in-progress case after a 'retry_now' call must not count."""
    await add_case(session, "pay_1", 50_000, CaseStatus.IN_PROGRESS, attempt_count=1)
    await session.commit()

    metrics = await compute_metrics(session)

    assert metrics.recovered_cases == 0
    assert metrics.amount_recovered == 0
    assert metrics.recovery_rate_by_amount == 0.0


async def test_partial_recovery_is_counted_at_the_amount_actually_paid(session):
    await add_case(session, "pay_1", 50_000, CaseStatus.RECOVERED, recovered_amount=30_000)
    await session.commit()

    metrics = await compute_metrics(session)

    assert metrics.amount_at_risk == 50_000
    assert metrics.amount_recovered == 30_000
    assert metrics.recovery_rate_by_amount == 0.6


async def test_breakdown_by_status_and_failure_code(session):
    await add_case(session, "pay_1", 10_000, CaseStatus.OPEN, failure_code="INSUFFICIENT_FUNDS")
    await add_case(session, "pay_2", 10_000, CaseStatus.OPEN, failure_code="INSUFFICIENT_FUNDS")
    await add_case(session, "pay_3", 10_000, CaseStatus.DECLINED, failure_code="CARD_EXPIRED")
    await add_case(session, "pay_4", 10_000, CaseStatus.OPEN)
    await session.commit()

    metrics = await compute_metrics(session)

    assert metrics.by_status == {CaseStatus.OPEN: 3, CaseStatus.DECLINED: 1}
    assert metrics.amount_by_status[CaseStatus.OPEN] == 30_000
    assert metrics.by_failure_code["INSUFFICIENT_FUNDS"] == 2
    assert metrics.by_failure_code["unknown"] == 1


async def test_call_intent_mix(session):
    case = await add_case(session, "pay_1", 10_000)
    session.add(
        VoiceCall(
            recovery_case_id=case.id, status=CallStatus.COMPLETED, detected_intent="retry_now"
        )
    )
    session.add(
        VoiceCall(
            recovery_case_id=case.id, status=CallStatus.COMPLETED, detected_intent="declined"
        )
    )
    session.add(VoiceCall(recovery_case_id=case.id, status=CallStatus.FAILED))
    await session.commit()

    metrics = await compute_metrics(session)

    assert metrics.calls_placed == 3
    assert metrics.calls_by_intent["retry_now"] == 1
    assert metrics.calls_by_intent["declined"] == 1
    assert metrics.calls_by_intent["none"] == 1


async def test_median_attempts_only_counts_recovered_cases(session):
    await add_case(
        session, "pay_1", 10_000, CaseStatus.RECOVERED, recovered_amount=10_000, attempt_count=1
    )
    await add_case(
        session, "pay_2", 10_000, CaseStatus.RECOVERED, recovered_amount=10_000, attempt_count=3
    )
    await add_case(session, "pay_3", 10_000, CaseStatus.STOPPED, attempt_count=99)
    await session.commit()

    metrics = await compute_metrics(session)

    assert metrics.median_attempts_to_recover == 2.0


async def test_report_renders_without_crashing_on_an_empty_batch(session):
    """The pitch script runs this; it must not blow up before there is data."""
    metrics = await compute_metrics(session)
    output = format_report(metrics)
    assert "Recovery batch summary" in output
    assert "₹0.00" in output


async def test_report_includes_the_headline_numbers(session):
    await add_case(session, "pay_1", 250_000, CaseStatus.RECOVERED, recovered_amount=250_000)
    await add_case(session, "pay_2", 250_000, CaseStatus.OPEN)
    await session.commit()

    output = format_report(await compute_metrics(session))

    assert "₹5,000.00" in output  # at risk
    assert "₹2,500.00" in output  # recovered
    assert "50.0%" in output


# --- source separation -------------------------------------------------


async def test_source_filter_isolates_seeded_cases(session):
    """Simulated recovery must never be reportable as real money."""
    await add_case(session, "pay_real", 100_000, CaseStatus.OPEN)
    await add_case(
        session, "pay_seed", 50_000, CaseStatus.RECOVERED,
        recovered_amount=50_000, source="seed",
    )
    await session.commit()

    real = await compute_metrics(session, source="razorpay")
    seeded = await compute_metrics(session, source="seed")

    assert real.total_cases == 1
    assert real.amount_recovered == 0
    assert real.recovery_rate_by_amount == 0.0

    assert seeded.total_cases == 1
    assert seeded.amount_recovered == 50_000
    assert seeded.recovery_rate_by_amount == 1.0


async def test_cases_default_to_the_razorpay_source(session):
    await add_case(session, "pay_1", 10_000)
    await session.commit()

    metrics = await compute_metrics(session)
    assert metrics.by_source == {"razorpay": 1}


async def test_report_warns_when_sources_are_mixed(session):
    """An unfiltered total spanning both is misleading, so say so loudly."""
    await add_case(session, "pay_real", 10_000)
    await add_case(session, "pay_seed", 10_000, source="seed")
    await session.commit()

    output = format_report(await compute_metrics(session))
    assert "MIXED SOURCES" in output
    assert "--source" in output


async def test_report_does_not_warn_on_a_single_source(session):
    await add_case(session, "pay_seed", 10_000, source="seed")
    await session.commit()

    output = format_report(await compute_metrics(session, source="seed"))
    assert "MIXED SOURCES" not in output


async def test_call_counts_respect_the_source_filter(session):
    real = await add_case(session, "pay_real", 10_000)
    seeded = await add_case(session, "pay_seed", 10_000, source="seed")
    session.add(VoiceCall(recovery_case_id=real.id, detected_intent="declined"))
    session.add(VoiceCall(recovery_case_id=seeded.id, detected_intent="retry_now"))
    await session.commit()

    assert (await compute_metrics(session, source="seed")).calls_placed == 1
    assert (await compute_metrics(session, source="razorpay")).calls_by_intent == {"declined": 1}
