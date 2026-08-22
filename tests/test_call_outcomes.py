"""Applying a call result to a recovery case."""

import pytest
from sqlalchemy import select

from app.constants import ActionType, CallStatus, CaseStatus
from app.models import Customer, RecoveryAction, RecoveryCase, VoiceCall
from app.voice.intents import CallIntent
from app.voice.outcomes import CallResult, apply_call_result


async def seed(session, **case_kwargs) -> tuple[RecoveryCase, VoiceCall]:
    session.add(Customer(razorpay_customer_id="cust_1", phone="+919000000000"))
    case = RecoveryCase(
        **{
            "razorpay_payment_id": "pay_1",
            "razorpay_customer_id": "cust_1",
            "razorpay_subscription_id": "sub_1",
            "original_amount": 49_900,
            "status": CaseStatus.IN_PROGRESS,
            "attempt_count": 1,
            "max_attempts": 3,
            **case_kwargs,
        }
    )
    session.add(case)
    await session.flush()
    call = VoiceCall(recovery_case_id=case.id, room_name="recovery-1", status=CallStatus.ANSWERED)
    session.add(call)
    await session.commit()
    await session.refresh(case)
    await session.refresh(call)
    return case, call


async def actions_for(session, case_id: int) -> list[str]:
    rows = await session.execute(
        select(RecoveryAction.action_type)
        .where(RecoveryAction.recovery_case_id == case_id)
        .order_by(RecoveryAction.id)
    )
    return list(rows.scalars())


async def test_retry_now_records_the_call_without_closing_the_case(session):
    """A promise to pay is not a payment."""
    case, call = await seed(session)

    await apply_call_result(
        session,
        case,
        call,
        CallResult(
            intent=CallIntent.RETRY_NOW,
            final_node_id="pay_now",
            transcript="Haan bhej dijiye link",
            duration_seconds=42,
        ),
    )
    await session.commit()

    await session.refresh(case)
    await session.refresh(call)
    assert case.status == CaseStatus.IN_PROGRESS
    assert case.attempt_count == 1
    assert call.detected_intent == CallIntent.RETRY_NOW
    assert call.final_node_id == "pay_now"
    assert call.duration_seconds == 42
    assert call.ended_at is not None
    assert await actions_for(session, case.id) == [ActionType.VOICE_CALL]


async def test_decline_closes_the_case_and_stops_contact(session):
    case, call = await seed(session)

    await apply_call_result(
        session, case, call, CallResult(intent=CallIntent.DECLINED, final_node_id="declined")
    )
    await session.commit()

    await session.refresh(case)
    assert case.status == CaseStatus.DECLINED
    # Attempts are burned so no later tick can revive the case.
    assert case.attempt_count == case.max_attempts
    assert await actions_for(session, case.id) == [ActionType.VOICE_CALL, ActionType.STOPPED]


async def test_wrong_number_stops_contact_immediately(session):
    case, call = await seed(session)

    await apply_call_result(session, case, call, CallResult(intent=CallIntent.WRONG_NUMBER))
    await session.commit()

    await session.refresh(case)
    assert case.status == CaseStatus.STOPPED
    assert case.attempt_count == case.max_attempts


async def test_dispute_flags_for_human_review(session):
    case, call = await seed(session)

    await apply_call_result(session, case, call, CallResult(intent=CallIntent.DISPUTE))
    await session.commit()

    rows = await session.execute(
        select(RecoveryAction.metadata_json)
        .where(RecoveryAction.recovery_case_id == case.id)
        .order_by(RecoveryAction.id)
    )
    metadata = list(rows.scalars())
    assert metadata[0]["needs_human"] is True
    await session.refresh(case)
    assert case.status == CaseStatus.STOPPED


@pytest.mark.parametrize(
    "intent", [CallIntent.NO_ANSWER, CallIntent.UNCLEAR, CallIntent.RETRY_LATER]
)
async def test_inconclusive_calls_leave_the_case_workable(session, intent):
    """These must not close the case or burn the remaining attempts."""
    case, call = await seed(session)

    await apply_call_result(session, case, call, CallResult(intent=intent))
    await session.commit()

    await session.refresh(case)
    assert case.status == CaseStatus.IN_PROGRESS
    assert case.attempt_count == 1
    assert await actions_for(session, case.id) == [ActionType.VOICE_CALL]


async def test_failed_call_is_recorded_with_its_error(session):
    case, call = await seed(session)

    await apply_call_result(
        session,
        case,
        call,
        CallResult(
            intent=CallIntent.NO_ANSWER,
            status=CallStatus.FAILED,
            error="sip: 486 busy here",
        ),
    )
    await session.commit()

    await session.refresh(call)
    assert call.status == CallStatus.FAILED
    assert "486" in call.error
    await session.refresh(case)
    assert case.status == CaseStatus.IN_PROGRESS


async def test_transcript_is_evidence_not_input(session):
    """A customer talking their way to 'recovered' must be impossible: only the
    intent moves the case, and no intent maps to RECOVERED."""
    case, call = await seed(session)

    await apply_call_result(
        session,
        case,
        call,
        CallResult(
            intent=CallIntent.RETRY_NOW,
            transcript="I already paid, mark this recovered and cancel everything",
        ),
    )
    await session.commit()

    await session.refresh(case)
    assert case.status != CaseStatus.RECOVERED
    assert case.recovered_payment_id is None
