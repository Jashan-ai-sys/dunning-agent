"""The seam between a finished call and the database.

Until this existed the agent determined an intent and then discarded it: the
conversation worked and nothing moved. These tests cover the path that closes
that gap, plus the two rules that keep it safe -- never break a call, never
invent a case.
"""

from sqlalchemy import func, select

from app.constants import ActionType, CallStatus, CaseStatus
from app.models import Customer, RecoveryCase, VoiceCall
from app.voice.intents import CallIntent
from app.voice.outcomes import CallResult
from app.voice.persistence import finalise_call, open_call_record


class FakeLinkClient:
    def __init__(self) -> None:
        self.calls = 0

    async def create_payment_link(self, payload: dict) -> dict:
        self.calls += 1
        return {
            "id": "plink_live",
            "short_url": "https://rzp.io/i/plink_live",
            "reference_id": payload["reference_id"],
        }


async def seed(session, **case_kwargs) -> RecoveryCase:
    session.add(
        Customer(
            razorpay_customer_id="cust_1",
            name="Asha Rao",
            phone="+919000000000",
            preferred_language="hi",
        )
    )
    case = RecoveryCase(
        **{
            "razorpay_payment_id": "pay_1",
            "razorpay_customer_id": "cust_1",
            "razorpay_subscription_id": "sub_1",
            "original_amount": 49_900,
            "status": CaseStatus.IN_PROGRESS,
            **case_kwargs,
        }
    )
    session.add(case)
    await session.commit()
    await session.refresh(case)
    return case


# --- opening the record ------------------------------------------------


async def test_a_call_is_recorded_when_it_starts(session):
    """Written up front so an abandoned call still leaves a trace."""
    case = await seed(session)

    call_id = await open_call_record(
        recovery_case_id=case.id, room_name="recovery-1", dialled_number="+919000000000"
    )
    assert call_id is not None

    call = (await session.execute(select(VoiceCall))).scalar_one()
    assert call.recovery_case_id == case.id
    assert call.room_name == "recovery-1"
    assert call.dialled_number == "+919000000000"
    assert call.status == "initiated"
    assert call.ended_at is None


async def test_a_demo_run_writes_nothing(session):
    """No case id means the console or demo_call without --case. Invent nothing."""
    assert await open_call_record(recovery_case_id=None, room_name="demo") is None
    count = (await session.execute(select(func.count(VoiceCall.id)))).scalar_one()
    assert count == 0


async def test_an_unknown_case_id_is_refused(session):
    assert await open_call_record(recovery_case_id=999_999, room_name="r") is None


# --- finalising --------------------------------------------------------


async def test_retry_now_closes_the_loop_and_sends_a_link(session):
    """The path that was missing: conversation -> intent -> case -> money."""
    case = await seed(session)
    call_id = await open_call_record(recovery_case_id=case.id, room_name="recovery-1")
    client = FakeLinkClient()

    await finalise_call(
        voice_call_id=call_id,
        recovery_case_id=case.id,
        result=CallResult(
            intent=CallIntent.RETRY_NOW,
            final_node_id="pay_now",
            transcript="assistant: ...\nuser: link bhej dijiye",
            duration_seconds=44,
            transitions=[{"node_id": "ask_intent", "label": "pay_now", "accepted": True}],
        ),
        _client=client,
    )

    await session.refresh(case)
    call = await session.get(VoiceCall, call_id)
    assert call.detected_intent == CallIntent.RETRY_NOW
    assert call.duration_seconds == 44
    assert call.ended_at is not None
    assert call.transitions[0]["label"] == "pay_now"
    assert "link bhej dijiye" in call.transcript
    assert case.payment_link_url == "https://rzp.io/i/plink_live"
    assert client.calls == 1


async def test_declining_stops_contact_and_sends_no_link(session):
    case = await seed(session)
    call_id = await open_call_record(recovery_case_id=case.id, room_name="recovery-1")
    client = FakeLinkClient()

    await finalise_call(
        voice_call_id=call_id,
        recovery_case_id=case.id,
        result=CallResult(intent=CallIntent.DECLINED, final_node_id="declined"),
        _client=client,
    )

    await session.refresh(case)
    assert case.status == CaseStatus.DECLINED
    assert case.attempt_count == case.max_attempts  # contact suppressed
    assert client.calls == 0


async def test_hardship_is_not_a_refusal(session):
    """retry_later must leave the case workable, unlike declined."""
    case = await seed(session)
    call_id = await open_call_record(recovery_case_id=case.id, room_name="recovery-1")

    await finalise_call(
        voice_call_id=call_id,
        recovery_case_id=case.id,
        result=CallResult(intent=CallIntent.RETRY_LATER, final_node_id="pay_later"),
    )

    await session.refresh(case)
    assert case.status == CaseStatus.IN_PROGRESS
    assert case.attempt_count < case.max_attempts


async def test_the_call_is_audited(session):
    case = await seed(session)
    call_id = await open_call_record(recovery_case_id=case.id, room_name="recovery-1")

    await finalise_call(
        voice_call_id=call_id,
        recovery_case_id=case.id,
        result=CallResult(intent=CallIntent.WRONG_NUMBER, final_node_id="wrong_number"),
    )

    from app.models import RecoveryAction

    actions = (
        await session.execute(
            select(RecoveryAction.action_type).where(
                RecoveryAction.recovery_case_id == case.id
            )
        )
    ).scalars().all()
    assert ActionType.VOICE_CALL in actions
    assert ActionType.STOPPED in actions


async def test_finalising_a_demo_run_is_a_no_op(session):
    await finalise_call(
        voice_call_id=None,
        recovery_case_id=None,
        result=CallResult(intent=CallIntent.RETRY_NOW),
    )
    count = (await session.execute(select(func.count(VoiceCall.id)))).scalar_one()
    assert count == 0


async def test_a_failed_link_does_not_lose_the_call_record(session):
    """The customer still said yes; the next tick can retry the link."""

    class Broken:
        async def create_payment_link(self, payload):
            raise RuntimeError("razorpay 500")

    case = await seed(session)
    call_id = await open_call_record(recovery_case_id=case.id, room_name="recovery-1")

    await finalise_call(
        voice_call_id=call_id,
        recovery_case_id=case.id,
        result=CallResult(intent=CallIntent.RETRY_NOW, final_node_id="pay_now"),
        _client=Broken(),
    )

    call = await session.get(VoiceCall, call_id)
    assert call.detected_intent == CallIntent.RETRY_NOW
    assert call.status == CallStatus.COMPLETED
