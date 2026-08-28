"""Answering machine detection: the verdict, the signature, and the refund."""

import hashlib
import hmac
from base64 import b64encode

import pytest
from sqlalchemy import select

from app.constants import ActionType, CaseStatus
from app.models import Customer, RecoveryAction, RecoveryCase, VoiceCall
from app.voice.amd import apply_amd_result, is_machine, verify_twilio_signature

TOKEN = "test_auth_token"
URL = "https://example.invalid/webhooks/twilio/amd"


def sign(url: str, params: dict[str, str], token: str = TOKEN) -> str:
    payload = url + "".join(f"{k}{params[k]}" for k in sorted(params))
    return b64encode(
        hmac.new(token.encode(), payload.encode(), hashlib.sha1).digest()
    ).decode()


# --- verdicts --------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict",
    ["machine_start", "machine_end_beep", "machine_end_silence", "machine_end_other", "fax"],
)
def test_recordings_are_machines(verdict):
    assert is_machine(verdict) is True


@pytest.mark.parametrize("verdict", ["human", "unknown", None, "", "HUMAN"])
def test_anything_else_is_treated_as_a_person(verdict):
    """`unknown` counts as human on purpose: hanging up on a real customer is
    a worse mistake than talking to a recording."""
    assert is_machine(verdict) is False


def test_the_verdict_is_case_insensitive():
    assert is_machine("MACHINE_START") is True


# --- signature -------------------------------------------------------------


def test_a_correctly_signed_callback_is_accepted():
    params = {"CallSid": "CA123", "AnsweredBy": "human"}
    assert verify_twilio_signature(URL, params, sign(URL, params), TOKEN)


def test_a_forged_signature_is_rejected():
    """This endpoint can end a call in progress, so it is not left open."""
    params = {"CallSid": "CA123", "AnsweredBy": "machine_start"}
    assert not verify_twilio_signature(URL, params, "not-the-signature", TOKEN)


def test_tampering_with_a_parameter_invalidates_the_signature():
    params = {"CallSid": "CA123", "AnsweredBy": "human"}
    signature = sign(URL, params)
    assert not verify_twilio_signature(URL, {**params, "AnsweredBy": "machine_start"},
                                       signature, TOKEN)


def test_parameter_order_does_not_matter():
    """Twilio sorts by key; a dict in another order must still verify."""
    params = {"AnsweredBy": "human", "CallSid": "CA123"}
    assert verify_twilio_signature(URL, dict(reversed(list(params.items()))),
                                   sign(URL, params), TOKEN)


@pytest.mark.parametrize("missing", [None, ""])
def test_a_missing_signature_is_rejected(missing):
    assert not verify_twilio_signature(URL, {"CallSid": "CA1"}, missing, TOKEN)


def test_no_auth_token_means_nothing_verifies():
    params = {"CallSid": "CA123"}
    assert not verify_twilio_signature(URL, params, sign(URL, params), "")


# --- applying the verdict --------------------------------------------------


async def seed(session, *, attempt_count: int = 1) -> VoiceCall:
    session.add(Customer(razorpay_customer_id="cust_amd", phone="+919000000000"))
    case = RecoveryCase(
        razorpay_payment_id="pay_amd", razorpay_customer_id="cust_amd",
        original_amount=49_900, status=CaseStatus.IN_PROGRESS,
        attempt_count=attempt_count, max_attempts=3,
    )
    session.add(case)
    await session.flush()
    call = VoiceCall(recovery_case_id=case.id, provider="twilio",
                     call_id="CA_AMD_1", status="initiated")
    session.add(call)
    await session.commit()
    await session.refresh(call)
    return call


async def test_a_voicemail_refunds_the_attempt(session):
    """The bug this exists for: attempt_count is spent when the call is placed,
    long before Twilio finishes listening. Three voicemails would otherwise
    close the case as max_attempts_reached without ever reaching a person."""
    await seed(session, attempt_count=1)

    case = await apply_amd_result(session, "CA_AMD_1", "machine_start")
    await session.commit()

    assert case is not None, "the caller needs the case back to hang up"
    await session.refresh(case)
    assert case.attempt_count == 0


async def test_a_human_keeps_the_attempt(session):
    call = await seed(session, attempt_count=1)

    case = await apply_amd_result(session, "CA_AMD_1", "human")
    await session.commit()

    assert case is None, "nothing to hang up"
    reloaded = await session.get(RecoveryCase, call.recovery_case_id)
    assert reloaded.attempt_count == 1


async def test_the_verdict_is_always_recorded_either_way(session):
    call = await seed(session)

    await apply_amd_result(session, "CA_AMD_1", "human")
    await session.commit()

    rows = await session.execute(
        select(RecoveryAction.metadata_json)
        .where(RecoveryAction.recovery_case_id == call.recovery_case_id)
        .where(RecoveryAction.action_type == ActionType.VOICE_CALL)
    )
    meta = rows.scalar_one()
    assert meta["answered_by"] == "human"
    assert meta["machine"] is False


async def test_the_refund_cannot_go_negative(session):
    await seed(session, attempt_count=0)

    case = await apply_amd_result(session, "CA_AMD_1", "fax")
    await session.commit()

    await session.refresh(case)
    assert case.attempt_count == 0


async def test_a_verdict_for_an_unknown_call_is_ignored(session):
    """Twilio retries callbacks; one for a call we never recorded is not an
    error worth failing on."""
    assert await apply_amd_result(session, "CA_NEVER_SEEN", "machine_start") is None
