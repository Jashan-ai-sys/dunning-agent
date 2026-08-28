"""Answering machine detection: the verdict, and what it does to the case.

Twilio decides whether a person or a recording answered and posts the result
here. Two things follow from it.

The call ends. There is no value in an agent explaining a failed mandate to a
voicemail greeting, and a recording of our own script sitting in someone's
inbox is worse than silence.

The attempt is refunded. ``attempt_count`` is spent when the call is *placed*,
because the tick that placed it has to commit long before Twilio finishes
listening. A voicemail is not a contact, so the count is given back -- without
that, three voicemails close a case as ``max_attempts_reached`` having never
reached a person, which is the failure this module exists to prevent.
"""

import hashlib
import hmac
from base64 import b64encode

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import ActionType
from app.models import RecoveryCase, VoiceCall
from app.store import log_action

#: Twilio's verdicts. Anything starting ``machine_`` is a recording; ``fax``
#: is not a person either. ``unknown`` means it could not tell, and we treat
#: that as human -- hanging up on a real customer is the worse mistake.
MACHINE_VERDICTS = frozenset(
    {"machine_start", "machine_end_beep", "machine_end_silence", "machine_end_other", "fax"}
)


def is_machine(answered_by: str | None) -> bool:
    return (answered_by or "").strip().lower() in MACHINE_VERDICTS


def verify_twilio_signature(url: str, params: dict[str, str], signature: str | None,
                            auth_token: str) -> bool:
    """Validate ``X-Twilio-Signature``.

    This endpoint can end a call in progress, so it is not left open. Twilio
    signs the full URL with every POST parameter appended in sorted key order,
    HMAC-SHA1 under the account's auth token, base64 encoded.
    """
    if not signature or not auth_token:
        return False
    payload = url + "".join(f"{k}{params[k]}" for k in sorted(params))
    digest = hmac.new(auth_token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha1)
    return hmac.compare_digest(b64encode(digest.digest()).decode(), signature)


async def apply_amd_result(
    session: AsyncSession, call_sid: str, answered_by: str | None
) -> RecoveryCase | None:
    """Record the verdict against the call, and refund the attempt if it was a
    machine. Returns the case when the call should be hung up."""
    call = (
        await session.execute(select(VoiceCall).where(VoiceCall.call_id == call_sid))
    ).scalar_one_or_none()
    if call is None:
        return None

    case = await session.get(RecoveryCase, call.recovery_case_id)
    if case is None:
        return None

    machine = is_machine(answered_by)
    await log_action(
        session,
        case,
        ActionType.VOICE_CALL,
        {"voice_call_id": call.id, "answered_by": answered_by, "machine": machine},
    )
    if not machine:
        return None

    call.status = "no_answer"
    call.detected_intent = None
    # Give the attempt back: it was spent when the call was placed.
    case.attempt_count = max((case.attempt_count or 1) - 1, 0)
    return case
