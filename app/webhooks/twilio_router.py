"""Twilio's answering-machine verdict.

Separate router from the Razorpay one: different provider, different signature
scheme, different failure modes. Mounted on the webhook service because that is
the one already reachable from the public internet.
"""

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.voice.amd import apply_amd_result, verify_twilio_signature

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks/twilio", tags=["twilio"])


@router.post("/amd", status_code=status.HTTP_200_OK)
async def answering_machine_detection(
    request: Request,
    session: AsyncSession = Depends(get_session),
    x_twilio_signature: str | None = Header(default=None),
) -> dict[str, str]:
    """Twilio has decided who answered.

    Verified before anything else: this endpoint can end a call in progress, so
    an unauthenticated caller must not be able to reach the hang-up.
    """
    settings = get_settings()
    form = await request.form()
    params = {k: str(v) for k, v in form.items()}

    # Twilio signs the URL it was configured with, which is the public one --
    # not whatever Cloud Run reports after proxying.
    signed_url = settings.twilio_amd_callback_url or str(request.url)
    if not verify_twilio_signature(
        signed_url, params, x_twilio_signature, settings.twilio_auth_token
    ):
        logger.warning("rejected AMD callback with an invalid signature")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid signature")

    call_sid = params.get("CallSid")
    answered_by = params.get("AnsweredBy")
    if not call_sid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="missing CallSid")

    logger.info("AMD verdict for %s: %s", call_sid, answered_by)
    case = await apply_amd_result(session, call_sid, answered_by)
    await session.commit()

    if case is None:
        return {"status": "recorded"}

    # A recording answered. End the call rather than talk to it.
    from app.voice.telephony import TwilioChannel

    try:
        await TwilioChannel().hang_up(call_sid)
    except Exception:  # noqa: BLE001 - the verdict is already recorded
        logger.exception("could not hang up %s after machine detection", call_sid)
    return {"status": "hung_up"}
