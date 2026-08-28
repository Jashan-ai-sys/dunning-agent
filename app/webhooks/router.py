import hashlib
import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.events import announce_event
from app.razorpay.signature import verify_webhook_signature
from app.webhooks.processor import process_event, record_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/razorpay", status_code=status.HTTP_200_OK)
async def razorpay_webhook(
    request: Request,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    x_razorpay_signature: str | None = Header(default=None),
    x_razorpay_event_id: str | None = Header(default=None),
) -> dict[str, str]:
    raw_body = await request.body()
    settings = get_settings()

    secret = settings.razorpay_webhook_secret
    if not verify_webhook_signature(raw_body, x_razorpay_signature, secret):
        # Do not echo any detail: an attacker probing the endpoint learns nothing.
        logger.warning("rejected webhook with invalid signature")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid signature")

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="malformed body"
        ) from None

    event_name = payload.get("event")
    if not event_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="missing event")

    # Razorpay always sends the header; fall back to a body digest so a missing
    # header degrades to weaker dedupe rather than dropping the event.
    event_id = x_razorpay_event_id or hashlib.sha256(raw_body).hexdigest()

    event = await record_event(
        session, razorpay_event_id=event_id, event_name=event_name, payload=payload
    )
    if event is None:
        return {"status": "duplicate"}

    # Announce it, then keep the in-process fallback. Three paths now lead to
    # the same envelope -- push, this background task, and the cron sweep --
    # and all three are safe because process_event returns early once
    # processed_at is set. Belt and braces is the right posture for the only
    # notification Razorpay will ever send us about money.
    await announce_event(event.id, event_name)
    background.add_task(process_event, event.id)
    return {"status": "accepted"}
