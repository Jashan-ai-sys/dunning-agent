"""The push subscriber.

Pub/Sub calls this within about a second of an envelope being stored, instead
of the cron sweep finding it up to five minutes later. The sweep still runs:
push is an accelerator on a mechanism that already worked, and everything here
is written so that losing push costs latency and nothing else.

Authentication is OIDC, not a shared secret. Pub/Sub push cannot send custom
headers, so the only alternatives are a token in the URL -- which lands in
every access log -- or verifying the signed token Google attaches. This
endpoint dispatches handlers that create payment links and place calls, so it
gets the real one.
"""

import base64
import logging

from fastapi import APIRouter, Header, HTTPException, Request, status

from app.config import get_settings
from app.webhooks.processor import process_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks/pubsub", tags=["pubsub"])


def _verify_push_token(authorization: str | None) -> None:
    """Confirm the request really came from our Pub/Sub subscription."""
    settings = get_settings()
    expected_sa = settings.pubsub_push_service_account.strip()
    if not expected_sa:
        # Refuse rather than run unauthenticated. A push endpoint that
        # dispatches handlers is not something to leave open because a setting
        # was forgotten.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="push delivery is not configured",
        )
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthenticated")

    token = authorization.split(" ", 1)[1]
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token

        claims = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            audience=settings.pubsub_push_audience.strip() or None,
        )
    except Exception:  # noqa: BLE001 - any verification failure is a rejection
        logger.warning("rejected a push with an unverifiable token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token"
        ) from None

    if claims.get("email") != expected_sa or not claims.get("email_verified"):
        logger.warning("rejected a push signed by %s", claims.get("email"))
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="wrong principal")


@router.post("/event", status_code=status.HTTP_204_NO_CONTENT)
async def pubsub_push(request: Request, authorization: str | None = Header(default=None)):
    """Process one announced webhook envelope.

    Returns 204 on anything that must not be redelivered, including a message
    we cannot parse: Pub/Sub retries until acknowledged, and a permanently
    malformed message would otherwise be redelivered forever.
    """
    _verify_push_token(authorization)

    body = await request.json()
    message = (body or {}).get("message") or {}
    raw = message.get("data")
    if not raw:
        logger.warning("push carried no data; acknowledging to stop redelivery")
        return

    try:
        webhook_event_pk = int(base64.b64decode(raw).decode("utf-8"))
    except Exception:  # noqa: BLE001 - unparseable is not retryable
        logger.warning("push carried an unreadable id %r; acknowledging", raw)
        return

    # process_event returns early when the envelope is already processed, so a
    # redelivery racing the cron sweep is a no-op rather than a double dispatch.
    await process_event(webhook_event_pk)
