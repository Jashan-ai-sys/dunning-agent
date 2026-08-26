"""Twilio contact channel: an outbound PSTN call without a SIP trunk.

Implements the same ``ContactChannel`` protocol as ``LoggingChannel`` and
``LiveKitChannel``, so choosing it is a configuration change rather than a code
change.

Why this exists alongside ``dispatch.LiveKitChannel``: LiveKit cannot reach the
phone network without an outbound SIP trunk negotiated against a carrier, which
is why the production worker has been stuck on ``LoggingChannel``. Twilio's
Media Streams need only a number and a public websocket, both of which a trial
account issues in minutes. This is the path that can actually be demonstrated.

The TwiML is sent inline with the dial request rather than served from a webhook
of ours. That buys two things: one public URL to run instead of two, and a case
context fixed at the moment we dial, so it cannot drift between placing the call
and Twilio asking us what to do with it.

Uses ``httpx`` directly rather than the ``twilio`` SDK. One authenticated POST
does not justify a dependency, and the SDK is synchronous by default -- which
this worker is not.
"""

import logging
from typing import Any
from xml.sax.saxutils import quoteattr

import httpx

from app.channels import ContactResult
from app.config import get_settings
from app.models import Customer, RecoveryCase
from app.voice.call_body import call_body

logger = logging.getLogger(__name__)

#: Twilio rejects inline TwiML above this with error 32018. We check first so
#: the failure names the cause rather than arriving as an opaque 400.
MAX_TWIML_CHARS = 4000

#: A dial that has not connected by now never will. Twilio keeps ringing and
#: bills for it, and a recovery case is better retried on the next tick than
#: left holding a line open.
RING_TIMEOUT_SECONDS = 30

#: Guards the HTTP call to Twilio, not the phone call. The orchestrator treats a
#: raised exception as a failed attempt and backs the case off, so hanging here
#: would stall the whole tick.
REQUEST_TIMEOUT_SECONDS = 15.0


class TwilioChannel:
    """Places real outbound PSTN calls through Twilio Media Streams."""

    name = "twilio"

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.twilio_configured:
            raise RuntimeError(
                "Twilio is not configured: set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
                "TWILIO_FROM_NUMBER and TWILIO_STREAM_URL"
            )
        if not settings.twilio_stream_url.startswith("wss://"):
            raise RuntimeError(
                "TWILIO_STREAM_URL must be a wss:// URL reachable from the public "
                f"internet, not {settings.twilio_stream_url!r}. Twilio dials out from "
                "its own network and cannot see localhost."
            )
        self._settings = settings

    @property
    def _calls_url(self) -> str:
        return (
            f"{self._settings.twilio_api_base}/Accounts/"
            f"{self._settings.twilio_account_sid}/Calls.json"
        )

    def _body(self, case: RecoveryCase, customer: Customer) -> dict[str, Any]:
        return call_body(case, customer, company_name=self._settings.company_name)

    def twiml(self, body: dict[str, Any]) -> str:
        """The instruction Twilio follows once the customer picks up.

        ``<Connect>`` rather than ``<Start>``: ``<Start>`` forks a copy of the
        audio to us and lets the call continue elsewhere, so the agent could
        hear the customer but never answer. ``<Connect>`` hands us the call.

        Every parameter here arrives back as ``runner_args.body`` on the agent
        side, which is the same shape LiveKit delivers as job metadata -- so
        both transports feed ``context_from_body`` an identical dict.
        """
        parameters = "".join(
            f"<Parameter name={quoteattr(key)} value={quoteattr(value)}/>"
            for key, value in stream_parameters(body).items()
        )
        markup = (
            "<?xml version='1.0' encoding='UTF-8'?>"
            "<Response><Connect>"
            f"<Stream url={quoteattr(self._settings.twilio_stream_url)}>"
            f"{parameters}"
            "</Stream></Connect></Response>"
        )
        if len(markup) > MAX_TWIML_CHARS:
            raise ValueError(
                f"TwiML is {len(markup)} characters, over Twilio's {MAX_TWIML_CHARS} "
                "limit (error 32018). Shorten the call body -- most likely the "
                "failure reason or the customer name."
            )
        return markup

    async def initiate(self, case: RecoveryCase, customer: Customer) -> ContactResult:
        if not customer.phone:
            raise RuntimeError(f"customer {customer.razorpay_customer_id} has no phone number")

        body = self._body(case, customer)
        payload = {
            "To": customer.phone,
            "From": self._settings.twilio_from_number,
            "Twiml": self.twiml(body),
            "Timeout": str(RING_TIMEOUT_SECONDS),
        }

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                self._calls_url,
                data=payload,
                auth=(self._settings.twilio_account_sid, self._settings.twilio_auth_token),
            )
        response.raise_for_status()
        call_sid = response.json().get("sid")

        logger.info("dialled %s for case %s as %s", customer.phone, case.id, call_sid)
        return ContactResult(
            channel=self.name,
            reference=call_sid,
            detail={
                "language": customer.preferred_language,
                "amount": case.original_amount,
                "placed": True,
            },
        )


def stream_parameters(body: dict[str, Any]) -> dict[str, str]:
    """Flatten a call body into TwiML ``<Parameter>`` values.

    Everything crossing this boundary becomes a string -- Twilio delivers custom
    parameters as strings and Pipecat hands them to the bot untouched. Booleans
    are therefore *omitted* when false rather than sent as ``"False"``: a
    non-empty string is truthy in Python, so ``"False"`` would arrive as True and
    silently tell the agent a subscription had been halted when it had not.
    """
    parameters: dict[str, str] = {}
    for key, value in body.items():
        if value is None or value is False:
            continue
        parameters[key] = "1" if value is True else str(value)
    return parameters


__all__ = ["MAX_TWIML_CHARS", "TwilioChannel", "stream_parameters"]
