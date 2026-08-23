"""LiveKit contact channel: the orchestrator's way of placing a call.

Implements the same ``ContactChannel`` protocol as ``LoggingChannel``, so
swapping it in is a one-line change in the worker and touches neither the policy
nor the orchestration loop.

Dispatch order matters. The agent is dispatched into the room *before* the phone
rings, so there is never a moment where a customer has answered and nothing is
listening. The agent itself places the SIP leg.
"""

import json
import logging

from livekit import api

from app.channels import ContactResult
from app.config import get_settings
from app.models import Customer, RecoveryCase
from app.voice.spoken import spoken_amount

logger = logging.getLogger(__name__)


class LiveKitChannel:
    """Places real outbound PSTN calls through LiveKit SIP."""

    name = "livekit"

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.livekit_configured:
            raise RuntimeError("LiveKit is not configured: set LIVEKIT_URL/API_KEY/API_SECRET")
        if not settings.livekit_sip_trunk_id:
            raise RuntimeError(
                "LIVEKIT_SIP_TRUNK_ID is not set. LiveKit Cloud alone cannot reach the "
                "phone network: create an outbound trunk against a SIP provider first."
            )
        self._settings = settings

    def _client(self) -> api.LiveKitAPI:
        return api.LiveKitAPI(
            url=self._settings.livekit_url,
            api_key=self._settings.livekit_api_key,
            api_secret=self._settings.livekit_api_secret,
        )

    @staticmethod
    def room_name(case: RecoveryCase) -> str:
        return f"recovery-{case.id}"

    def _metadata(self, case: RecoveryCase, customer: Customer) -> str:
        return json.dumps(
            {
                "recovery_case_id": case.id,
                "phone": customer.phone,
                "customer_name": customer.name or "there",
                "preferred_language": customer.preferred_language,
                "company_name": self._settings.company_name,
                # Already in spoken form -- "5 लाख रुपये", not "500000".
                "amount_spoken": spoken_amount(
                    case.original_amount, customer.preferred_language
                ),
                "amount_paise": case.original_amount,
                "failure_reason": case.failure_reason or "the bank declined it",
            }
        )

    async def initiate(self, case: RecoveryCase, customer: Customer) -> ContactResult:
        room = self.room_name(case)
        client = self._client()
        try:
            dispatch = await client.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=self._settings.livekit_agent_name,
                    room=room,
                    metadata=self._metadata(case, customer),
                )
            )
        finally:
            await client.aclose()

        logger.info("dispatched agent to %s for case %s", room, case.id)
        return ContactResult(
            channel=self.name,
            reference=room,
            detail={
                "dispatch_id": getattr(dispatch, "id", None),
                "language": customer.preferred_language,
                "amount": case.original_amount,
                "placed": True,
            },
        )
