from typing import Any

import httpx

from app.config import get_settings


class RazorpayError(RuntimeError):
    """Raised when the Razorpay API returns a non-2xx response."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"Razorpay API returned {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class RazorpayClient:
    """Minimal async client for the endpoints this service needs.

    The official ``razorpay`` SDK is synchronous (requests-based); calling it from
    the webhook path would block the event loop, so we talk to the REST API over
    httpx instead.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        settings = get_settings()
        self._base = settings.razorpay_api_base.rstrip("/")
        self._auth = (settings.razorpay_key_id, settings.razorpay_key_secret)
        self._timeout = timeout

    async def _get(self, path: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{self._base}{path}", auth=self._auth)
        if response.status_code >= 400:
            raise RazorpayError(response.status_code, response.text)
        return response.json()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base}{path}", json=payload, auth=self._auth)
        if response.status_code >= 400:
            raise RazorpayError(response.status_code, response.text)
        return response.json()

    async def create_payment_link(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a standard payment link. Returns the link entity, including
        ``id`` and ``short_url``."""
        return await self._post("/payment_links", payload)

    async def notify_payment_link(self, link_id: str, medium: str) -> dict[str, Any]:
        """Re-send an existing payment link over ``sms`` or ``email``.

        Razorpay delivers the link once, when it is created. A customer who is
        called a second time about the same debt therefore hears the agent say
        it is sending a link and receives nothing -- the case already has one,
        so nothing is created and nothing is sent. This is how you deliver the
        link you already made without minting a second one.
        """
        return await self._post(f"/payment_links/{link_id}/notify_by/{medium}", {})

    async def fetch_customer_tokens(self, customer_id: str) -> dict[str, Any]:
        """Every saved instrument for a customer, including e-mandates.

        This is how a mandate retry finds something to charge without us
        storing card or bank data ourselves: Razorpay holds the token, we only
        ever hold its id, and even that is fetched fresh rather than cached.
        """
        return await self._get(f"/customers/{customer_id}/tokens")

    async def create_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        """A recurring charge cannot be made against nothing -- Razorpay wants
        an order to hang it on, created immediately before the charge."""
        return await self._post("/orders", payload)

    async def create_recurring_payment(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Charge an existing mandate.

        The only call in this client that moves money without the customer
        touching anything, which is why nothing reaches it except through
        app.mandate and an explicitly enabled setting.
        """
        return await self._post("/payments/create/recurring", payload)

    async def fetch_invoice(self, invoice_id: str) -> dict[str, Any]:
        """Invoices carry the subscription_id and customer_id a payment lacks."""
        return await self._get(f"/invoices/{invoice_id}")

    async def fetch_subscription(self, subscription_id: str) -> dict[str, Any]:
        return await self._get(f"/subscriptions/{subscription_id}")

    async def fetch_customer(self, customer_id: str) -> dict[str, Any]:
        return await self._get(f"/customers/{customer_id}")
