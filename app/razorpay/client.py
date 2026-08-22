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

    async def fetch_invoice(self, invoice_id: str) -> dict[str, Any]:
        """Invoices carry the subscription_id and customer_id a payment lacks."""
        return await self._get(f"/invoices/{invoice_id}")

    async def fetch_subscription(self, subscription_id: str) -> dict[str, Any]:
        return await self._get(f"/subscriptions/{subscription_id}")

    async def fetch_customer(self, customer_id: str) -> dict[str, Any]:
        return await self._get(f"/customers/{customer_id}")
