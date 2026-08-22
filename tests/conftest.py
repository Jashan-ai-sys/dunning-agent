import os

# Settings are cached at first import, so the test environment must be in place
# before anything under app/ is imported.
TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://recovery:recovery@localhost:5433/recovery_test",
)
TEST_WEBHOOK_SECRET = "test_webhook_secret"

os.environ["DATABASE_URL"] = TEST_DB_URL
os.environ["RAZORPAY_WEBHOOK_SECRET"] = TEST_WEBHOOK_SECRET
os.environ["RAZORPAY_KEY_ID"] = "rzp_test_dummy"
os.environ["RAZORPAY_KEY_SECRET"] = "dummy_secret"

import asyncpg  # noqa: E402
import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db import SessionLocal, engine  # noqa: E402
from app.models import Base  # noqa: E402

TABLES = [
    "recovery_actions",
    "recovery_cases",
    "payments",
    "subscriptions",
    "customers",
    "webhook_events",
]


async def _ensure_test_database() -> None:
    """Create the test database if missing. Skips (rather than fails) when
    Postgres is unreachable, so the pure unit tests still run on a bare
    checkout with no Docker."""
    dsn = TEST_DB_URL.replace("postgresql+asyncpg://", "postgresql://")
    admin_dsn, db_name = dsn.rsplit("/", 1)
    try:
        conn = await asyncpg.connect(f"{admin_dsn}/postgres")
    except Exception as exc:  # noqa: BLE001 - any connection failure means "no DB"
        pytest.skip(f"Postgres not reachable: {exc}")
    try:
        if not await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name):
            await conn.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await conn.close()


@pytest.fixture
async def session():
    """A clean database and one session. Requesting this fixture is what marks
    a test as needing Postgres.

    Everything is function-scoped and the engine is disposed afterwards:
    asyncpg connections are bound to the loop that opened them, so pooling them
    across pytest-asyncio's per-test loops would fail.
    """
    await _ensure_test_database()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE"))
    try:
        async with SessionLocal() as s:
            yield s
    finally:
        await engine.dispose()


class FakeRazorpayClient:
    """Stands in for the REST client so handler tests make no network calls."""

    def __init__(
        self,
        invoices: dict | None = None,
        subscriptions: dict | None = None,
        customers: dict | None = None,
    ) -> None:
        self.invoices = invoices or {}
        self.subscriptions = subscriptions or {}
        self.customers = customers or {}
        self.calls: list[tuple[str, str]] = []

    async def fetch_invoice(self, invoice_id: str) -> dict:
        self.calls.append(("invoice", invoice_id))
        return self.invoices[invoice_id]

    async def fetch_subscription(self, subscription_id: str) -> dict:
        self.calls.append(("subscription", subscription_id))
        return self.subscriptions[subscription_id]

    async def fetch_customer(self, customer_id: str) -> dict:
        self.calls.append(("customer", customer_id))
        return self.customers[customer_id]


@pytest.fixture
def fake_client():
    from tests.payloads import customer_entity, invoice_entity, subscription_entity

    return FakeRazorpayClient(
        invoices={"inv_1": invoice_entity()},
        subscriptions={"sub_1": subscription_entity()},
        customers={"cust_1": customer_entity()},
    )
