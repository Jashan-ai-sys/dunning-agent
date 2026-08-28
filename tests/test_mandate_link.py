"""Sending the link that re-authorises a mandate.

A payment link settles what is owed. It does not bring a revoked mandate back,
so the next cycle fails identically -- which is why CUSTOMER_INSTRUMENT earns a
call. Until this existed the agent could diagnose that and had no tool to act
on it.
"""


from app.config import Settings
from app.constants import ActionType, CaseStatus
from app.mandate import send_mandate_link
from app.models import Customer, RecoveryCase

SETTINGS = Settings(company_name="Acme")
AUTH_URL = "https://rzp.io/rzp/7Egyhyh"


class FakeClient:
    def __init__(self, short_url: str | None = AUTH_URL, status: str = "halted"):
        self._short_url = short_url
        self._status = status
        self.fetched: list[str] = []

    async def fetch_subscription(self, subscription_id: str) -> dict:
        self.fetched.append(subscription_id)
        body = {"id": subscription_id, "status": self._status}
        if self._short_url:
            body["short_url"] = self._short_url
        return body


class FakeSms:
    def __init__(self, sid: str | None = "SM123"):
        self.sid = sid
        self.sent: list[tuple[str, str]] = []

    async def __call__(self, to: str, body: str) -> str | None:
        self.sent.append((to, body))
        return self.sid


async def seed(session, **overrides):
    customer = Customer(
        **{
            "razorpay_customer_id": "cust_m",
            "phone": "+919000000000",
            "email": "m@example.com",
            **overrides.pop("customer", {}),
        }
    )
    session.add(customer)
    case = RecoveryCase(
        **{
            "razorpay_payment_id": "pay_m",
            "razorpay_customer_id": "cust_m",
            "razorpay_subscription_id": "sub_m",
            "original_amount": 49_900,
            "status": CaseStatus.IN_PROGRESS,
            **overrides,
        }
    )
    session.add(case)
    await session.commit()
    await session.refresh(case)
    await session.refresh(customer)
    return case, customer


async def actions(session, case_id):
    from sqlalchemy import select

    from app.models import RecoveryAction

    rows = await session.execute(
        select(RecoveryAction.action_type, RecoveryAction.metadata_json)
        .where(RecoveryAction.recovery_case_id == case_id)
        .order_by(RecoveryAction.id)
    )
    return list(rows)


async def test_the_subscriptions_own_authorisation_link_is_sent(session):
    case, customer = await seed(session)
    client, sms = FakeClient(), FakeSms()

    url = await send_mandate_link(
        session, case, customer, client, settings=SETTINGS, sms=sms
    )
    await session.commit()

    assert url == AUTH_URL
    assert client.fetched == ["sub_m"], "the link is fetched live, not stored"
    to, body = sms.sent[0]
    assert to == "+919000000000"
    assert AUTH_URL in body
    assert "Acme" in body


async def test_it_is_recorded_in_the_audit_trail(session):
    case, customer = await seed(session)

    await send_mandate_link(
        session, case, customer, FakeClient(), settings=SETTINGS, sms=FakeSms()
    )
    await session.commit()

    kinds = [a for a, _ in await actions(session, case.id)]
    assert ActionType.MANDATE_LINK_SENT in kinds
    meta = dict(await actions(session, case.id))[ActionType.MANDATE_LINK_SENT]
    assert meta["short_url"] == AUTH_URL
    assert meta["message_sid"] == "SM123"


async def test_a_case_with_no_subscription_sends_nothing(session):
    """A one-off checkout has no mandate to re-authorise."""
    case, customer = await seed(session, razorpay_subscription_id=None)
    sms = FakeSms()

    assert await send_mandate_link(
        session, case, customer, FakeClient(), settings=SETTINGS, sms=sms
    ) is None
    assert sms.sent == []


async def test_a_wrong_number_is_not_texted(session):
    """Somebody else answered that phone; do not send them a billing link."""
    case, customer = await seed(session, customer={"phone_is_wrong": True})
    sms = FakeSms()

    assert await send_mandate_link(
        session, case, customer, FakeClient(), settings=SETTINGS, sms=sms
    ) is None
    assert sms.sent == []


async def test_a_subscription_without_a_link_reports_failure(session):
    case, customer = await seed(session)

    assert await send_mandate_link(
        session, case, customer, FakeClient(short_url=None), settings=SETTINGS, sms=FakeSms()
    ) is None


async def test_a_failed_sms_is_not_reported_as_sent(session):
    """The agent is told never to claim it sent a link it did not, so failure
    has to stay distinguishable from success."""
    case, customer = await seed(session)

    result = await send_mandate_link(
        session, case, customer, FakeClient(), settings=SETTINGS, sms=FakeSms(sid=None)
    )
    await session.commit()

    assert result is None
    kinds = [a for a, _ in await actions(session, case.id)]
    assert ActionType.MANDATE_LINK_SENT not in kinds
