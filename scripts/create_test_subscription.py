"""Create a real Razorpay test-mode subscription, so a genuine failure can reach us.

    uv run python -m scripts.create_test_subscription
    uv run python -m scripts.create_test_subscription --amount 499 --period daily

Why this exists
===============
A failed payment *link* carries no ``invoice_id``, so it resolves to no
subscription and our handler correctly opens no recovery case -- that is a
checkout failure, not subscription recovery. Only a subscription invoice
failure produces a case. To demonstrate the real loop we therefore need a real
subscription.

What it does, and what you do
=============================
This creates a plan and a subscription through the live test-mode API and
prints the authentication link. You open that link and authorise with a test
card. From then on Razorpay drives the billing cycle itself, and a failed
charge arrives at the webhook endpoint as a genuine ``payment.failed`` carrying
a real ``invoice_id``.

To make the charge fail, enter an **OTP shorter than 4 digits** on the checkout
page. That is Razorpay's documented test-mode failure lever.

Timing constraint worth knowing
===============================
Test-mode card tokens are valid for **3 days**, and a subsequent debit only
works inside that window.

That looks fatal at first, because Razorpay's shortest billing cycle is 7 days
(``period=daily`` is rejected below ``interval=7``). It is not: the **first**
charge happens at authorisation, so the demo does not wait for a cycle. Set this
up shortly before demoing and fail that first charge.

Nothing here writes to our database. The recovery case appears only when
Razorpay's webhook arrives, which is the point -- it proves the real path
rather than simulating it.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.razorpay.client import RazorpayClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


#: Razorpay enforces a minimum interval per period. Discovered from the API,
#: not the docs: period=daily with interval=1 is rejected with
#: "Interval provided is less than the minimum interval (7) allowed".
MIN_INTERVAL = {"daily": 7, "weekly": 1, "monthly": 1, "yearly": 1}


async def create_plan(
    client: RazorpayClient, *, amount_paise: int, period: str, interval: int
) -> dict:
    return await client._post(
        "/plans",
        {
            "period": period,
            "interval": interval,
            "item": {
                "name": "Dunning demo subscription",
                "amount": amount_paise,
                "currency": "INR",
                "description": "Recovery pipeline demonstration",
            },
            "notes": {"purpose": "buildathon-demo"},
        },
    )


async def create_customer(client: RazorpayClient, *, name: str, contact: str, email: str) -> dict:
    return await client._post(
        "/customers",
        {"name": name, "contact": contact, "email": email, "fail_existing": "0"},
    )


async def create_subscription(
    client: RazorpayClient, *, plan_id: str, total_count: int, customer_id: str | None
) -> dict:
    payload: dict = {
        "plan_id": plan_id,
        "total_count": total_count,
        "quantity": 1,
        # Razorpay emails/SMSes the auth link itself when this is on. We print
        # it too, so the demo does not depend on inbox delivery.
        "customer_notify": 1,
        "notes": {"purpose": "buildathon-demo"},
    }
    if customer_id:
        payload["customer_id"] = customer_id
    return await client._post("/subscriptions", payload)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amount", type=int, default=499, help="rupees per cycle")
    parser.add_argument(
        "--period",
        default="daily",
        choices=["daily", "weekly", "monthly", "yearly"],
        help="billing cycle. The first charge happens at authorisation, so the "
        "cycle length does not gate the demo",
    )
    parser.add_argument(
        "--interval",
        type=int,
        help="cycles per period; defaults to Razorpay's minimum for the period",
    )
    parser.add_argument("--count", type=int, default=12, help="total billing cycles")
    parser.add_argument("--name", default="Asha Rao")
    parser.add_argument("--contact", default="+919000000000")
    parser.add_argument("--email", default="asha@example.com")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.razorpay_key_id.startswith("rzp_test"):
        raise SystemExit(
            f"refusing to run against a non-test key ({settings.razorpay_key_id[:8]}...). "
            "This creates real subscriptions."
        )

    client = RazorpayClient()

    interval = args.interval or MIN_INTERVAL[args.period]
    print(f"creating plan: ₹{args.amount} every {interval} x {args.period} ...")
    plan = await create_plan(
        client, amount_paise=args.amount * 100, period=args.period, interval=interval
    )
    print(f"  plan_id {plan['id']}")

    print("creating customer ...")
    try:
        customer = await create_customer(
            client, name=args.name, contact=args.contact, email=args.email
        )
        customer_id = customer["id"]
        print(f"  customer_id {customer_id}")
    except Exception as exc:  # noqa: BLE001 - the subscription works without one
        print(f"  could not create a customer ({exc}); continuing without")
        customer_id = None

    print("creating subscription ...")
    subscription = await create_subscription(
        client, plan_id=plan["id"], total_count=args.count, customer_id=customer_id
    )

    print()
    print("=" * 70)
    print(f"  subscription : {subscription['id']}")
    print(f"  status       : {subscription.get('status')}")
    print(f"  amount       : ₹{args.amount} every {interval} x {args.period}")
    # Razorpay's hosted subscription page needs the feature enabled on the
    # account and returns "Hosted page is not available" when it is not. Render
    # Checkout.js locally instead -- same authorisation, no hosted dependency.
    template = Path(__file__).with_name("checkout_template.html").read_text(encoding="utf-8")
    page = (
        template.replace("__KEY_ID__", settings.razorpay_key_id)
        .replace("__SUBSCRIPTION__", subscription["id"])
        .replace("__AMOUNT__", str(args.amount))
        .replace("__COMPANY__", settings.company_name)
        .replace("__NAME__", args.name)
        .replace("__EMAIL__", args.email)
        .replace("__CONTACT__", args.contact)
    )
    checkout = Path(__file__).resolve().parents[1] / "authorise.html"
    checkout.write_text(page, encoding="utf-8")

    auth_link = subscription.get("short_url")
    print(f"\n  AUTHORISE — open this file in your browser:\n    {checkout}")
    if auth_link:
        print(f"\n  (Razorpay's hosted page, if enabled on the account:\n    {auth_link})")
    else:
        print("\n  no short_url in the response:")
        print(json.dumps(subscription, indent=2)[:600])
    print("=" * 70)
    print(
        "\nNext:\n"
        "  1. Open authorise.html and pay with a Razorpay test card.\n"
        "  2. To make a charge FAIL, enter an OTP shorter than 4 digits.\n"
        "  3. Razorpay then fires payment.failed with a real invoice_id at\n"
        "     the webhook endpoint, and a recovery case opens by itself.\n"
        "\nWatch it land:\n"
        "  gcloud logging read 'resource.labels.service_name=\"dunning-agent\"' \\\n"
        "    --project dunning-agent --freshness=10m --limit=20\n"
        "\nRemember: test card tokens expire after 3 days."
    )


if __name__ == "__main__":
    asyncio.run(main())
