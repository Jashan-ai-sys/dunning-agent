"""Send a correctly-signed webhook to a running instance.

Razorpay's dashboard can only deliver to a public URL, so this is the fast local
loop for exercising signature checks, dedupe and storage.

    uv run python -m scripts.send_test_webhook --event payment.failed --invoice-id inv_XXX

Note: the payment.failed and payment.captured handlers resolve the invoice
through the live Razorpay API. Pass an invoice id that really exists in your
test account, or pass --invoice-id "" to send a one-off (non-subscription)
failure, which exercises the transport without needing an API call.
"""

import argparse
import hashlib
import hmac
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.payloads import (  # noqa: E402
    payment_captured_event,
    payment_failed_event,
    subscription_charged_event,
    subscription_halted_event,
    subscription_pending_event,
)

BUILDERS = {
    "payment.failed": payment_failed_event,
    "payment.captured": payment_captured_event,
    "subscription.pending": subscription_pending_event,
    "subscription.halted": subscription_halted_event,
    "subscription.charged": subscription_charged_event,
}


def _read_secret() -> str:
    secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET")
    if secret:
        return secret
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "RAZORPAY_WEBHOOK_SECRET":
                return value.strip()
    raise SystemExit("RAZORPAY_WEBHOOK_SECRET is not set (env or .env)")


def build_payload(args: argparse.Namespace) -> dict:
    builder = BUILDERS[args.event]
    if args.event in ("payment.failed", "payment.captured"):
        return builder(payment_id=args.payment_id, invoice_id=args.invoice_id or None)
    if args.event == "subscription.charged":
        return builder(payment_id=args.payment_id, subscription_id=args.subscription_id)
    return builder(subscription_id=args.subscription_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", choices=sorted(BUILDERS), default="payment.failed")
    parser.add_argument("--url", default="http://localhost:8000/webhooks/razorpay")
    parser.add_argument("--event-id", default="evt_local_1", help="X-Razorpay-Event-Id")
    parser.add_argument("--payment-id", default="pay_LOCAL1")
    parser.add_argument("--subscription-id", default="sub_1")
    parser.add_argument("--invoice-id", default="inv_1", help='pass "" for a one-off payment')
    parser.add_argument("--tamper", action="store_true", help="send a bad signature")
    args = parser.parse_args()

    body = json.dumps(build_payload(args)).encode()
    secret = _read_secret() if not args.tamper else "wrong_secret"
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    request = urllib.request.Request(
        args.url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Signature": signature,
            "X-Razorpay-Event-Id": args.event_id,
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            print(response.status, response.read().decode())
    except urllib.error.HTTPError as exc:
        print(exc.code, exc.read().decode())


if __name__ == "__main__":
    main()
