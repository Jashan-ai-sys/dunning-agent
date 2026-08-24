"""Inspect recovery cases, and send a real payment link for one.

    uv run python -m app.recover --list
    uv run python -m app.recover --case 3

Runs against whatever DATABASE_URL points at, so the same command works locally
and as a Cloud Run Job against Cloud SQL. That matters here: the cases opened by
genuine Razorpay webhooks live in production, not on a laptop.

``--case`` creates a **real** Razorpay payment link through the live API and
records it on the case, exactly as a ``retry_now`` call would. It is the same
code path -- ``create_recovery_link`` -- so paying that link fires a real
``payment_link.paid`` webhook, which credits the case by ``reference_id`` and
closes it as recovered.

This exists to exercise the last untested link in the chain without needing a
human to hold a conversation first.

It lives under ``app/`` rather than ``scripts/`` because it has to run inside
the deployed container -- the cases opened by real webhooks are in Cloud SQL,
and ``scripts/`` is excluded from the image.
"""

import argparse
import asyncio
import sys

from sqlalchemy import select

from app.constants import CaseStatus
from app.db import SessionLocal, engine
from app.metrics import rupees
from app.models import Customer, RecoveryCase
from app.payment_links import create_recovery_link
from app.razorpay.client import RazorpayClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def list_cases() -> None:
    async with SessionLocal() as session:
        rows = await session.execute(
            select(RecoveryCase).order_by(RecoveryCase.id.desc()).limit(25)
        )
        cases = list(rows.scalars())
        if not cases:
            print("no recovery cases")
            return
        print(f"{'id':>4}  {'status':<12} {'source':<9} {'amount':>12}  payment link")
        for c in cases:
            link = c.payment_link_url or "-"
            print(
                f"{c.id:>4}  {c.status:<12} {c.source:<9} "
                f"{rupees(c.original_amount):>12}  {link}"
            )


async def send_link(case_id: int) -> None:
    async with SessionLocal() as session:
        case = await session.get(RecoveryCase, case_id)
        if case is None:
            raise SystemExit(f"no case {case_id}")
        if case.status == CaseStatus.RECOVERED:
            print(f"case {case_id} is already recovered; nothing to send")
            return

        customer = None
        if case.razorpay_customer_id:
            customer = (
                await session.execute(
                    select(Customer).where(
                        Customer.razorpay_customer_id == case.razorpay_customer_id
                    )
                )
            ).scalar_one_or_none()
        if customer is None:
            # Razorpay rejects an empty customer block; a bare object is fine.
            customer = Customer(razorpay_customer_id=case.razorpay_customer_id or "unknown")

        link = await create_recovery_link(session, case, customer, RazorpayClient())
        await session.commit()

        if link is None:
            print("no link was created")
            return
        print(f"case {case_id}: {rupees(case.original_amount)}")
        print(f"  payment_link_id {link.id}")
        print(f"  reference_id    {link.reference_id}")
        print(f"  PAY HERE        {link.short_url}")
        print(
            "\nPaying that fires payment_link.paid at the webhook endpoint, which\n"
            "credits this case by reference_id and closes it as recovered."
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show recent cases")
    parser.add_argument("--case", type=int, help="send a real payment link for this case")
    args = parser.parse_args()

    try:
        if args.case:
            await send_link(args.case)
        else:
            await list_cases()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
