"""Run the dunning conversation in a LiveKit room you join from a browser.

No SIP trunk, no phone number, no DLT registration. The agent, the conversation
graph, the Hinglish prompts and the intent capture are all identical to a real
call -- only the transport differs, and a recording cannot tell the difference.

Two terminals:

    # 1. the agent worker
    uv run --group voice python -m app.voice.agent dev

    # 2. this, which dispatches it into a room and prints a join token
    uv run --group voice python -m scripts.demo_call

Then open https://agents-playground.livekit.io, paste the URL and token, and
talk to it.

Pass --case <id> to use a real recovery case from the database; otherwise a
representative one is made up so the demo runs on an empty database.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from livekit import api
from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal, engine
from app.models import Customer, RecoveryCase
from app.voice.spoken import spoken_amount

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PLAYGROUND = "https://agents-playground.livekit.io"


async def context_from_case(case_id: int) -> dict:
    async with SessionLocal() as session:
        case = await session.get(RecoveryCase, case_id)
        if case is None:
            raise SystemExit(f"no recovery case with id {case_id}")
        customer = (
            await session.execute(
                select(Customer).where(
                    Customer.razorpay_customer_id == case.razorpay_customer_id
                )
            )
        ).scalar_one_or_none()
        return {
            "recovery_case_id": case.id,
            "customer_name": (customer.name if customer else None) or "there",
            "preferred_language": customer.preferred_language if customer else "hinglish",
            "amount_spoken": spoken_amount(case.original_amount, "hi"),
            "failure_reason": case.failure_reason or "the bank declined it",
        }


def sample_context() -> dict:
    return {
        "recovery_case_id": 0,
        "customer_name": "Asha",
        "preferred_language": "hi",
        "amount_spoken": "499 रुपये",
        "failure_reason": "your card had insufficient funds",
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, help="use a real recovery case id")
    parser.add_argument("--room", default="dunning-demo")
    parser.add_argument("--identity", default="demo-listener")
    parser.add_argument("--name", help="override the customer name")
    parser.add_argument("--amount-paise", type=int, help="override the amount, in paise")
    parser.add_argument("--reason", help="override the failure reason")
    parser.add_argument(
        "--language", choices=["hi", "hinglish", "en"], help="override the language"
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.livekit_configured:
        raise SystemExit("LiveKit is not configured; set LIVEKIT_URL/API_KEY/API_SECRET")

    context = await context_from_case(args.case) if args.case else sample_context()

    # Overrides let the demo speak a real case pulled from Razorpay even when the
    # row itself lives in the production database rather than this one.
    if args.name:
        context["customer_name"] = args.name
    if args.language:
        context["preferred_language"] = args.language
    if args.reason:
        context["failure_reason"] = args.reason
    if args.amount_paise:
        context["amount_spoken"] = spoken_amount(
            args.amount_paise, context.get("preferred_language", "hi")
        )
    # No "phone" key, so the agent skips the SIP leg entirely and just waits in
    # the room for a browser participant.
    context["company_name"] = settings.company_name

    lk = api.LiveKitAPI(
        url=settings.livekit_url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
    )
    try:
        await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=settings.livekit_agent_name,
                room=args.room,
                metadata=json.dumps(context),
            )
        )
    finally:
        await lk.aclose()
        await engine.dispose()

    token = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(args.identity)
        .with_name("Demo listener")
        .with_grants(api.VideoGrants(room_join=True, room=args.room))
        .to_jwt()
    )

    # A self-contained page, so joining does not depend on a third-party
    # playground UI that moves around.
    template = Path(__file__).with_name("join_template.html").read_text(encoding="utf-8")
    page = (
        template.replace("__URL__", settings.livekit_url)
        .replace("__TOKEN__", token)
        .replace("__ROOM__", args.room)
        .replace("__AMOUNT__", context["amount_spoken"])
        .replace("__CUSTOMER__", context["customer_name"])
    )
    out = Path(__file__).resolve().parents[1] / "join.html"
    out.write_text(page, encoding="utf-8")

    print(f"dispatched '{settings.livekit_agent_name}' into room '{args.room}'")
    print(f"  customer : {context['customer_name']}")
    print(f"  amount   : {context['amount_spoken']}")
    print(f"  language : {context['preferred_language']}")
    print()
    print()
    print("OPEN THIS FILE IN YOUR BROWSER, then click Connect:")
    print(f"  {out}")
    print()
    print(f"(or {PLAYGROUND} with URL {settings.livekit_url} and the token in that file)")
    print()
    print("If nothing answers, the agent worker is not running. Start it with:")
    print("  uv run --group voice python -m app.voice.agent dev")


if __name__ == "__main__":
    asyncio.run(main())
