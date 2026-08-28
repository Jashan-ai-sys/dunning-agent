"""MCP server exposing the recovery actions an agent may take.

    uv run --group pipecat python -m app.mcp_server        # stdio

Pipecat's ``MCPClient`` connects to this over stdio and registers the tools with
the LLM, so the voice agent gains them without knowing anything about Razorpay
or our schema.

Why expose these over MCP rather than calling them in-process: the same tools
become available to any agent that speaks MCP -- a second channel, a support
copilot, a human agent's console -- without reimplementing the money path or
its guarantees. The cost is a transport hop per call, which on a voice loop with
a ~500 ms budget is not free. In-process remains available on the LiveKit path;
this is the shared surface.

Every tool here is deliberately narrow. The agent can send a link and read a
case; it cannot mark anything recovered, change an amount, or reopen a closed
case. Recovery is something only a real Razorpay webhook may assert.
"""

import logging
import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from sqlalchemy import select

from app.db import SessionLocal
from app.metrics import rupees
from app.models import Customer, RecoveryCase
from app.voice.persistence import send_mandate_link_now, send_payment_link_now

logger = logging.getLogger(__name__)

# Loaded here rather than passed through the client config: DATABASE_URL and
# the Razorpay keys must never end up in a committed .mcp.json.
load_dotenv()

mcp = FastMCP("dunning-recovery")


def _bound_case_id() -> int | None:
    """The one case this server process is allowed to act on.

    Injected by the caller that spawns us, never supplied by the model. An
    earlier version took ``recovery_case_id`` as a tool argument and the model
    did exactly what you would expect of a number it was never told: it
    invented one (12345) and the link silently failed to send.

    The hallucination is the mild failure. The dangerous one is a plausible
    id -- there is no reason a model guessing at integers would miss by much,
    and the tool would then have sent a stranger's payment link to whoever is
    on this call. Binding it here means the agent cannot name a case at all.
    """
    raw = os.environ.get("DUNNING_CASE_ID")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("DUNNING_CASE_ID is not an integer: %r", raw)
        return None


@mcp.tool()
async def send_payment_link() -> dict:
    """Send the customer their Razorpay payment link by SMS and email.

    Call this the moment the customer agrees to pay. Takes no arguments -- the
    account is already known from the call. Returns ``sent: true`` with the
    link, or ``sent: false`` with a reason; never claim a link was sent unless
    this returns ``sent: true``.

    Asking twice is safe: the existing link is returned rather than a second one
    being created for the same debt.
    """
    result = await send_payment_link_now(_bound_case_id())
    if result.get("sent"):
        return {
            "sent": True,
            "amount": rupees(result.get("amount_paise")),
            "guidance": (
                "Tell the customer it is on its way by SMS. Do not read the URL "
                "aloud -- it is long and error-prone by voice."
            ),
        }
    return {
        "sent": False,
        "reason": result.get("reason", "unknown"),
        "guidance": (
            "Do NOT tell the customer a link was sent. Apologise briefly and say "
            "a colleague will follow up."
        ),
    }


@mcp.tool()
async def send_mandate_link() -> dict:
    """Send the link that re-activates the customer's auto-pay mandate, by SMS.

    Use this when the mandate itself is the problem -- it was cancelled,
    revoked, or the card behind it is dead. A payment link settles what is owed
    today; only this stops the same failure happening again next cycle.

    Both can be right on one call: send the payment link for the arrears and
    this one for the mandate. Takes no arguments. Never claim it was sent
    unless this returns ``sent: true``.
    """
    result = await send_mandate_link_now(_bound_case_id())
    if result.get("sent"):
        return {
            "sent": True,
            "guidance": (
                "Tell them a link to set up auto-pay again has gone by SMS, and "
                "that it takes a minute. Do not read the URL aloud."
            ),
        }
    return {
        "sent": False,
        "reason": result.get("reason", "unknown"),
        "guidance": (
            "Do NOT tell the customer a link was sent. Say a colleague will "
            "follow up about re-activating auto-pay."
        ),
    }


@mcp.tool()
async def get_case() -> dict:
    """Look up this call's recovery case: the amount, why the payment failed,
    and who it is for.

    Read-only, and scoped to the customer you are speaking to -- it takes no
    arguments, so it can never read someone else's account.
    """
    recovery_case_id = _bound_case_id()
    if recovery_case_id is None:
        return {"found": False}

    async with SessionLocal() as session:
        case = await session.get(RecoveryCase, recovery_case_id)
        if case is None:
            return {"found": False}

        customer = None
        if case.razorpay_customer_id:
            customer = (
                await session.execute(
                    select(Customer).where(
                        Customer.razorpay_customer_id == case.razorpay_customer_id
                    )
                )
            ).scalar_one_or_none()

        return {
            "found": True,
            "amount": rupees(case.original_amount),
            "failure_reason": case.failure_reason,
            "status": case.status,
            "attempts_so_far": case.attempt_count,
            "customer_name": (customer.name if customer else None),
            "link_already_sent": bool(case.payment_link_url),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="stdio")
