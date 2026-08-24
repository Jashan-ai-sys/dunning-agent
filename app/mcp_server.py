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

from mcp.server.fastmcp import FastMCP
from sqlalchemy import select

from app.db import SessionLocal
from app.metrics import rupees
from app.models import Customer, RecoveryCase
from app.voice.persistence import send_payment_link_now

logger = logging.getLogger(__name__)

mcp = FastMCP("dunning-recovery")


@mcp.tool()
async def send_payment_link(recovery_case_id: int) -> dict:
    """Send the customer their Razorpay payment link by SMS and email.

    Call this the moment the customer agrees to pay. Returns ``sent: true`` with
    the link, or ``sent: false`` with a reason -- never claim a link was sent
    unless this returns ``sent: true``.

    Asking twice is safe: the existing link is returned rather than a second one
    being created for the same debt.
    """
    result = await send_payment_link_now(recovery_case_id)
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
async def get_case(recovery_case_id: int) -> dict:
    """Look up a recovery case: amount, why the payment failed, and who it is for.

    Read-only. Use it to answer a customer asking what the charge was for.
    """
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
