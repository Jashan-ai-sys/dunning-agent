"""The dunning conversation, as a graph.

Hinglish by default -- code-switching between Hindi and English is how most
Indian customers actually speak, and forcing either pure language makes the call
sound like a robocall. ``{language_hint}`` lets a customer's stored preference
override the default.

Two rules are baked into the prompts rather than left to the model's judgement,
because both are compliance issues rather than style choices:

* never argue about a disputed charge -- hand it to a human,
* never read out card details or ask for them; recovery happens through a
  Razorpay payment link, never over the phone.
"""

from app.voice.graph import ConversationGraph, Edge, Node, NodeKind
from app.voice.intents import CallIntent

SYSTEM_STYLE = """You are a polite billing assistant for {company_name}, calling
{customer_name} about a subscription payment that failed.

Speak {language_hint}. Keep every turn to one or two short sentences -- this is a
phone call, not an essay. Never ask for card, CVV, OTP or UPI PIN details; the
customer pays through a secure link we send, never over the phone. If the
customer sounds annoyed, stay calm and offer to end the call."""


GREET = Node(
    id="greet",
    kind=NodeKind.START,
    prompt=(
        SYSTEM_STYLE
        + "\n\nOpen the call. Greet them, say you are calling from {company_name}, "
        "and confirm you are speaking to {customer_name}. Do not mention the "
        "failed payment until identity is confirmed -- this is someone's billing "
        "information."
    ),
    extracts=("identity_confirmed",),
    edges=(
        Edge(
            to="explain",
            label="identity_confirmed",
            condition="The person confirms they are the customer, or clearly indicates it is them.",
        ),
        Edge(
            to="wrong_number",
            label="not_the_customer",
            condition=(
                "The person says this is the wrong number, that they are someone else, "
                "or that they do not know the customer."
            ),
            speech="Sorry for the trouble, main number check karwa leta hoon.",
        ),
    ),
)

EXPLAIN = Node(
    id="explain",
    kind=NodeKind.AGENT,
    prompt=(
        SYSTEM_STYLE
        + "\n\nTell them their subscription payment of Rs {amount_rupees} did not go "
        "through. If it helps, mention the reason: {failure_reason}. Be matter of "
        "fact, not accusatory -- most failures are bank-side, not the customer's "
        "fault. Then pause for their reaction."
    ),
    edges=(
        Edge(
            to="ask_intent",
            label="acknowledged",
            condition="The customer acknowledges the failed payment or asks what to do next.",
        ),
        Edge(
            to="dispute",
            label="disputes_charge",
            condition=(
                "The customer says the charge is wrong, that they already paid, that they "
                "cancelled the subscription, or that they never signed up."
            ),
        ),
    ),
)

ASK_INTENT = Node(
    id="ask_intent",
    kind=NodeKind.AGENT,
    prompt=(
        SYSTEM_STYLE
        + "\n\nOffer exactly three options and let them choose: pay now via a link "
        "you send on WhatsApp or SMS, pay later at a time they pick, or stop the "
        "subscription. Do not push. Do not offer a discount -- you have no "
        "authority to change the amount."
    ),
    extracts=("preferred_time",),
    edges=(
        Edge(
            to="pay_now",
            label="pay_now",
            condition="The customer wants to pay now, or agrees to receive a payment link.",
            speech="Bilkul, main abhi aapko link bhej deta hoon.",
        ),
        Edge(
            to="pay_later",
            label="pay_later",
            condition=(
                "The customer wants to pay later, asks us to call back, or names a "
                "specific day or time."
            ),
        ),
        Edge(
            to="declined",
            label="declined",
            condition=(
                "The customer refuses to pay, wants to cancel the subscription, or asks "
                "not to be contacted again."
            ),
        ),
        Edge(
            to="dispute",
            label="disputes_charge",
            condition="The customer disputes the charge or says they already paid.",
        ),
    ),
)

PAY_NOW = Node(
    id="pay_now",
    kind=NodeKind.END,
    intent=CallIntent.RETRY_NOW,
    prompt=(
        "Confirm you are sending a secure payment link right now, tell them it is "
        "for Rs {amount_rupees}, thank them, and end the call."
    ),
)

PAY_LATER = Node(
    id="pay_later",
    kind=NodeKind.END,
    intent=CallIntent.RETRY_LATER,
    prompt=(
        "Acknowledge their preferred time, say you will follow up then, thank them "
        "and end the call. Do not promise an exact hour."
    ),
)

DECLINED = Node(
    id="declined",
    kind=NodeKind.END,
    intent=CallIntent.DECLINED,
    prompt=(
        "Accept their decision without pushing back even once. Confirm they will "
        "not be called again about this payment, thank them and end the call."
    ),
)

WRONG_NUMBER = Node(
    id="wrong_number",
    kind=NodeKind.END,
    intent=CallIntent.WRONG_NUMBER,
    prompt=(
        "Apologise briefly for the wrong number, say the number will be removed, "
        "and end the call. Do not reveal any billing details."
    ),
)

DISPUTE = Node(
    id="dispute",
    kind=NodeKind.END,
    intent=CallIntent.DISPUTE,
    prompt=(
        "Do not argue and do not try to resolve it. Say a human colleague will "
        "review the account and get back to them, thank them and end the call."
    ),
)


DUNNING_FLOW = ConversationGraph(
    nodes=(GREET, EXPLAIN, ASK_INTENT, PAY_NOW, PAY_LATER, DECLINED, WRONG_NUMBER, DISPUTE)
)

LANGUAGE_HINTS = {
    "hinglish": "in natural Hinglish, mixing Hindi and English the way urban Indians do",
    "hi": "in simple Hindi",
    "en": "in clear Indian English",
}


def language_hint(preferred_language: str | None) -> str:
    return LANGUAGE_HINTS.get(preferred_language or "hinglish", LANGUAGE_HINTS["hinglish"])
