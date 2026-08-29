"""The dunning conversation, as a graph.

Hindi by default. ``{language_hint}`` sets the opening language from the
customer's stored preference; from the second turn on the model mirrors whatever
the customer actually speaks, so an English or Hinglish speaker is never trapped
in a language they did not choose.

Three rules are baked into the prompts rather than left to the model's
judgement, because all three are compliance issues rather than style choices:

* never reveal the amount before identity is confirmed,
* never argue about a disputed charge -- hand it to a human,
* never read out card details or ask for them; recovery happens through a
  Razorpay payment link, never over the phone.
"""

from app.voice.graph import ConversationGraph, Edge, Node, NodeKind
from app.voice.intents import CallIntent

#: Carried by every node, terminals included. Sarvam runs in `transcribe` mode,
#: so the model genuinely sees the customer's own language rather than an English
#: translation -- which is what makes mirroring possible at all.
LANGUAGE_RULE = """LANGUAGE -- this rule overrides every other instruction.

SCRIPT: Write every Hindi word in Devanagari. Never romanise Hindi. Do not write
"haan ji", "aapka payment", "link bhej deta hoon" -- write "हाँ जी", "आपका
पेमेंट", "लिंक भेज देता हूँ". Everyday English loanwords that Hindi speakers use
are written in Devanagari too: लिंक, पेमेंट, कार्ड, बैंक, सब्सक्रिप्शन. This
holds even when the customer types romanised Hindi at you -- read their meaning,
answer in Devanagari.

LANGUAGE: If the customer speaks Hindi, or Hindi mixed with English, answer in
Hindi in Devanagari. Only if they speak plain English throughout should you
answer in English. Judge this from what they actually just said, not from their
name or where they are calling from."""

#: Everything that is true for the whole call, and changes at no point during
#: it. Kept in one piece deliberately: it is sent to Gemini as
#: ``system_instruction`` and forms the cacheable prefix of every request, so
#: anything that varies per stage must stay out of it or the prefix changes and
#: the cache misses on every single turn.
SYSTEM_STYLE = (
    """You are a polite billing assistant for {company_name}, calling
{customer_name} about a subscription payment that failed.

"""
    + LANGUAGE_RULE
    + """

Speak {language_hint}, and follow the language rule above on every turn.

Keep every turn to one or two short sentences -- this is a phone call, not an
essay. Never ask for card, CVV, OTP or UPI PIN details; the customer pays through
a secure link we send, never over the phone. If the customer sounds annoyed, stay
calm and offer to end the call.

The call moves through stages, and you are told what to do at each one. Act on
the MOST RECENT stage instruction you have been given -- including one that
arrives as the result of the `transition` tool. A stage you have already left is
finished: never repeat it, and never greet or introduce yourself twice.

Two tools are available at EVERY stage, not only where a stage mentions them,
because a customer asks a question when they think of it and not when the
script expects it:

* `send_payment_link` -- when they want to settle what is owed.
* `send_mandate_link` -- when they ask how to restore auto-pay, re-authorise a
  mandate, or set up UPI autopay again. Answer that question the moment it is
  asked, whatever stage you are in, then carry on where you were.

Never say a link has been sent unless the tool has told you it was."""
)


#: The opening line, rendered rather than generated.
#:
#: The greeting is the one turn with no input to reason about -- there is
#: nothing the customer has said yet. Asking the LLM for it costs a full round
#: trip (~0.5s measured) at the moment the customer has just said "hello" and
#: is listening hardest. Rendering it locally means the first word leaves as
#: soon as TTS can synthesise it (~80ms).
#:
#: It is appended to the context as the assistant's own first turn, so the
#: model knows it has already greeted. Without that it opens the next turn by
#: greeting again, which is a bug this repo has fixed once already.
GREETING = {
    "hi": "नमस्ते, मैं {company_name} से बात कर रहा हूँ। क्या मेरी बात {customer_name} से हो रही है?",
    "hinglish": "नमस्ते, मैं {company_name} से बात कर रहा हूँ। क्या मेरी बात {customer_name} से हो रही है?",
    "en": "Hello, I am calling from {company_name}. Am I speaking with {customer_name}?",
}


def greeting_for(language: str, context: dict) -> str:
    """The opening line for a language, with the call's names filled in."""
    template = GREETING.get(language, GREETING["hi"])
    return template.format(
        company_name=context.get("company_name", "we"),
        customer_name=context.get("customer_name", "you"),
    )


GREET = Node(
    id="greet",
    kind=NodeKind.START,
    prompt=(
        "Open the call. Greet them, say you are calling from {company_name}, "
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
            speech="माफ़ कीजिए, मैं नंबर की जाँच करवा लेता हूँ।",
        ),
    ),
)

EXPLAIN = Node(
    id="explain",
    kind=NodeKind.AGENT,
    prompt=(
        "Tell them their subscription payment of {amount_spoken} did not go "
        "through. If it helps, mention the reason: {failure_reason}. Be matter of "
        "fact, not accusatory -- most failures are bank-side, not the customer's "
        "fault.\n\n{halt_note}\n\nThen pause for their reaction."
    ),
    edges=(
        Edge(
            to="reason_inquiry",
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

REASON_INQUIRY = Node(
    id="reason_inquiry",
    kind=NodeKind.AGENT,
    prompt=(
        "Ask one short question: do they know why it failed? An expired "
        "card, a daily limit, or simply not enough balance that day. Do not "
        "lecture, do not offer options yet, and do not ask twice. This is one "
        "question, then you listen."
    ),
    extracts=("failure_cause",),
    edges=(
        Edge(
            to="ask_intent",
            label="reason_given",
            condition=(
                "The customer gives any reason, says they do not know, asks to get "
                "on with it, or asks what they should do about it."
            ),
        ),
        Edge(
            to="pay_later",
            label="financial_difficulty",
            condition=(
                "The customer says they cannot afford it right now, are short of money, "
                "have lost work, or are in financial trouble."
            ),
        ),
        Edge(
            to="dispute",
            label="disputes_charge",
            condition=(
                "The customer says the charge is wrong, that they already paid, or that "
                "they cancelled the subscription."
            ),
        ),
    ),
)

ASK_INTENT = Node(
    id="ask_intent",
    kind=NodeKind.AGENT,
    prompt=(
        "Offer exactly three options and let them choose: pay now via a link "
        "you send on SMS, pay later at a time they pick, or stop the "
        "subscription. Do not push. Do not offer a discount -- you have no "
        "authority to change the amount.\n\n"
        "The right remedy for this particular failure has already been decided; "
        "work it into what you offer rather than inventing your own:\n"
        "{suggested_route}\n\n"
        "If the mandate itself is dead -- cancelled, revoked, or the card behind "
        "it gone -- also call `send_mandate_link`. The payment link settles what "
        "is owed; only that one stops the same failure next cycle. Both can be "
        "right on one call.\n\n"
        "If they agree to pay now, call the `send_payment_link` tool FIRST and "
        "wait for its result. Only say the link has been sent once that tool "
        "confirms it. Never promise a link you have not actually sent. Then "
        "call `transition` with `pay_now`: sending a link is not the same as "
        "recording that they agreed to pay, and a call that ends here counts "
        "as one where nobody found out what the customer wanted."
    ),
    extracts=("preferred_time",),
    edges=(
        Edge(
            to="pay_now",
            label="pay_now",
            condition="The customer wants to pay now, or agrees to receive a payment link.",
            speech="बिल्कुल, मैं अभी आपको लिंक भेज देता हूँ।",
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
            to="pay_later",
            label="financial_difficulty",
            condition=(
                "The customer says they cannot afford it right now or are short of "
                "money. This is not a refusal -- do not treat it as one."
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
        "Confirm you are sending a secure payment link right now, tell them it "
        "is for {amount_spoken}, thank them, and end the call. Say the amount "
        "exactly as written -- it is already in spoken form."
    ),
)

PAY_LATER = Node(
    id="pay_later",
    kind=NodeKind.END,
    intent=CallIntent.RETRY_LATER,
    prompt=(
        "Acknowledge their preferred time, say you will follow up then, thank "
        "them and end the call. Do not promise an exact hour."
    ),
)

DECLINED = Node(
    id="declined",
    kind=NodeKind.END,
    intent=CallIntent.DECLINED,
    prompt=(
        "Accept their decision without pushing back even once. Confirm they "
        "will not be called again about this payment, thank them and end the call."
    ),
)

WRONG_NUMBER = Node(
    id="wrong_number",
    kind=NodeKind.END,
    intent=CallIntent.WRONG_NUMBER,
    prompt=(
        "Apologise briefly for the wrong number, say the number will be "
        "removed, and end the call. Do not reveal any billing details."
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
    preamble=SYSTEM_STYLE,
    nodes=(
        GREET,
        EXPLAIN,
        REASON_INQUIRY,
        ASK_INTENT,
        PAY_NOW,
        PAY_LATER,
        DECLINED,
        WRONG_NUMBER,
        DISPUTE,
    )
)

LANGUAGE_HINTS = {
    "hi": "in simple, everyday Hindi, written in Devanagari",
    # Hinglish is a speech register, not a script. The English loanwords stay,
    # but they are written in Devanagari like any Hindi speaker would -- never
    # romanised, because a Hindi voice stumbles at script boundaries.
    "hinglish": (
        "in everyday spoken Hindi that naturally uses common English words "
        "(लिंक, पेमेंट, कार्ड), all written in Devanagari"
    ),
    "en": "in clear Indian English",
}

#: Hindi unless a customer record says otherwise. The mirroring rule still
#: applies from the second turn on, so an English-speaking customer is not
#: trapped in a language they did not choose.
DEFAULT_LANGUAGE = "hi"


def language_hint(preferred_language: str | None) -> str:
    key = preferred_language or DEFAULT_LANGUAGE
    return LANGUAGE_HINTS.get(key, LANGUAGE_HINTS[DEFAULT_LANGUAGE])


#: What the agent adds once Razorpay has stopped retrying on its own. Kept as
#: two fixed strings rather than left to the model: whether someone's service is
#: about to stop is a fact about their account, and it must not be softened,
#: dramatised, or invented on a call where it is not true.
HALT_NOTES = {
    True: (
        "Their bank has stopped retrying this payment automatically, so the "
        "subscription will not restart on its own. Say this plainly and only "
        "once -- it is a fact, not a threat, and it is the reason paying today "
        "actually matters. Do not name a cut-off date; you do not have one."
    ),
    False: (
        "The subscription is still active and the bank may retry on its own. "
        "Do not suggest their service is at risk -- it is not."
    ),
}


def halt_note(subscription_halted: bool | None) -> str:
    """The halt line, defaulting to the safe half.

    Unknown is treated as not halted on purpose: wrongly telling someone their
    subscription has stopped is the more damaging error of the two.
    """
    return HALT_NOTES[bool(subscription_halted)]
