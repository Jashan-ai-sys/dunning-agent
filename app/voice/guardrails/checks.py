"""What the agent is not allowed to say, checked without asking a model.

The conversation graph already refuses an illegal *move*. Nothing until now
refused an illegal *sentence*: the graph validates the label the model picks,
never the words it speaks on the way there. These are that second check.

Every rule here is deterministic. That is not a shortcut -- it is the only way
the rail can run on a live call. `self_check_output` asks an LLM whether the
output is acceptable, which is a second round trip on a turn that currently
costs 0.512s end to end. A regex costs microseconds, and unlike a model it
returns the same verdict twice on the same sentence, which is the property you
want in the thing that decides whether a customer hears something.

Both scripts are matched. The agent speaks Devanagari by rule (`LANGUAGE_RULE`
in `flow.py`), but a model that has slipped its script is exactly the model most
likely to have slipped its instructions, so the romanised forms are checked too.
"""

import re
from dataclasses import dataclass

# `\b` cannot be used on the Devanagari terms. Python's `\w` follows
# `str.isalnum()`, and a combining vowel sign such as "ी" (U+0940) is not
# alphanumeric -- so `\bओटीपी\b` never matches, because the word ends on a
# combining mark and there is no word boundary after it. The Latin terms keep
# their boundaries; the Devanagari ones are matched bare, which is safe here
# because none of them is a substring of an innocent word.

#: Never legitimate on this call, in any phrasing. A dunning agent has no
#: reason to know a card number, and Razorpay's own checkout is the only thing
#: that should ever ask. An agent that asks is indistinguishable, to the person
#: on the other end, from the fraud call they have been warned about -- so this
#: is a fraud-surface rule before it is a compliance one.
_CREDENTIAL_LATIN = re.compile(
    r"\b(cvv|cvc|card\s*number|otp|pin\s*number|upi\s*pin|expiry\s*date)\b",
    re.IGNORECASE,
)
_CREDENTIAL_DEVANAGARI = re.compile(
    r"(कार्ड\s*नंबर|ओटीपी|यूपीआई\s*पिन|पिन\s*नंबर|एक्सपायरी|सीवीवी)",
    re.UNICODE,
)

#: A bare mention is not a request. "आपका कार्ड एक्सपायर हो गया है" is the
#: correct thing to say to someone whose card expired; asking them to read the
#: number out is not. The solicitation verb is what separates them.
_SOLICITS = re.compile(
    r"(बताइए|बताएं|बताइये|बता\s*दीजिए|शेयर\s*कर|भेजिए|भेज\s*दीजिए|"
    r"डालिए|दर्ज\s*कर|लिख\s*कर|"
    r"\btell me\b|\bshare\b|\bsend me\b|\benter\b|\bread out\b|\bgive me\b|"
    r"\bwhat is your\b|\bconfirm your\b)",
    re.IGNORECASE | re.UNICODE,
)

#: Only a Razorpay webhook may assert that money moved, and only the MCP tool
#: may assert that a link was sent. The model saying either is the failure the
#: whole audit trail exists to prevent -- a customer told their debt is settled
#: stops worrying about a debt that is still open.
#: The Devanagari half allows up to a few characters between the noun and the
#: verb, because Hindi puts the object and politeness markers in between:
#: "रिफंड कर देंगे", "रिफंड आपको मिल जाएगा". Anchoring them adjacent missed the
#: most natural phrasings, which is the way a rule like this fails in practice.
_GIVING = (
    r"(कर\s*दिया|कर\s*दी|कर\s*दूंगा|कर\s*दूँगा|कर\s*देंगे|कर\s*देता"
    r"|हो\s*गया|हो\s*गयी|मिल\s*जाएगा|सफल)"
)
_SETTLEMENT_CLAIMS = re.compile(
    r"((?:पेमेंट|भुगतान|रिफंड|माफ़?|छूट)[^।.!?]{0,15}" + _GIVING + r"|"
    r"\brefunded\b|\brefund (has been|will be) (processed|issued)\b|"
    r"\bwaive[dr]?\b|\bdiscount\b|\bcancelled the charge\b|"
    r"\bpayment (is |has been )?(successful|complete|received|done)\b)",
    re.IGNORECASE | re.UNICODE,
)

#: Rupee figures the model produced. Matches "₹499", "499 रुपये", "Rs. 499",
#: "rupees 499" -- decimals and thousands separators included, because a model
#: inventing an amount is as likely to invent 4,990.00 as 499.
_AMOUNT = re.compile(
    r"(?:₹|\brs\.?\s*|\brupees?\s*)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)"
    r"|([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*(?:रुपये|रुपए|रूपये)",
    re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True)
class Violation:
    """Why a sentence was withheld. Carried into the audit trail, not just logs.

    ``rule`` is stable and machine-readable so a run of calls can be counted by
    failure kind; ``detail`` is what actually matched, for a human reading one.
    """

    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.detail}"


def _rupees_spoken(text: str) -> list[str]:
    found = []
    for whole, trailing in _AMOUNT.findall(text):
        raw = whole or trailing
        if raw:
            found.append(raw.replace(",", ""))
    return found


def check_output(text: str, *, expected_amount_rupees: str | None = None) -> Violation | None:
    """The one entry point. Returns the first violation, or ``None`` to allow.

    First match wins rather than collecting every violation: the sentence is
    withheld either way, and the first reason is the one worth logging.

    ``expected_amount_rupees`` is the case's own amount as a plain string of
    rupees ("499", "1499.50"). When it is ``None`` the amount rule is skipped
    entirely -- a guard that does not know the right answer must not guess at a
    wrong one, and silently failing open is better than muting a correct
    sentence about someone's money.
    """
    if not text or not text.strip():
        return None

    credential = _CREDENTIAL_LATIN.search(text) or _CREDENTIAL_DEVANAGARI.search(text)
    if credential:
        term = credential.group(0)
        # CVV has no innocent reading. The others do, so they need the verb.
        if re.fullmatch(r"cvv|cvc|सीवीवी", term, re.IGNORECASE) or _SOLICITS.search(text):
            return Violation("credential_solicitation", term.strip())

    settlement = _SETTLEMENT_CLAIMS.search(text)
    if settlement:
        return Violation("unauthorised_settlement_claim", settlement.group(0).strip())

    if expected_amount_rupees is not None:
        expected = expected_amount_rupees.replace(",", "")
        for spoken in _rupees_spoken(text):
            if not _same_amount(spoken, expected):
                return Violation("amount_mismatch", f"said {spoken}, case is {expected}")

    return None


def _same_amount(spoken: str, expected: str) -> bool:
    """Compare as numbers, so "499" and "499.00" are the same amount.

    A model that says "four hundred and ninety nine rupees" is not caught here
    and is not meant to be -- `spoken.py` renders the amount for speech, and
    words are its job. This rule exists for digits, which is the form a
    hallucinated figure actually takes.
    """
    try:
        return abs(float(spoken) - float(expected)) < 0.005
    except ValueError:
        return True
