"""The remedy to suggest, decided by rule before the call starts.

A failed card and an empty account are not the same problem, and "here is a
payment link" is only the right answer to some of them. Sending a link to
someone whose card expired asks them to re-enter a card that will fail again;
telling someone with no balance to pay right now asks for money that is not
there.

The route is chosen here, not by the model. Two reasons:

* it is the same decision every time for the same failure code, which is the
  definition of a rule, and a rule that drifts per call cannot be audited;
* it is the one place where the agent tells a customer what to *do*, and that
  should be a sentence the business agreed to, not one the model composed.

The model still owns delivery -- when to say it, in whose language, and how to
respond if the customer pushes back. It just does not get to choose the remedy.

Written as instructions to the agent rather than as speech: the agent renders
them in the call's language under the rule in ``flow.py``. Contrast
``reasons.py``, which is read aloud close to verbatim and is therefore stored
in Devanagari.
"""

import re

#: Matched against the lowercased failure reason, first hit wins. The keys
#: mirror ``reasons.py`` so a failure is explained and remedied consistently.
_ROUTES: tuple[tuple[str, str], ...] = (
    (
        r"expir",
        "Their card is expired, so retrying the same card will fail again. The link "
        "you send lets them pay with a different card or UPI -- say that, so they do "
        "not simply re-enter the dead card.",
    ),
    (
        r"insufficient|low balance|not enough",
        "There was no balance on the day we tried, which is usually timing rather "
        "than inability. If they cannot pay this moment, ask which day suits them "
        "and offer to try again then -- that is often a better recovery than "
        "pushing for payment now.",
    ),
    (
        r"limit|exceed",
        "They hit a per-transaction or daily limit, not a lack of money. Suggest "
        "paying through the link with UPI or another card, which usually clears a "
        "card limit.",
    ),
    (
        r"mandate|autopay|emandate|subscription.*cancel",
        "Their auto-pay mandate is no longer active, so future payments will not "
        "collect on their own either. The link settles what is owed; mention that "
        "auto-pay will need setting up again so this does not repeat.",
    ),
    (
        r"gateway|timeout|timed out|network|technical|could not process|unable to process",
        "This failed on the bank's side, not theirs. It very often succeeds on a "
        "second attempt, so the link is likely all that is needed -- say so, and do "
        "not imply they did anything wrong.",
    ),
    (
        r"invalid|incorrect|wrong",
        "The card details did not go through. The link lets them pay with a "
        "different method, which is faster than correcting the saved card.",
    ),
)

_DEFAULT_ROUTE = (
    "Offer the payment link as the way to settle this. Do not speculate about why "
    "the payment failed beyond the reason you have already given."
)

#: Appended when Razorpay has stopped retrying. Additive on purpose: the remedy
#: for the failure does not change, only the urgency around it.
_HALTED_SUFFIX = (
    " Because the bank has stopped retrying automatically, this will not resolve on "
    "its own -- paying through the link is now the only way it clears."
)


def suggested_route(failure_reason: str | None, *, halted: bool = False) -> str:
    """The remedy the agent should steer toward for this failure.

    Falls back to the plain link route rather than guessing: an agent that
    invents a diagnosis is worse than one that simply offers to take payment.
    """
    text = (failure_reason or "").strip().lower()

    route = _DEFAULT_ROUTE
    if text:
        for pattern, instruction in _ROUTES:
            if re.search(pattern, text):
                route = instruction
                break

    return route + (_HALTED_SUFFIX if halted else "")


__all__ = ["suggested_route"]
