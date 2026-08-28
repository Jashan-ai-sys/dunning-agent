"""Which case gets called first.

Ordering the queue by ``created_at`` treats a dead mandate on a Rs 5,000 plan
the same as a bank blip on a Rs 60 one. This is the weighted alternative the
Blostem backend notes argue for (``docs/lessons-from-blostem-backend.md``).

The policy, in the operator's words: a broken mandate first, then a customer
who was actually sitting there trying to pay, then an abandoned checkout,
then everything else.

Pure, like :mod:`app.diagnosis` and :mod:`app.policy`: it reads columns and
returns a number, so every rule is a plain unit test.

Why a stored column rather than a SQL expression: the tier depends on
:func:`app.diagnosis.diagnose`, and duplicating that mapping as a ``CASE``
would give the queue and the report two different opinions about the same
failure. It is computed once, in Python, when the case is opened -- and the
columns it reads are written at that same moment and never touched again, so a
stored tier cannot go stale.
"""

import math
from enum import IntEnum

from sqlalchemy import case as sql_case
from sqlalchemy import func

from app.diagnosis import RootCause, classify
from app.models import RecoveryCase

#: ``error_step`` values that mean the customer was in the payment flow when it
#: failed -- an OTP screen, a bank redirect -- rather than a charge being
#: attempted on their behalf while they were nowhere near it.
#:
#: ``payment_initiation`` is deliberately absent: that is the recurring-charge
#: case, where nobody was watching.
IN_THE_FLOW = frozenset({"payment_authentication", "payment_authorization"})


class Priority(IntEnum):
    """The tier a case falls in. Lower is more urgent.

    The tier is the *taxonomy*; :func:`score` turns it into an ordering that
    the size of the debt can argue with.
    """

    #: The mandate itself is broken: revoked, or the card behind it is dead.
    #: Most urgent because it is the only tier where *every future charge*
    #: fails too, not just this one -- and no payment link fixes that.
    MANDATE_BROKEN = 1
    #: They were trying to pay and it failed under them. Warm, recent intent.
    PAYMENT_ATTEMPTED = 2
    #: Abandoned checkout. Reserved rather than reachable: Razorpay sends no
    #: webhook for an abandoned checkout, so nothing produces this tier yet --
    #: detecting it means sweeping our own unpaid orders, which is a separate
    #: ingestion path. The tier exists so the ordering does not have to change
    #: when that lands.
    CHECKOUT_ABANDONED = 3
    #: Bank-side, transient, or undiagnosed. Worth recovering, not worth
    #: jumping the queue for.
    BACKGROUND = 4


def tier_from(
    source: str | None, reason_code: str | None, step: str | None
) -> Priority:
    """The calling priority behind one set of Razorpay failure fields.

    Split out from :func:`tier_for` so a case can be scored as it is inserted,
    before there is a row to read it back from.
    """
    cause = classify(source, reason_code)

    if cause is RootCause.CUSTOMER_INSTRUMENT:
        return Priority.MANDATE_BROKEN

    # The step is the stronger signal and is checked first: a customer who hit
    # an authentication failure was demonstrably present, whatever the reason
    # code says about it.
    if (step or "").strip().lower() in IN_THE_FLOW:
        return Priority.PAYMENT_ATTEMPTED

    if cause in (RootCause.CUSTOMER_FUNDS, RootCause.CUSTOMER_ACTION):
        return Priority.PAYMENT_ATTEMPTED

    return Priority.BACKGROUND


def tier_for(case: RecoveryCase) -> Priority:
    """The calling priority of one case."""
    return tier_from(case.failure_source, case.failure_reason_code, case.failure_step)


# --- Scoring --------------------------------------------------------------
#
# Strict tier ordering has one bad property: it lets a trivial case in a high
# tier outrank a large one in a lower tier forever. A Rs 200 broken mandate is
# genuinely more urgent than a Rs 200 bank decline -- but it is not more urgent
# than a Rs 5,000 payment the customer was in the middle of making.
#
# So the tier sets a weight and the debt argues with it. The log is what keeps
# that argument civil: it takes a 25x difference in money to overcome one tier
# step, rather than a 2x one, so the tiers still decide most of the ordering
# and only a genuinely large debt jumps.

#: Multiplier per tier. The gaps are what decide how much money it takes to
#: climb a tier -- see ``test_the_weights_mean_what_the_comment_says``.
TIER_WEIGHTS: dict[int, float] = {
    Priority.MANDATE_BROKEN: 1.0,
    Priority.PAYMENT_ATTEMPTED: 0.7,
    Priority.CHECKOUT_ABANDONED: 0.5,
    Priority.BACKGROUND: 0.35,
}

#: Razorpay has stopped retrying, so we are the only route left to this money.
HALTED_BOOST = 1.2


def score(tier: int, amount_paise: int, *, halted: bool = False) -> float:
    """How badly this case wants to be called. Higher is more urgent.

    Note what is deliberately *not* in here: recency. Favouring fresh failures
    would read well -- a customer whose card failed an hour ago may still be at
    their desk -- but it starves old cases, and a case that never reaches the
    top of the queue never spends an attempt, so it never reaches the attempt
    cap that is supposed to end it. Age stays a tiebreak, oldest first.
    """
    weight = TIER_WEIGHTS.get(int(tier), TIER_WEIGHTS[Priority.BACKGROUND])
    boost = HALTED_BOOST if halted else 1.0
    return weight * math.log1p(max(amount_paise, 0) / 100) * boost


def score_expression():
    """:func:`score`, as SQL, so the claim query can order by it.

    The same constants render both, and a parity test asserts the two agree on
    real rows -- the queue and any explanation of the queue must not be able to
    disagree about which case comes first.
    """
    weight = sql_case(
        *[(RecoveryCase.priority_tier == tier, value) for tier, value in TIER_WEIGHTS.items()],
        else_=TIER_WEIGHTS[Priority.BACKGROUND],
    )
    boost = sql_case((RecoveryCase.halted_at.isnot(None), HALTED_BOOST), else_=1.0)
    return weight * func.ln(1 + func.greatest(RecoveryCase.original_amount, 0) / 100.0) * boost


def score_for(case: RecoveryCase) -> float:
    """:func:`score` for one case."""
    return score(case.priority_tier, case.original_amount, halted=case.halted_at is not None)
