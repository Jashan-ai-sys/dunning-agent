"""Root cause of a failed charge.

Track 03's first example direction is "payment degradation -> root cause ->
recovery action". This module is the middle term: it reads the failure fields
Razorpay put on the case and names the cause. It does not decide what to do --
that is :mod:`app.policy`, which owns the stopping rules and has to be able to
overrule any recommendation made here.

Pure by design: no database, no clock, no network, so every rule below is a
plain unit test.

Razorpay describes a failure with four fields. Two of them are usable:

``error_source``
    Who the failure belongs to -- ``customer``, ``issuer``, ``bank``,
    ``gateway``, ``internal``, ``business``. A small, stable set, and the
    reason this module classifies by source first.

``error_reason``
    The machine-readable cause, e.g. ``insufficient_funds``. More precise, but
    Razorpay documents its values *per payment method* rather than as one list,
    so no table here can claim to be exhaustive.

``error_description`` is prose written for a human and is not read at all.

The consequence of that asymmetry is the shape of :func:`diagnose`: a specific
reason wins when we recognise it, and anything unrecognised falls through to
the source. A reason we have never seen degrades to a coarser but still correct
answer rather than to ``UNKNOWN``.
"""

from enum import StrEnum

from app.models import RecoveryCase


class RootCause(StrEnum):
    """Why the charge failed, at the granularity an intervention cares about."""

    #: The instrument works; there was no money behind it.
    CUSTOMER_FUNDS = "customer_funds"
    #: The instrument itself is dead -- expired, blocked, mandate revoked. No
    #: amount of retrying the same charge will fix it.
    CUSTOMER_INSTRUMENT = "customer_instrument"
    #: The customer had to do something and did not -- OTP not entered, 3DS
    #: abandoned.
    CUSTOMER_ACTION = "customer_action"
    #: The issuer refused. Often temporary, and not ours to fix.
    BANK_DECLINE = "bank_decline"
    #: Razorpay or the gateway broke. Ours to retry, never the customer's to
    #: hear about.
    TRANSIENT = "transient"
    #: Our own configuration is wrong. A human has to look at it.
    CONFIGURATION = "configuration"
    #: No usable failure fields on the case.
    UNKNOWN = "unknown"


#: ``error_source`` -> cause. The documented, method-independent half.
_BY_SOURCE: dict[str, RootCause] = {
    "customer": RootCause.CUSTOMER_ACTION,
    "issuer": RootCause.BANK_DECLINE,
    "bank": RootCause.BANK_DECLINE,
    "gateway": RootCause.TRANSIENT,
    "internal": RootCause.TRANSIENT,
    "business": RootCause.CONFIGURATION,
}

#: ``error_reason`` -> cause, for the reasons worth separating from their
#: source. Deliberately not exhaustive: Razorpay publishes reasons per payment
#: method, so this table can only ever be the ones we have evidence for.
#: Everything else is handled correctly, if less precisely, by ``_BY_SOURCE``.
_BY_REASON: dict[str, RootCause] = {
    "insufficient_funds": RootCause.CUSTOMER_FUNDS,
    "card_expired": RootCause.CUSTOMER_INSTRUMENT,
    "expired_card": RootCause.CUSTOMER_INSTRUMENT,
    "card_blocked": RootCause.CUSTOMER_INSTRUMENT,
    "invalid_card": RootCause.CUSTOMER_INSTRUMENT,
    "incorrect_card_details": RootCause.CUSTOMER_INSTRUMENT,
    "mandate_revoked": RootCause.CUSTOMER_INSTRUMENT,
    "mandate_cancelled": RootCause.CUSTOMER_INSTRUMENT,
    "invalid_otp": RootCause.CUSTOMER_ACTION,
    "incorrect_otp": RootCause.CUSTOMER_ACTION,
    "payment_timeout": RootCause.TRANSIENT,
    "gateway_timeout": RootCause.TRANSIENT,
    "server_error": RootCause.TRANSIENT,
    "international_transaction_not_allowed": RootCause.BANK_DECLINE,
}

#: Causes where the customer cannot make the payment succeed by trying again.
#: Sending these a payment link and nothing else is how a case quietly rots.
NEEDS_A_CONVERSATION = frozenset({RootCause.CUSTOMER_INSTRUMENT})

#: Causes where Razorpay's own retry sequence deserves the first attempt.
#:
#: ``CONFIGURATION`` is deliberately absent even though it is equally not the
#: customer's fault: a misconfiguration is not something a retry will fix, so
#: the policy stops those cases for a human instead of waiting on a bank.
DEFERS_TO_THE_BANK = frozenset({RootCause.TRANSIENT, RootCause.BANK_DECLINE})


def classify(source: str | None, reason_code: str | None) -> RootCause:
    """Name the root cause behind one ``(error_source, error_reason)`` pair.

    Reason first, source second. Both are normalised because Razorpay is not
    consistent about case across payment methods.

    Split out from :func:`diagnose` so the batch report can classify grouped
    rows without building a ``RecoveryCase`` per group.
    """
    reason = (reason_code or "").strip().lower()
    if reason in _BY_REASON:
        return _BY_REASON[reason]

    normalised_source = (source or "").strip().lower()
    if normalised_source in _BY_SOURCE:
        return _BY_SOURCE[normalised_source]

    return RootCause.UNKNOWN


def diagnose(case: RecoveryCase) -> RootCause:
    """Name the root cause of ``case``'s failed charge."""
    return classify(case.failure_source, case.failure_reason_code)
