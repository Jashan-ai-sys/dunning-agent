from enum import StrEnum


class CaseStatus(StrEnum):
    """Lifecycle of a single recovery attempt against one failed charge."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RECOVERED = "recovered"
    DECLINED = "declined"
    STOPPED = "stopped"


class CaseSource(StrEnum):
    """Where a case came from. In the schema rather than a naming convention so
    a report can never blend simulated recovery with real money, or a one-off
    checkout with a subscriber whose mandate broke."""

    #: A failed recurring charge on a live subscription.
    RAZORPAY = "razorpay"
    #: A one-off checkout the customer attempted and walked away from.
    CHECKOUT = "checkout"
    #: Synthetic demo data.
    SEED = "seed"


class PaymentStatus(StrEnum):
    CAPTURED = "captured"
    FAILED = "failed"
    REFUNDED = "refunded"


class ActionType(StrEnum):
    """Audit-trail vocabulary. Every state change on a case appends one of these."""

    CASE_OPENED = "case_opened"
    CHECKOUT_ABANDONED = "checkout_abandoned"
    REPEAT_FAILURE = "repeat_failure"
    SUBSCRIPTION_PENDING = "subscription_pending"
    SUBSCRIPTION_HALTED = "subscription_halted"
    POLICY_DECISION = "policy_decision"
    VOICE_CALL = "voice_call"
    PROMISE_MADE = "promise_made"
    PAYMENT_LINK_CREATED = "payment_link_created"
    MANDATE_RETRIED = "mandate_retried"
    MANDATE_LINK_SENT = "mandate_link_sent"
    PAYMENT_CAPTURED = "payment_captured"
    STOPPED = "stopped"


# Razorpay subscription statuses we care about for recovery.
SUBSCRIPTION_AT_RISK = {"pending", "halted"}


class CallStatus(StrEnum):
    """Lifecycle of one outbound call attempt."""

    INITIATED = "initiated"
    ANSWERED = "answered"
    COMPLETED = "completed"
    NO_ANSWER = "no_answer"
    FAILED = "failed"
