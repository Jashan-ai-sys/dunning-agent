from enum import StrEnum


class CaseStatus(StrEnum):
    """Lifecycle of a single recovery attempt against one failed charge."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RECOVERED = "recovered"
    DECLINED = "declined"
    STOPPED = "stopped"


class PaymentStatus(StrEnum):
    CAPTURED = "captured"
    FAILED = "failed"
    REFUNDED = "refunded"


class ActionType(StrEnum):
    """Audit-trail vocabulary. Every state change on a case appends one of these."""

    CASE_OPENED = "case_opened"
    SUBSCRIPTION_PENDING = "subscription_pending"
    SUBSCRIPTION_HALTED = "subscription_halted"
    POLICY_DECISION = "policy_decision"
    VOICE_CALL = "voice_call"
    PAYMENT_LINK_CREATED = "payment_link_created"
    PAYMENT_CAPTURED = "payment_captured"
    STOPPED = "stopped"


# Razorpay subscription statuses we care about for recovery.
SUBSCRIPTION_AT_RISK = {"pending", "halted"}
