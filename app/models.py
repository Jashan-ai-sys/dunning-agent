from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class WebhookEvent(Base, TimestampMixin):
    """Raw, verified webhook envelope.

    Persisted synchronously before any processing so that a crash in the handler
    never loses an event -- ``processed_at is NULL`` is the replay queue.
    """

    __tablename__ = "webhook_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    razorpay_event_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    event: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    signature_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processing_error: Mapped[str | None] = mapped_column(Text)
    #: How many times a handler has been tried on this envelope. Without it
    #: there is no way to tell a blip from an event that will never succeed.
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    #: Dead-lettered: given up on, and skipped by the replay sweep.
    #:
    #: The sweep reads the oldest unprocessed rows first, so an envelope that
    #: fails deterministically sorts to the front of every pass forever. Once
    #: there are more of those than the sweep's limit, no newer event is
    #: replayed again -- and because ``record_event`` dedupes on
    #: ``razorpay_event_id``, Razorpay's own redelivery is a no-op, making the
    #: sweep the only retry path there is. This column is what lets it move on.
    dead_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_webhook_events_unprocessed", "processed_at"),)


class Customer(Base, TimestampMixin):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    razorpay_customer_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(32))
    email: Mapped[str | None] = mapped_column(String(255))
    # 'hi' | 'en' | 'hinglish' -- drives the voice agent prompt. Hindi by
    # default; the agent still mirrors whatever the customer actually speaks.
    preferred_language: Mapped[str] = mapped_column(String(16), nullable=False, default="hi")
    # Set when a call establishes we must never dial this person again: an
    # explicit refusal, a wrong number, a disputed charge.
    #
    # Held on the customer rather than the case because the obligation follows
    # the person, not the debt. A second failed charge opens a second case, and
    # a case-scoped suppression would happily call a wrong number all over
    # again -- which is the compliance and privacy problem, not a smaller
    # version of it.
    #: Nothing clears this today. A dispute later resolved in the merchant's
    #: favour leaves the customer unreachable until somebody clears the flag by
    #: hand, which is the safe direction to fail but is not a workflow yet.
    do_not_contact: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    #: The CallIntent that closed the door, kept for the audit trail.
    do_not_contact_reason: Mapped[str | None] = mapped_column(String(64))
    do_not_contact_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Someone else answered this number. That is a fact about the *number*, not
    # about the customer: suppressing the whole person would also block an
    # email payment link that has nothing to do with the bad phone, and the
    # debt is still owed. Kept as a flag rather than by blanking ``phone``
    # because ``upsert_customer`` refreshes that field from Razorpay.
    phone_is_wrong: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    # When we last contacted this *person*, by any channel, about any debt.
    #
    # The attempt budget and backoff are per case, which silently assumed one
    # case per customer. That does not hold: four cases opened against one
    # subscription in two hours from repeated authorisation attempts, and each
    # carried its own untouched budget. Nothing in the policy said "do not ring
    # the same person four times in a minute", because nothing had to until
    # cases stopped being one-per-debt.
    last_contacted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Subscription(Base, TimestampMixin):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    razorpay_subscription_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    razorpay_customer_id: Mapped[str | None] = mapped_column(String(64), index=True)
    razorpay_plan_id: Mapped[str | None] = mapped_column(String(64))
    plan_amount: Mapped[int | None] = mapped_column(Integer)  # paise
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    status: Mapped[str | None] = mapped_column(String(32))


class Payment(Base, TimestampMixin):
    __tablename__ = "payments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    razorpay_payment_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # The only subscription linkage a payment.* webhook carries; everything else
    # is resolved by fetching this invoice from the API.
    razorpay_invoice_id: Mapped[str | None] = mapped_column(String(64), index=True)
    razorpay_subscription_id: Mapped[str | None] = mapped_column(String(64), index=True)
    razorpay_customer_id: Mapped[str | None] = mapped_column(String(64), index=True)
    #: Present on every payment. The only linkage a one-off checkout failure
    #: has -- it carries no invoice and no subscription.
    razorpay_order_id: Mapped[str | None] = mapped_column(String(64), index=True)
    amount: Mapped[int] = mapped_column(Integer, nullable=False)  # paise
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    method: Mapped[str | None] = mapped_column(String(32))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_description: Mapped[str | None] = mapped_column(Text)
    error_reason: Mapped[str | None] = mapped_column(String(128))
    error_source: Mapped[str | None] = mapped_column(String(64))
    error_step: Mapped[str | None] = mapped_column(String(64))
    notes: Mapped[dict | None] = mapped_column(JSONB)


class RecoveryCase(Base, TimestampMixin):
    """One recoverable failure. The unit the batch metrics are computed over."""

    __tablename__ = "recovery_cases"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    razorpay_payment_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    razorpay_invoice_id: Mapped[str | None] = mapped_column(String(64))
    razorpay_subscription_id: Mapped[str | None] = mapped_column(String(64), index=True)
    razorpay_customer_id: Mapped[str | None] = mapped_column(String(64), index=True)
    #: The order a checkout case belongs to. A subscription charge has one too,
    #: but only checkout recovery attributes payment back through it -- an
    #: abandoned checkout has no invoice and no subscription to reconcile by.
    razorpay_order_id: Mapped[str | None] = mapped_column(String(64), index=True)
    original_amount: Mapped[int] = mapped_column(Integer, nullable=False)  # paise
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    #: Razorpay's coarse class: BAD_REQUEST_ERROR, GATEWAY_ERROR, SERVER_ERROR.
    failure_code: Mapped[str | None] = mapped_column(String(64))
    #: ``error_description`` -- prose, written for a human, and changed without
    #: notice. Displayed and logged, never branched on.
    failure_reason: Mapped[str | None] = mapped_column(Text)
    #: ``error_source`` -- who the failure belongs to: customer, issuer, bank,
    #: gateway, internal, business. The stable half of the diagnosis.
    failure_source: Mapped[str | None] = mapped_column(String(32))
    #: ``error_reason`` -- the machine-readable cause, e.g. insufficient_funds.
    #: Kept apart from ``failure_reason`` precisely because that one is prose.
    failure_reason_code: Mapped[str | None] = mapped_column(String(64))
    #: ``error_step`` -- how far the customer got: payment_initiation,
    #: payment_authentication, payment_authorization. The difference between
    #: "a charge was attempted on their behalf" and "they were sitting there
    #: trying to pay", which is what the calling priority turns on.
    failure_step: Mapped[str | None] = mapped_column(String(64))
    #: Calling priority, 1 is most urgent. Denormalised so the claim query can
    #: order on it: the tier is a pure function of the failure columns above,
    #: which never change after the case is opened, so it cannot drift.
    #: :mod:`app.priority` is the source of truth for how it is derived.
    priority_tier: Mapped[int] = mapped_column(
        Integer, nullable=False, default=4, server_default="4", index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open", index=True)
    # 'razorpay' for cases opened by a real webhook, 'seed' for synthetic demo
    # data. Kept in the schema rather than a naming convention, so simulated
    # recovery can never be reported as real money.
    source: Mapped[str] = mapped_column(
        String(16), nullable=False, default="razorpay", server_default="razorpay", index=True
    )
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    # Consecutive failures on *our* side -- telephony down, Razorpay refusing
    # the link. Counted separately from attempt_count on purpose: an outage
    # must not spend the customer's contact budget. But something has to bound
    # it, or a case whose delivery can never succeed retries every backoff
    # window forever and never reaches max_attempts. Reset by any success.
    delivery_failures: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # When this case is next worth looking at. Written whenever the policy
    # parks a case, and honoured by the orchestrator's claim query so that
    # waiting cases do not occupy the batch.
    #
    # Without it the queue starves: a parked case stays OPEN and is among the
    # oldest rows, so it sorts to the front of every batch and, once enough of
    # them accumulate to fill worker_batch_size, nothing newer is ever claimed.
    next_eligible_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    # Set when Razorpay stops its own retries -- the hard escalation signal.
    halted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # A promise to pay, made on a call. Kept as two timestamps rather than a
    # status so "kept" stays *derived* -- a promise is kept when the money
    # arrives before the deadline, and nothing has to remember to update a flag.
    promised_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    promise_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    recovered_payment_id: Mapped[str | None] = mapped_column(String(64))
    recovered_amount: Mapped[int | None] = mapped_column(Integer)
    recovered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # The recovery payment link, if one has been sent. One active link per case
    # is enough for this workflow; a new attempt reuses or replaces it.
    payment_link_id: Mapped[str | None] = mapped_column(String(64))
    payment_link_url: Mapped[str | None] = mapped_column(String(255))
    payment_link_sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    actions: Mapped[list["RecoveryAction"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )


class RecoveryAction(Base, TimestampMixin):
    """Append-only audit trail. Never updated, never deleted."""

    __tablename__ = "recovery_actions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    recovery_case_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("recovery_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    action_type: Mapped[str] = mapped_column(String(48), nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSONB)

    case: Mapped[RecoveryCase] = relationship(back_populates="actions")


class VoiceCall(Base, TimestampMixin):
    """One outbound call attempt against a recovery case.

    Separate from ``recovery_actions`` on purpose: the audit trail records that
    a call happened, this records what the call *was* -- room, duration,
    transcript, where the conversation graph ended up. Metrics on answer rate
    and intent mix come from here.
    """

    __tablename__ = "voice_calls"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    recovery_case_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("recovery_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="livekit")
    room_name: Mapped[str | None] = mapped_column(String(128), index=True)
    call_id: Mapped[str | None] = mapped_column(String(128))
    # The number actually dialled, so the audit trail stands on its own even if
    # the customer record is later corrected.
    dialled_number: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="initiated")
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    detected_intent: Mapped[str | None] = mapped_column(String(32))
    #: Terminal node the conversation graph reached.
    final_node_id: Mapped[str | None] = mapped_column(String(64))
    #: Labelled turns from the conversation graph: which node, what the
    #: customer said, which edge label the model chose, and whether it was
    #: valid. This is the training set for offline prompt optimisation --
    #: collected from real traffic rather than invented.
    transitions: Mapped[list | None] = mapped_column(JSONB)
    transcript: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
