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
    original_amount: Mapped[int] = mapped_column(Integer, nullable=False)  # paise
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_reason: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open", index=True)
    # 'razorpay' for cases opened by a real webhook, 'seed' for synthetic demo
    # data. Kept in the schema rather than a naming convention, so simulated
    # recovery can never be reported as real money.
    source: Mapped[str] = mapped_column(
        String(16), nullable=False, default="razorpay", server_default="razorpay", index=True
    )
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Set when Razorpay stops its own retries -- the hard escalation signal.
    halted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
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
