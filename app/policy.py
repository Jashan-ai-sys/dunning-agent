"""The recovery policy: given a case, decide what to do about it.

Two questions, in order: *should* we act on this case at all, and if so *which*
intervention does its root cause argue for. The first is the bounded-workflow
and compliance half; the second is the "payment degradation -> root cause ->
recovery action" half, and it reads :mod:`app.diagnosis`.

The cheap intervention comes first. A payment link costs a fraction of a call
and interrupts nobody, so a case earns a call by having ignored a link, or by
having a root cause a link cannot actually fix.

Deliberately pure -- no database, no clock, no network. ``now`` and ``settings``
are passed in, so every rule below is exercised by a plain unit test. This is
also where the "bounded workflow" and "compliant escalation" requirements live:
a case can only ever leave here as LINK, CALL, WAIT or STOP, and STOP is
permanent.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

from app.config import Settings
from app.constants import CaseStatus
from app.diagnosis import DEFERS_TO_THE_BANK, NEEDS_A_CONVERSATION, RootCause, diagnose
from app.models import Customer, RecoveryCase


class Action(StrEnum):
    #: Charge the mandate the customer already authorised. Cheapest of all --
    #: they do nothing -- but only honest where the instrument is fine and the
    #: money simply was not there.
    RETRY_MANDATE = "retry_mandate"
    #: Send the payment link and nothing else. Cheap, silent, and enough for
    #: any cause the customer can clear by paying.
    LINK = "link"
    #: Spend a call. Reserved for cases a link has already failed to move, or
    #: whose root cause a link cannot fix on its own.
    CALL = "call"
    WAIT = "wait"
    STOP = "stop"


class StopReason(StrEnum):
    ALREADY_CLOSED = "already_closed"
    DO_NOT_CONTACT = "do_not_contact"
    MAX_ATTEMPTS_REACHED = "max_attempts_reached"
    BELOW_MIN_AMOUNT = "below_min_amount"
    NO_CONTACT_DETAILS = "no_contact_details"
    #: Our own misconfiguration. Chasing the customer for it would be wrong.
    NEEDS_HUMAN = "needs_human"
    #: Every attempt to reach them failed on our side, repeatedly.
    UNDELIVERABLE = "undeliverable"


class WaitReason(StrEnum):
    WITHIN_BACKOFF = "within_backoff"
    OUTSIDE_CONTACT_WINDOW = "outside_contact_window"
    #: Razorpay has not finished its own retry sequence, and the failure was
    #: not the customer's doing. Contacting them now would be premature.
    AWAITING_BANK_RETRY = "awaiting_bank_retry"
    #: We have already spoken to this person recently, about this debt or a
    #: different one.
    CUSTOMER_RECENTLY_CONTACTED = "customer_recently_contacted"


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    #: Why the charge failed. Carried on every decision, including the ones
    #: taken before the cause was consulted, so the audit trail always answers
    #: "what was wrong with this payment" alongside "what did we do about it".
    root_cause: RootCause = RootCause.UNKNOWN
    #: For a WAIT, the earliest time this case is worth re-reading. The
    #: orchestrator stores it so a parked case stops being claimed every tick.
    retry_after: datetime | None = None

    def as_metadata(self) -> dict[str, str]:
        """Shape written to the audit trail."""
        return {
            "action": str(self.action),
            "reason": self.reason,
            "root_cause": str(self.root_cause),
        }


CLOSED_STATUSES = {CaseStatus.RECOVERED, CaseStatus.DECLINED, CaseStatus.STOPPED}


def decide(
    case: RecoveryCase,
    customer: Customer | None,
    *,
    now: datetime,
    settings: Settings,
) -> Decision:
    """Decide the next step for one case.

    Rule order matters, in three bands:

    1. **Stop conditions**, checked before everything else, so a case that has
       exhausted its attempts is closed out rather than parked forever waiting
       for the contact window to open.
    2. **Wait conditions** -- backoff, contact window, and Razorpay's own retry
       sequence.
    3. **Which intervention**, decided last, from the root cause and how many
       attempts this case has already had.

    Two of the rules are customer-scoped rather than case-scoped, and both
    exist because "one case per customer" turned out to be false. Someone who
    told us we have the wrong number must not be dialled again because a
    *different* charge of theirs failed tomorrow; and someone we spoke to an
    hour ago must not be rung again today just because a second debt of theirs
    also came due.
    """
    cause = diagnose(case)

    if case.status in CLOSED_STATUSES:
        return Decision(Action.STOP, StopReason.ALREADY_CLOSED, cause)

    # Checked before the attempt budget so the trail records *why* we stopped.
    # All STOPs end the case, but "do_not_contact" and "max_attempts_reached"
    # are very different things to have to explain afterwards.
    if customer is not None and customer.do_not_contact:
        return Decision(Action.STOP, StopReason.DO_NOT_CONTACT, cause)

    if case.attempt_count >= case.max_attempts:
        return Decision(Action.STOP, StopReason.MAX_ATTEMPTS_REACHED, cause)

    # `or 0`: the column default lands at INSERT, so an unflushed case has None.
    if (case.delivery_failures or 0) >= settings.max_delivery_failures:
        # Our side keeps failing. Not burning attempt_count is right -- an
        # outage must not spend the customer's budget -- but something has to
        # end it, or the case retries every backoff window forever and never
        # reaches the attempt cap at all.
        return Decision(Action.STOP, StopReason.UNDELIVERABLE, cause)

    if case.original_amount < settings.min_recoverable_amount_paise:
        return Decision(Action.STOP, StopReason.BELOW_MIN_AMOUNT, cause)

    if cause is RootCause.CONFIGURATION:
        # Our own setup is broken. The customer did nothing wrong and cannot
        # fix it, so there is nobody to contact -- only somebody to tell.
        return Decision(Action.STOP, StopReason.NEEDS_HUMAN, cause)

    if not _reachable_at_all(customer):
        return Decision(Action.STOP, StopReason.NO_CONTACT_DETAILS, cause)

    if _deferring_to_the_bank(case, cause, now=now, settings=settings):
        return Decision(
            Action.WAIT,
            WaitReason.AWAITING_BANK_RETRY,
            cause,
            retry_after=_grace_expires_at(case, now=now, settings=settings),
        )

    if _within_backoff(case, now=now, settings=settings):
        return Decision(
            Action.WAIT,
            WaitReason.WITHIN_BACKOFF,
            cause,
            retry_after=case.last_attempt_at + timedelta(hours=settings.retry_backoff_hours),
        )

    # Checked after the case's own backoff and before the contact window, so a
    # person is never rung twice in a day because they happen to owe us twice.
    # One subscription produced four cases in two hours here; without this each
    # one would have called with a full, untouched budget of its own.
    if customer.last_contacted_at is not None:
        quiet_until = customer.last_contacted_at + timedelta(
            hours=settings.customer_contact_cooldown_hours
        )
        if now < quiet_until:
            return Decision(
                Action.WAIT, WaitReason.CUSTOMER_RECENTLY_CONTACTED, cause,
                retry_after=quiet_until,
            )

    if not _within_contact_window(now, settings=settings):
        return Decision(Action.WAIT, WaitReason.OUTSIDE_CONTACT_WINDOW, cause)

    return Decision(_intervention_for(case, customer, cause, settings), "eligible", cause)


def _intervention_for(
    case: RecoveryCase, customer: Customer, cause: RootCause, settings: Settings
) -> Action:
    """Pick the cheapest intervention that can actually work.

    A payment link settles any cause the customer can clear by paying, costs
    almost nothing and interrupts nobody, so it is the default opener. Two
    things earn a call instead:

    * a root cause a link cannot fix. An expired card or a revoked mandate
      means this charge and *every future one* fails; the link would recover
      today's money and leave the subscription to break again next cycle, so
      somebody has to talk to them about the instrument.
    * a link that has already been sent and ignored -- which is what a second
      attempt on a case means.

    Both of those want a phone. A customer we can only email keeps getting the
    link; the attempt cap still bounds how many times.

    Below the link sits one cheaper rung: re-charging a mandate that is still
    valid. It only applies where the diagnosis says the money was missing
    rather than the instrument, and never alongside a link -- two live ways to
    settle one debt is how somebody gets charged twice.
    """
    if (
        settings.mandate_retry_enabled
        and cause is RootCause.CUSTOMER_FUNDS
        and case.attempt_count == 0
    ):
        return Action.RETRY_MANDATE

    if not _can_call(customer):
        return Action.LINK
    if cause in NEEDS_A_CONVERSATION:
        return Action.CALL
    return Action.LINK if case.attempt_count == 0 else Action.CALL


def _grace_expires_at(
    case: RecoveryCase, *, now: datetime, settings: Settings
) -> datetime:
    started = case.created_at or now
    return started + timedelta(hours=settings.bank_retry_grace_hours)


def _can_call(customer: Customer | None) -> bool:
    """A number we have not been told is somebody else's."""
    return bool(customer and customer.phone and not customer.phone_is_wrong)


def _reachable_at_all(customer: Customer | None) -> bool:
    """A payment link goes by SMS *or* email, so a phone is not the only way in.

    Before the link intervention existed this rule was "has a phone", which was
    right when the only thing we could do was call. Keeping it that way now
    would abandon money we can still collect from a customer whose email we
    have.
    """
    return bool(customer and (_can_call(customer) or customer.email))


def _deferring_to_the_bank(
    case: RecoveryCase, cause: RootCause, *, now: datetime, settings: Settings
) -> bool:
    """True while Razorpay's own retry sequence still deserves the first go.

    Bounded by the clock, not by ``halted_at`` alone. ``subscription.halted``
    is the signal that Razorpay has given up, but it is a webhook: a
    subscription cancelled rather than halted, an event not subscribed in the
    dashboard, or a delivery lost past the replay window all mean it never
    arrives. Waiting on it unconditionally makes a second permanent terminal
    state that nothing can leave -- and because these cases stay OPEN and are
    the oldest rows in the queue, they crowd every later case out of the batch.
    """
    if cause not in DEFERS_TO_THE_BANK or case.halted_at is not None:
        return False
    # A case built in memory has no created_at until it is flushed; treat it as
    # brand new rather than crashing on the subtraction.
    age = now - case.created_at if case.created_at is not None else timedelta(0)
    return age < timedelta(hours=settings.bank_retry_grace_hours)


def _within_backoff(case: RecoveryCase, *, now: datetime, settings: Settings) -> bool:
    if case.last_attempt_at is None:
        return False
    return now - case.last_attempt_at < timedelta(hours=settings.retry_backoff_hours)


def _within_contact_window(now: datetime, *, settings: Settings) -> bool:
    """True if the customer's local time is inside the permitted calling hours."""
    local_hour = now.astimezone(ZoneInfo(settings.contact_timezone)).hour
    return settings.contact_window_start_hour <= local_hour < settings.contact_window_end_hour
