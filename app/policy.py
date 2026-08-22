"""The recovery policy: given a case, decide whether to contact the customer.

Deliberately pure -- no database, no clock, no network. ``now`` and ``settings``
are passed in, so every rule below is exercised by a plain unit test. This is
also where the "bounded workflow" and "compliant escalation" requirements live:
a case can only ever leave here as CALL, WAIT or STOP, and STOP is permanent.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

from app.config import Settings
from app.constants import CaseStatus
from app.models import Customer, RecoveryCase


class Action(StrEnum):
    CALL = "call"
    WAIT = "wait"
    STOP = "stop"


class StopReason(StrEnum):
    ALREADY_CLOSED = "already_closed"
    MAX_ATTEMPTS_REACHED = "max_attempts_reached"
    BELOW_MIN_AMOUNT = "below_min_amount"
    NO_CONTACT_NUMBER = "no_contact_number"


class WaitReason(StrEnum):
    WITHIN_BACKOFF = "within_backoff"
    OUTSIDE_CONTACT_WINDOW = "outside_contact_window"


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str

    def as_metadata(self) -> dict[str, str]:
        """Shape written to the audit trail."""
        return {"action": str(self.action), "reason": self.reason}


CLOSED_STATUSES = {CaseStatus.RECOVERED, CaseStatus.DECLINED, CaseStatus.STOPPED}


def decide(
    case: RecoveryCase,
    customer: Customer | None,
    *,
    now: datetime,
    settings: Settings,
) -> Decision:
    """Decide the next step for one case.

    Rule order matters: every STOP condition is checked before every WAIT
    condition, so a case that has exhausted its attempts is closed out rather
    than parked forever waiting for the contact window to open.
    """
    if case.status in CLOSED_STATUSES:
        return Decision(Action.STOP, StopReason.ALREADY_CLOSED)

    if case.attempt_count >= case.max_attempts:
        return Decision(Action.STOP, StopReason.MAX_ATTEMPTS_REACHED)

    if case.original_amount < settings.min_recoverable_amount_paise:
        return Decision(Action.STOP, StopReason.BELOW_MIN_AMOUNT)

    if customer is None or not customer.phone:
        return Decision(Action.STOP, StopReason.NO_CONTACT_NUMBER)

    if _within_backoff(case, now=now, settings=settings):
        return Decision(Action.WAIT, WaitReason.WITHIN_BACKOFF)

    if not _within_contact_window(now, settings=settings):
        return Decision(Action.WAIT, WaitReason.OUTSIDE_CONTACT_WINDOW)

    return Decision(Action.CALL, "eligible")


def _within_backoff(case: RecoveryCase, *, now: datetime, settings: Settings) -> bool:
    if case.last_attempt_at is None:
        return False
    return now - case.last_attempt_at < timedelta(hours=settings.retry_backoff_hours)


def _within_contact_window(now: datetime, *, settings: Settings) -> bool:
    """True if the customer's local time is inside the permitted calling hours."""
    local_hour = now.astimezone(ZoneInfo(settings.contact_timezone)).hour
    return settings.contact_window_start_hour <= local_hour < settings.contact_window_end_hour
