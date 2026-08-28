"""Batch recovery metrics -- the number Track 03 is actually judged on.

Everything here is derived from ``recovery_cases`` and ``voice_calls``; nothing
is tracked incrementally, so the figures cannot drift out of sync with the
underlying rows. Amounts stay in paise until the moment they are formatted.

The honest definitions matter more than the queries:

* **at risk** is every case we opened, including the ones policy later declined
  to work. Excluding those would flatter the recovery rate.
* **recovered** counts only cases closed by a real ``payment.captured`` or
  ``subscription.charged`` webhook. A customer promising to pay does not count.
* the headline rate is **by amount**, not by count, because recovering one
  Rs 5,000 subscription is not the same as recovering one Rs 50 one.
* a **promise is kept** only when the money landed before the deadline the
  customer was given. Paying a week late is a recovery, but it is not a kept
  promise, and conflating the two would make the agent look more persuasive
  than it is. Promises whose deadline has not passed are counted separately as
  *pending* rather than silently scored as broken.

Promises are counted from the ``promise_made`` **audit rows**, not from the two
columns on the case. The columns hold only the live promise, so a customer who
breaks one and then promises again on a later call would overwrite the broken
one and be scored on the second -- biasing the kept rate upward on exactly the
population it exists to catch. One row per promise is the honest denominator.
"""

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import DateTime, Integer, case, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import ActionType, CaseSource, CaseStatus
from app.diagnosis import classify
from app.models import RecoveryAction, RecoveryCase, VoiceCall
from app.store import utcnow


def rupees(paise: int | None) -> str:
    """Format paise as rupees with Indian thousands grouping."""
    value = (paise or 0) / 100
    whole, _, frac = f"{value:.2f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"₹{whole}.{frac}"


@dataclass
class BatchMetrics:
    total_cases: int = 0
    recovered_cases: int = 0
    amount_at_risk: int = 0
    amount_recovered: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    amount_by_status: dict[str, int] = field(default_factory=dict)
    by_failure_code: dict[str, int] = field(default_factory=dict)
    #: Cases grouped by *why* the charge failed rather than by Razorpay's
    #: coarse error class -- the view that says which interventions the batch
    #: actually needed.
    by_root_cause: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    calls_placed: int = 0
    calls_by_intent: dict[str, int] = field(default_factory=dict)
    median_attempts_to_recover: float | None = None
    #: Promise-to-pay tracking, counted per *promise* -- one customer who
    #: promises twice counts twice, because they made two commitments.
    promises_made: int = 0
    promises_kept: int = 0
    promises_pending: int = 0
    #: How many distinct cases those promises came from, and what they were
    #: worth. Amount is per case, not per promise: promising the same debt
    #: twice does not double the debt.
    cases_with_a_promise: int = 0
    amount_promised: int = 0

    @property
    def recovery_rate_by_amount(self) -> float:
        """The headline number. 0.0 when nothing was at risk."""
        if not self.amount_at_risk:
            return 0.0
        return self.amount_recovered / self.amount_at_risk

    @property
    def recovery_rate_by_count(self) -> float:
        if not self.total_cases:
            return 0.0
        return self.recovered_cases / self.total_cases

    @property
    def amount_outstanding(self) -> int:
        return self.amount_at_risk - self.amount_recovered

    @property
    def promises_broken(self) -> int:
        """Deadline passed, money never arrived."""
        return self.promises_made - self.promises_kept - self.promises_pending

    @property
    def promise_kept_rate(self) -> float | None:
        """Scored over *resolved* promises only. None when none have resolved.

        Counting still-open promises as broken would punish a batch for having
        been run recently, which says nothing about whether the agent works.
        And a batch where nothing has resolved yet has no rate -- printing
        0.0% next to "broken: 0" would have the report contradict itself.
        """
        resolved = self.promises_made - self.promises_pending
        if not resolved:
            return None
        return self.promises_kept / resolved

    def as_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "recovered_cases": self.recovered_cases,
            "amount_at_risk_paise": self.amount_at_risk,
            "amount_recovered_paise": self.amount_recovered,
            "amount_outstanding_paise": self.amount_outstanding,
            "recovery_rate_by_amount": round(self.recovery_rate_by_amount, 4),
            "recovery_rate_by_count": round(self.recovery_rate_by_count, 4),
            "by_status": self.by_status,
            "amount_by_status_paise": self.amount_by_status,
            "by_failure_code": self.by_failure_code,
            "by_root_cause": self.by_root_cause,
            "by_source": self.by_source,
            "calls_placed": self.calls_placed,
            "calls_by_intent": self.calls_by_intent,
            "median_attempts_to_recover": self.median_attempts_to_recover,
            "promises_made": self.promises_made,
            "cases_with_a_promise": self.cases_with_a_promise,
            "promises_kept": self.promises_kept,
            "promises_pending": self.promises_pending,
            "promises_broken": self.promises_broken,
            "promise_kept_rate": (
                None if self.promise_kept_rate is None else round(self.promise_kept_rate, 4)
            ),
            "amount_promised_paise": self.amount_promised,
        }


async def compute_metrics(
    session: AsyncSession,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    source: str | None = None,
    now: datetime | None = None,
) -> BatchMetrics:
    """Aggregate the batch.

    ``since``/``until`` bound it by case creation time. ``source`` restricts it
    to real ('razorpay') or seeded ('seed') cases -- never report a figure that
    silently blends the two.

    ``now`` is injectable because one figure here is not purely derived from
    the rows: whether an unpaid promise is *broken* or merely *pending* depends
    on the clock.
    """
    metrics = BatchMetrics()
    now = now or utcnow()

    def scoped(stmt, column=RecoveryCase.created_at, case_scoped=True):
        if since is not None:
            stmt = stmt.where(column >= since)
        if until is not None:
            stmt = stmt.where(column < until)
        if source is not None and case_scoped:
            stmt = stmt.where(RecoveryCase.source == source)
        return stmt

    totals = await session.execute(
        scoped(
            select(
                func.count(RecoveryCase.id),
                func.coalesce(func.sum(RecoveryCase.original_amount), 0),
                func.coalesce(func.sum(RecoveryCase.recovered_amount), 0),
                func.coalesce(
                    func.sum(
                        case((RecoveryCase.status == CaseStatus.RECOVERED, 1), else_=0)
                    ),
                    0,
                ),
            )
        )
    )
    (
        metrics.total_cases,
        metrics.amount_at_risk,
        metrics.amount_recovered,
        metrics.recovered_cases,
    ) = totals.one()

    status_rows = await session.execute(
        scoped(
            select(
                RecoveryCase.status,
                func.count(RecoveryCase.id),
                func.coalesce(func.sum(RecoveryCase.original_amount), 0),
            ).group_by(RecoveryCase.status)
        )
    )
    for status, count, amount in status_rows:
        metrics.by_status[status] = count
        metrics.amount_by_status[status] = amount

    # Group on the raw column: coalescing inside both SELECT and GROUP BY emits
    # two different bind params, which Postgres will not treat as the same
    # expression. Nulls are mapped here instead.
    failure_rows = await session.execute(
        scoped(
            select(RecoveryCase.failure_code, func.count(RecoveryCase.id))
            .group_by(RecoveryCase.failure_code)
            .order_by(func.count(RecoveryCase.id).desc())
        )
    )
    metrics.by_failure_code = {(code or "unknown"): count for code, count in failure_rows}

    # Grouped in SQL on the two raw columns, then folded into causes in Python:
    # the mapping lives in app.diagnosis and must not be duplicated as a CASE
    # expression that could drift away from the rules the policy actually runs.
    cause_rows = await session.execute(
        scoped(
            select(
                RecoveryCase.failure_source,
                RecoveryCase.failure_reason_code,
                func.count(RecoveryCase.id),
            ).group_by(RecoveryCase.failure_source, RecoveryCase.failure_reason_code)
        )
    )
    causes: dict[str, int] = {}
    for failure_source, reason_code, count in cause_rows:
        cause = str(classify(failure_source, reason_code))
        causes[cause] = causes.get(cause, 0) + count
    metrics.by_root_cause = dict(sorted(causes.items(), key=lambda kv: -kv[1]))

    source_rows = await session.execute(
        scoped(select(RecoveryCase.source, func.count(RecoveryCase.id)).group_by(
            RecoveryCase.source
        ))
    )
    metrics.by_source = {src: count for src, count in source_rows}

    call_total = await session.execute(
        scoped(
            select(func.count(VoiceCall.id)).join(
                RecoveryCase, RecoveryCase.id == VoiceCall.recovery_case_id
            ),
            column=VoiceCall.created_at,
        )
    )
    metrics.calls_placed = call_total.scalar_one()

    intent_rows = await session.execute(
        scoped(
            select(VoiceCall.detected_intent, func.count(VoiceCall.id))
            .join(RecoveryCase, RecoveryCase.id == VoiceCall.recovery_case_id)
            .group_by(VoiceCall.detected_intent),
            column=VoiceCall.created_at,
        )
    )
    metrics.calls_by_intent = {(intent or "none"): count for intent, count in intent_rows}

    # One row per promise, joined back to its case for the payment date. The
    # deadline lives in the action's metadata because it is a property of that
    # promise, not of the case -- the case only ever remembers the latest one.
    due_at = cast(RecoveryAction.metadata_json["due_at"].astext, DateTime(timezone=True))
    promises = await session.execute(
        scoped(
            select(
                func.count(),
                func.coalesce(
                    func.sum(
                        case(
                            (
                                RecoveryCase.recovered_at.isnot(None)
                                & (RecoveryCase.recovered_at <= due_at),
                                1,
                            ),
                            else_=0,
                        )
                    ),
                    0,
                ),
                func.coalesce(
                    func.sum(
                        case(
                            (RecoveryCase.recovered_at.is_(None) & (due_at > now), 1),
                            else_=0,
                        )
                    ),
                    0,
                ),
                func.count(func.distinct(RecoveryAction.recovery_case_id)),
            )
            .select_from(RecoveryAction)
            .join(RecoveryCase, RecoveryCase.id == RecoveryAction.recovery_case_id)
            .where(RecoveryAction.action_type == ActionType.PROMISE_MADE)
            .where(RecoveryAction.metadata_json["due_at"].astext.isnot(None))
        )
    )
    (
        metrics.promises_made,
        metrics.promises_kept,
        metrics.promises_pending,
        metrics.cases_with_a_promise,
    ) = promises.one()

    promised_amount = await session.execute(
        scoped(
            select(func.coalesce(func.sum(RecoveryCase.original_amount), 0)).where(
                RecoveryCase.promised_at.isnot(None)
            )
        )
    )
    metrics.amount_promised = promised_amount.scalar_one()

    median = await session.execute(
        scoped(
            select(
                func.percentile_cont(0.5).within_group(
                    RecoveryCase.attempt_count.cast(Integer)
                )
            ).where(RecoveryCase.status == CaseStatus.RECOVERED)
        )
    )
    metrics.median_attempts_to_recover = median.scalar_one_or_none()

    return metrics


def _percent_or_na(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.1%}"


def format_report(metrics: BatchMetrics) -> str:
    """Human-readable summary, for the README and the pitch."""
    lines = [
        "Recovery batch summary",
        "=" * 46,
        f"  Cases opened          {metrics.total_cases:>10}",
        f"  Cases recovered       {metrics.recovered_cases:>10}",
        f"  At risk               {rupees(metrics.amount_at_risk):>10}",
        f"  Recovered             {rupees(metrics.amount_recovered):>10}",
        f"  Outstanding           {rupees(metrics.amount_outstanding):>10}",
        "",
        f"  Recovery rate (₹)     {metrics.recovery_rate_by_amount:>9.1%}",
        f"  Recovery rate (cases) {metrics.recovery_rate_by_count:>9.1%}",
    ]
    if metrics.median_attempts_to_recover is not None:
        lines.append(f"  Median attempts       {metrics.median_attempts_to_recover:>10.1f}")

    if metrics.by_status:
        lines += ["", "  By status"]
        for status, count in sorted(metrics.by_status.items()):
            amount = rupees(metrics.amount_by_status.get(status, 0))
            lines.append(f"    {status:<16} {count:>5}   {amount:>12}")

    # Only seeded data poisons a total. A batch spanning failed subscriptions
    # and abandoned checkouts is two kinds of real money and adds up fine.
    if CaseSource.SEED in metrics.by_source and len(metrics.by_source) > 1:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(metrics.by_source.items()))
        lines += [
            "",
            f"  !! MIXED SOURCES ({counts}) -- this total blends real and seeded",
            f"     cases. Re-run with --source {CaseSource.RAZORPAY} or --source"
            f" {CaseSource.SEED}.",
        ]

    if metrics.promises_made:
        lines += [
            "",
            "  Promises to pay",
            f"    made               {metrics.promises_made:>5}   "
            f"across {metrics.cases_with_a_promise} cases, "
            f"{rupees(metrics.amount_promised)} at stake",
            f"    kept               {metrics.promises_kept:>5}",
            f"    broken             {metrics.promises_broken:>5}",
            f"    pending            {metrics.promises_pending:>5}",
            f"    kept rate          {_percent_or_na(metrics.promise_kept_rate):>8}   "
            f"(of resolved)",
        ]

    if metrics.by_root_cause:
        lines += ["", "  By root cause"]
        for cause, count in metrics.by_root_cause.items():
            lines.append(f"    {cause:<32} {count:>5}")

    if metrics.by_failure_code:
        lines += ["", "  By failure code"]
        for code, count in metrics.by_failure_code.items():
            lines.append(f"    {code:<32} {count:>5}")

    if metrics.calls_placed:
        lines += ["", f"  Calls placed          {metrics.calls_placed:>10}"]
        for intent, count in sorted(metrics.calls_by_intent.items()):
            lines.append(f"    {intent:<16} {count:>5}")

    return "\n".join(lines)
