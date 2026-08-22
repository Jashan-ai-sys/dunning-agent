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
"""

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import Integer, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import CaseStatus
from app.models import RecoveryCase, VoiceCall


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
    calls_placed: int = 0
    calls_by_intent: dict[str, int] = field(default_factory=dict)
    median_attempts_to_recover: float | None = None

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
            "calls_placed": self.calls_placed,
            "calls_by_intent": self.calls_by_intent,
            "median_attempts_to_recover": self.median_attempts_to_recover,
        }


async def compute_metrics(
    session: AsyncSession,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> BatchMetrics:
    """Aggregate the batch. ``since``/``until`` bound it by case creation time."""
    metrics = BatchMetrics()

    def scoped(stmt, column=RecoveryCase.created_at):
        if since is not None:
            stmt = stmt.where(column >= since)
        if until is not None:
            stmt = stmt.where(column < until)
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

    call_total = await session.execute(
        scoped(select(func.count(VoiceCall.id)), column=VoiceCall.created_at)
    )
    metrics.calls_placed = call_total.scalar_one()

    intent_rows = await session.execute(
        scoped(
            select(VoiceCall.detected_intent, func.count(VoiceCall.id)).group_by(
                VoiceCall.detected_intent
            ),
            column=VoiceCall.created_at,
        )
    )
    metrics.calls_by_intent = {(intent or "none"): count for intent, count in intent_rows}

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

    if metrics.by_failure_code:
        lines += ["", "  By failure code"]
        for code, count in metrics.by_failure_code.items():
            lines.append(f"    {code:<32} {count:>5}")

    if metrics.calls_placed:
        lines += ["", f"  Calls placed          {metrics.calls_placed:>10}"]
        for intent, count in sorted(metrics.calls_by_intent.items()):
            lines.append(f"    {intent:<16} {count:>5}")

    return "\n".join(lines)
