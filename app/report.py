"""Print the recovery batch metrics.

    uv run python -m app.report            # human-readable
    uv run python -m app.report --json     # machine-readable
    uv run python -m app.report --csv out.csv

Runs against whatever DATABASE_URL points at, so the same command works locally
and as a Cloud Run Job against Cloud SQL.
"""

import argparse
import asyncio
import csv
import json
import sys
from datetime import UTC, datetime

from app.constants import CaseSource
from app.db import SessionLocal, engine
from app.metrics import compute_metrics, format_report

# The report is full of rupee signs and the Windows console defaults to cp1252,
# which cannot encode U+20B9. Without this the command dies on its own output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--csv", metavar="PATH", help="also write a flat CSV summary")
    parser.add_argument("--since", type=_parse_date, help="ISO date, inclusive")
    parser.add_argument("--until", type=_parse_date, help="ISO date, exclusive")
    parser.add_argument("--source", choices=[s.value for s in CaseSource],
                        help="report on one source only: a failed subscription "
                             "charge, an abandoned checkout, or seeded demo data")
    args = parser.parse_args()

    try:
        async with SessionLocal() as session:
            metrics = await compute_metrics(
                session, since=args.since, until=args.until, source=args.source
            )
    finally:
        await engine.dispose()

    print(json.dumps(metrics.as_dict(), indent=2) if args.json else format_report(metrics))

    if args.csv:
        payload = metrics.as_dict()
        flat = {k: v for k, v in payload.items() if not isinstance(v, dict)}
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat))
            writer.writeheader()
            writer.writerow(flat)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    asyncio.run(main())
