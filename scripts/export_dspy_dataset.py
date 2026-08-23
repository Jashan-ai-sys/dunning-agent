"""Export labelled graph turns as a DSPy-ready JSONL dataset.

    uv run python -m scripts.export_dspy_dataset --out turns.jsonl
    uv run python -m scripts.export_dspy_dataset --node ask_intent --accepted-only

Each row is one decision the model made during a real call:

    {"node_id": "ask_intent",
     "utterance": "abhi nahi yaar, kal karta hoon",
     "options": ["pay_now", "pay_later", "declined", "financial_difficulty", ...],
     "label": "pay_later",
     "accepted": true}

That maps onto a DSPy signature of ``(node_id, utterance, options) -> label``,
with exact-match on ``label`` as the metric. Optimisers like GEPA and MIPROv2
then tune the edge-condition wording against our own traffic instead of
against examples we invented.

Two deliberate choices:

* ``accepted: false`` rows are kept by default. A label the model reached for
  and could not have is a harder negative than anything we would write by hand
  -- and getting those to zero is a real objective.
* Rows with no utterance are dropped. A decision with no input is not a
  training example; it usually means the model moved on its own opening turn.

Nothing here runs during a call. Optimise offline, then paste the improved
instruction into the node's edge conditions.
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal, engine
from app.models import VoiceCall
from app.voice.flow import DUNNING_FLOW

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def options_for(node_id: str) -> list[str]:
    """The labels that were legal at that node, so the signature sees choices."""
    try:
        return [edge.label for edge in DUNNING_FLOW.node(node_id).edges]
    except Exception:  # noqa: BLE001 - a node may have been renamed since
        return []


async def collect(*, node: str | None, accepted_only: bool) -> list[dict]:
    rows: list[dict] = []
    async with SessionLocal() as session:
        result = await session.execute(
            select(VoiceCall.transitions).where(VoiceCall.transitions.isnot(None))
        )
        for (transitions,) in result:
            for turn in transitions or []:
                if not turn.get("utterance"):
                    continue
                if node and turn.get("node_id") != node:
                    continue
                if accepted_only and not turn.get("accepted"):
                    continue
                rows.append(
                    {
                        "node_id": turn.get("node_id"),
                        "utterance": turn.get("utterance"),
                        "options": options_for(turn.get("node_id", "")),
                        "label": turn.get("label"),
                        "accepted": bool(turn.get("accepted")),
                    }
                )
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="turns.jsonl", help="output JSONL path")
    parser.add_argument("--node", help="only export turns from this node")
    parser.add_argument(
        "--accepted-only", action="store_true", help="drop rejected label attempts"
    )
    args = parser.parse_args()

    try:
        rows = await collect(node=args.node, accepted_only=args.accepted_only)
    finally:
        await engine.dispose()

    if not rows:
        print("No labelled turns yet. This fills up once real calls run.")
        return

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_node = Counter(r["node_id"] for r in rows)
    rejected = sum(1 for r in rows if not r["accepted"])
    print(f"wrote {len(rows)} turns to {out}")
    for node_id, count in by_node.most_common():
        print(f"  {node_id:<16} {count:>4}")
    if rejected:
        print(f"  ({rejected} rejected label attempts kept as hard negatives)")
    if len(rows) < 50:
        print("\nFewer than ~50 turns: too few to optimise against meaningfully.")


if __name__ == "__main__":
    asyncio.run(main())
