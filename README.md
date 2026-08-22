# Dunning Agent

Razorpay AI Buildathon — **Track 03: AI Revenue Recovery**.

Detects subscription revenue at risk, decides on an intervention, and executes a
bounded recovery workflow with stopping rules and a full audit trail.

**Status: Phase 1 complete** — the trigger layer (webhook intake, correlation,
case creation) is built and tested. See [Roadmap](#roadmap).

---

## The event model (read this first)

There is **no `subscription.charged.failed` webhook**. That event does not exist
in Razorpay's API, and building the trigger layer around it is the most common
way this integration goes wrong. A failed recurring charge actually surfaces as
three separate signals:

| Event | Meaning | What we do |
|---|---|---|
| `payment.failed` | One charge attempt failed | Open a recovery case |
| `subscription.pending` | Razorpay is retrying on its own | Append to the case's audit trail |
| `subscription.halted` | Razorpay gave up retrying | Stamp `halted_at` — the hard escalation trigger |
| `subscription.charged` | A recurring charge succeeded | Close open cases as recovered |
| `payment.captured` | Any payment succeeded | Attribute to a case and close it |

The second trap: **`payload.payment.entity` contains no `customer_id` and no
`subscription_id`.** The only linkage a failed payment carries is `invoice_id`.
So the pipeline resolves `payment.failed → invoice_id → GET /invoices/{id} →
subscription_id + customer_id` before it can open a case. That lookup is in
[`handlers._resolve_invoice_context`](app/webhooks/handlers.py).

## Architecture

```
Razorpay ──webhook──▶ FastAPI  ──┐
                                 │ 1. verify HMAC-SHA256 over the raw body
                                 │ 2. INSERT envelope (dedupe on event id)
                                 │ 3. return 200
                                 ▼
                          background task
                                 │
                                 ├─▶ resolve invoice → subscription + customer
                                 ├─▶ upsert payment / subscription / customer
                                 └─▶ open recovery_case + append recovery_action
```

Intake is deliberately two-stage. The envelope is persisted **synchronously**
before any handler runs, so a crash mid-processing cannot lose an event —
`webhook_events.processed_at IS NULL` is the replay queue, and Razorpay's own
at-least-once redelivery is a backstop rather than the only safety net.

### Tables

| Table | Purpose |
|---|---|
| `webhook_events` | Raw verified envelopes; dedupe key and replay queue |
| `customers` / `subscriptions` / `payments` | Mirrored Razorpay state |
| `recovery_cases` | One row per recoverable failure — the unit metrics are computed over |
| `recovery_actions` | Append-only audit trail; never updated, never deleted |

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
uv sync
cp .env.example .env          # then fill in your Razorpay test keys
docker compose up -d db       # Postgres on localhost:5433
uv run alembic upgrade head
```

> Port 5433, not 5432 — this machine already had a Postgres on 5432 and
> `localhost:5432` silently resolved to it instead of the container.

### Run

```bash
uv run uvicorn app.main:app --reload
curl http://localhost:8000/healthz
```

### Point Razorpay at it

1. Expose the port publicly (`cloudflared tunnel --url http://localhost:8000`,
   ngrok, or a deployed Cloud Run URL).
2. Dashboard → Settings → Webhooks → add `https://<host>/webhooks/razorpay`.
3. Set the secret to the same value as `RAZORPAY_WEBHOOK_SECRET` in `.env`.
4. Subscribe to: `payment.failed`, `payment.captured`, `subscription.pending`,
   `subscription.halted`, `subscription.charged`.

### Local smoke test

```bash
# Signed request, one-off failure — exercises signature + storage, no API call
uv run python -m scripts.send_test_webhook --event payment.failed --invoice-id ""

# Rejected: bad signature
uv run python -m scripts.send_test_webhook --tamper
```

To exercise the full case-creation path locally, pass `--invoice-id` for an
invoice that really exists in your test account — the handler fetches it to
resolve the subscription.

## Tests

```bash
uv run pytest
```

23 tests. Signature tests are pure unit tests; the rest need the Postgres
container and are skipped automatically if it is not reachable. They cover
webhook redelivery (no duplicate cases), one-off vs subscription failures, the
escalation stamp, payment-link attribution via `notes.recovery_case_id`, and the
double-counting hazard where `subscription.charged` and `payment.captured` both
fire for a single recovery.

## Deploy (GCP)

```bash
gcloud run deploy dunning-agent \
  --source . \
  --region asia-south1 \
  --add-cloudsql-instances PROJECT:REGION:INSTANCE \
  --set-env-vars "DATABASE_URL=postgresql+asyncpg://USER:PASS@/recovery?host=/cloudsql/PROJECT:REGION:INSTANCE" \
  --set-secrets "RAZORPAY_KEY_SECRET=razorpay-key-secret:latest,RAZORPAY_WEBHOOK_SECRET=razorpay-webhook-secret:latest"
```

`/healthz` does a real database round-trip so Cloud Run will not route to an
instance that cannot reach Cloud SQL.

## Roadmap

- [x] **Phase 1** — Webhook intake, signature verification, idempotency,
      invoice correlation, case creation, audit trail
- [ ] **Phase 2** — Policy engine and orchestrator (who to contact, when, how
      many times; stopping rules)
- [ ] **Phase 3** — LiveKit outbound voice agent (Hinglish/Hindi/English), intent
      capture
- [ ] **Phase 4** — Razorpay Payment Links for the `retry_now` intent, recovery
      attribution
- [ ] **Phase 5** — Batch metrics: cases, ₹ at risk, ₹ recovered, recovery rate

## Known gaps

- **`subscription.pending` with no open case.** If Razorpay's `payment.failed`
  never reaches us (endpoint down past the retry window), the subscription goes
  pending with nothing to attach to. Currently logged as a warning; Phase 2 adds
  a reconciler that backfills from `GET /invoices?subscription_id=`.
- **Background tasks, not a queue.** Processing runs in-process via FastAPI
  `BackgroundTasks`. The durable envelope means nothing is lost, but nothing
  automatically retries yet — the replay sweep lands with the Phase 2 worker.
- **No outbound telephony yet.** LiveKit Cloud alone does not place PSTN calls;
  that needs a SIP trunk (Twilio/Plivo) wired to LiveKit SIP. Phase 3.
