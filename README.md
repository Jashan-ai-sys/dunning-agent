# Dunning Agent

Razorpay AI Buildathon — **Track 03: AI Revenue Recovery**.

Detects subscription revenue at risk, decides on an intervention, and executes a
bounded recovery workflow with stopping rules and a full audit trail.

**Status: Phases 1–2 complete and deployed; Phase 3 partly built.** The trigger
layer, the policy engine and the orchestrator are live on Cloud Run against
Cloud SQL, with the worker ticking every 5 minutes via Cloud Scheduler. The
voice agent's conversation graph, intent model and post-call handling are built
and tested; the LiveKit runtime that walks the graph is not. See
[Roadmap](#roadmap).

Webhook endpoint:
`https://dunning-agent-862702522215.asia-south1.run.app/webhooks/razorpay`

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

A separate worker loop drives recovery forward:

```
worker tick (every 5 min in prod, 60s locally)
  │
  ├─▶ replay sweep ── re-dispatch envelopes whose handler died
  │
  └─▶ orchestrator ── claim open cases (FOR UPDATE SKIP LOCKED)
                        │
                        ▼
                    policy.decide()  ──▶  CALL / WAIT / STOP
                        │
                        └─▶ execute + append to the audit trail
```

## The recovery policy

Every case leaves the policy as exactly one of **CALL**, **WAIT** or **STOP**,
and STOP is permanent. That is what makes the workflow bounded. The rules, in
evaluation order — all STOP conditions are checked before any WAIT condition, so
a dead case gets closed out rather than parked forever outside the calling
window:

| # | Rule | Outcome |
|---|---|---|
| 1 | Case already recovered / declined / stopped | **STOP** `already_closed` |
| 2 | `attempt_count >= max_attempts` (default 3) | **STOP** `max_attempts_reached` |
| 3 | Amount below `MIN_RECOVERABLE_AMOUNT_PAISE` (default ₹50) | **STOP** `below_min_amount` |
| 4 | No phone number on file | **STOP** `no_contact_number` |
| 5 | Last attempt within `RETRY_BACKOFF_HOURS` (default 24h) | **WAIT** `within_backoff` |
| 6 | Outside 09:00–21:00 in `CONTACT_TIMEZONE` | **WAIT** `outside_contact_window` |
| — | Otherwise | **CALL** |

Rule 6 is the compliance rule: TRAI restricts commercial calls to 09:00–21:00,
and the window is evaluated in the customer's local time rather than UTC —
getting that wrong means calling people at 3am. Rule 5 is what stops a
once-a-minute tick from becoming a once-a-minute call.

[`app/policy.py`](app/policy.py) is pure — no database, no clock, no network.
`now` and `settings` are arguments, so every rule above is covered by a plain
unit test.

Cases Razorpay has itself given up on (`halted_at` set, from
`subscription.halted`) are worked first. Contact is made through the
`ContactChannel` protocol, so Phase 3 swaps in LiveKit without touching the
policy or the loop; until then `LoggingChannel` records the intent and is named
`logging` in the audit trail so a run can never be mistaken for evidence that a
customer was actually called.

### Tables

| Table | Purpose |
|---|---|
| `webhook_events` | Raw verified envelopes; dedupe key and replay queue |
| `customers` / `subscriptions` / `payments` | Mirrored Razorpay state |
| `recovery_cases` | One row per recoverable failure — the unit metrics are computed over |
| `recovery_actions` | Append-only audit trail; never updated, never deleted |
| `voice_calls` | One call attempt: room, duration, transcript, terminal node, intent |

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
curl http://localhost:8000/health
```

The worker runs as a separate process:

```bash
uv run python -m app.worker
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

97 tests. The signature, policy and conversation-graph tests are pure unit tests;
the rest need the Postgres container and are skipped automatically if it is not
reachable.

They cover webhook redelivery (no duplicate cases), one-off vs subscription
failures, the escalation stamp, payment-link attribution via
`notes.recovery_case_id`, and the double-counting hazard where
`subscription.charged` and `payment.captured` both fire for a single recovery.
On the policy and orchestrator side: every rule and its boundary, the contact
window evaluated across five times of day, backoff suppressing a second call,
STOP being permanent, waiting cases writing no audit noise, a channel outage
neither burning the customer's attempt budget nor aborting the rest of the
batch, and the replay sweep leaving in-flight events alone while recovering
stale ones.

## Deploy (GCP)

Three pieces: a Cloud Run **service** for the webhook, a Cloud Run **job** for
migrations, and a Cloud Run **job on a Cloud Scheduler trigger** for the worker.
The worker cannot be part of the service — Cloud Run scales to zero and would
kill a long-lived background loop.

The connection string is stored whole in Secret Manager rather than assembled
from a plaintext env var, so the database password never appears in the Cloud
Run config or in shell history.

```bash
# 1. the webhook service
gcloud run deploy dunning-agent --source . \
  --region asia-south1 --allow-unauthenticated \
  --add-cloudsql-instances PROJECT:REGION:INSTANCE \
  --set-env-vars "RAZORPAY_KEY_ID=rzp_test_xxx,LOG_LEVEL=INFO" \
  --set-secrets "DATABASE_URL=database-url:latest,\
RAZORPAY_KEY_SECRET=razorpay-key-secret:latest,\
RAZORPAY_WEBHOOK_SECRET=razorpay-webhook-secret:latest"

# 2. migrations, as a one-off job off the same image
gcloud run jobs create dunning-migrate --image "$IMAGE" \
  --region asia-south1 \
  --set-cloudsql-instances PROJECT:REGION:INSTANCE \
  --set-secrets "DATABASE_URL=database-url:latest" \
  --command alembic --args "upgrade,head"
gcloud run jobs execute dunning-migrate --wait

# 3. the worker, one tick per invocation, every 5 minutes
gcloud run jobs create dunning-worker --image "$IMAGE" ... \
  --command python --args "-m,app.worker,--once"
gcloud scheduler jobs create http dunning-worker-tick \
  --schedule "*/5 * * * *" --time-zone Asia/Kolkata \
  --uri "https://run.googleapis.com/v2/projects/PROJECT/locations/REGION/jobs/dunning-worker:run" \
  --http-method POST --oauth-service-account-email "$SA"
```

`/health` does a real database round-trip so Cloud Run will not route to an
instance that cannot reach Cloud SQL.

> **Not `/healthz`.** Google's front end intercepts that exact path on
> `*.run.app` and answers 404 itself in ~35ms; the request never reaches the
> container. `/healthzz` and `/health` both get through — only `/healthz` does
> not. Cloud Run still reports the revision as Ready, so this fails silently.

## Roadmap

- [x] **Phase 1** — Webhook intake, signature verification, idempotency,
      invoice correlation, case creation, audit trail
- [x] **Phase 2** — Policy engine, orchestrator, stopping rules, contact-window
      compliance, replay sweep
- [~] **Phase 3** — Conversation graph, intent model, `voice_calls`, post-call
      outcome handling **done**; the LiveKit runtime that walks the graph and
      the SIP dispatch are **not**
- [ ] **Phase 4** — Razorpay Payment Links for the `retry_now` intent, recovery
      attribution
- [ ] **Phase 5** — Batch metrics: cases, ₹ at risk, ₹ recovered, recovery rate

## Known gaps

- **`subscription.pending` with no open case — still open.** If Razorpay's
  `payment.failed` never reaches us (endpoint down past its retry window), the
  subscription goes pending with nothing to attach to. Still only logged as a
  warning: the reconciler that would backfill from
  `GET /invoices?subscription_id=` is not built. The replay sweep does not cover
  this, because it only retries events we actually received.
- **Background tasks, not a queue.** Webhook processing runs in-process via
  FastAPI `BackgroundTasks`. The durable envelope plus the worker's replay sweep
  means nothing is lost and stalled events are retried, but there is no backoff
  or dead-letter queue — an event whose handler fails permanently is retried
  every tick. Fine at hackathon volume; it would need a real queue in
  production.
- **No outbound telephony yet.** The conversation graph, intents and post-call
  handling are built and tested, but nothing places a call. LiveKit Cloud alone
  does not do PSTN — that needs a SIP trunk (Twilio/Plivo/Exotel) wired to
  LiveKit SIP, and for Indian numbers, DLT registration. Until then the
  orchestrator uses `LoggingChannel`, which records the intent to call and
  explicitly does **not** pretend a call happened.
- **The app connects to Cloud SQL as the `postgres` superuser.** A dedicated
  least-privilege role is the right answer; it was skipped because setting up
  grants needs a `psql` session the deploy box did not have. The credential
  lives in Secret Manager, not in the repo, but this is a real shortcut.
- **No backups on the Cloud SQL instance** (`--no-backup`, zonal) to keep the
  hackathon cost at the floor. Do not copy that into anything real.
