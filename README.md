# Dunning Agent

Razorpay AI Buildathon — **Track 03: AI Revenue Recovery**.

Detects subscription revenue at risk, decides on an intervention, and executes a
bounded recovery workflow with stopping rules and a full audit trail.

**Status: end to end, on real phone calls.** The trigger layer, policy engine,
orchestrator, payment links, mandate re-charging and batch metrics are live on
Cloud Run against Cloud SQL, with the worker ticking every 5 minutes via Cloud
Scheduler.

The voice agent places real calls over Twilio and holds the conversation in
Hindi. A call walks the conversation graph, records a labelled transition at
every turn, and writes a detected intent back onto the recovery case — the same
`apply_call_result` path a simulated batch uses, so nothing about the outcome is
special-cased for a live call.

Measured on production calls, not a laptop:

| | p50 | n |
| ------------------------------- | ------ | --- |
| STT time to first byte (Sarvam `saaras:v3`) | 0.396s | 6 |
| Turn detection (SmartTurn v3, ONNX, on CPU) | 0.173s | 7 |
| LLM time to first token (Gemini 2.5 Flash on Vertex) | 0.553s | 11 |
| TTS time to first byte (Cartesia) | 0.068s | 10 |
| TTS time to first audio | 0.209s | 10 |
| **Customer stops speaking → agent replies** | **~1.13s** | |

Measured on a single production call on 2026-09-04, from the Cloud Run logs.
An earlier run of this table reported 0.313s for the LLM; that figure predates
a change that runs a second inference after every graph transition, so the
model is now asked twice on those turns. It buys the turn that used to be
dropped — before it, the stage the agent had just moved to went undelivered
until the customer spoke again, measured at 5.7s of dead air. One extra round
trip is the cheaper of the two.

One turn in that sample returned in **0.045s**, an order of magnitude under the
median: Gemini's implicit cache hitting the stable prefix. That prefix is
deliberate — see `SYSTEM_STYLE` and the transition handler, which appends the
stage handoff as a tool result rather than rewriting the system message,
precisely so the prefix never shifts.

The agent can also send a payment link or re-authorise a dead mandate mid-call,
as MCP tools it is allowed to call while talking. See [Known gaps](#known-gaps)
for what is still rough — the honest list includes calls that end with no intent
recorded, and a Razorpay account setting that currently breaks the mandate link.

Webhook endpoint:
`https://dunning-agent-862702522215.asia-south1.run.app/webhooks/razorpay`

---

## Where to look

If you have five minutes, read [The event model](#the-event-model-read-this-first)
and [The recovery policy](#the-recovery-policy). Those two sections contain the
decisions; everything else is how they are wired up.

| I want to know… | Go to | Code |
|---|---|---|
| Why a failed subscription charge is three webhooks, not one | [The event model](#the-event-model-read-this-first) | [`webhooks/handlers.py`](app/webhooks/handlers.py) |
| How an event becomes a case without being lost | [Architecture](#architecture) | [`webhooks/processor.py`](app/webhooks/processor.py) |
| When we call, when we wait, and when we stop | [The recovery policy](#the-recovery-policy) | [`policy.py`](app/policy.py) |
| Why the model cannot invent an outcome | [The conversation is a graph](docs/architecture.md#the-conversation-is-a-graph-not-a-prompt) | [`voice/walker.py`](app/voice/walker.py) |
| Why the agent cannot name a case | [MCP tools](docs/architecture.md#mcp--what-the-agent-may-do) | [`mcp_server.py`](app/mcp_server.py) |
| Whether it survives more than one worker | [Running more than one worker](#running-more-than-one-worker) | [`orchestrator.py`](app/orchestrator.py) |
| What runs where, and the VAD/STT settings | [Architecture doc](docs/architecture.md) | — |
| What does **not** work yet | [Known gaps](#known-gaps) | — |

Four properties hold everywhere, and each is enforced in code rather than by
convention:

1. **The model cannot invent an outcome.** It picks an edge *label*, validated
   against the current node; an illegal one is refused, not followed.
2. **The agent cannot name a case.** The MCP tools take no arguments — the case
   is injected out of the model's reach.
3. **Simulated work can never be reported as recovered money.** `source` is a
   column, not a naming convention.
4. **Nobody is rung twice about the same debt.** `FOR UPDATE SKIP LOCKED`, with
   a concurrency test that reads the guarantee from both sides.

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
| `payment_link.paid` | A recovery link was paid | Credit the case by `reference_id` |

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

Three paths reach that envelope, and all three are safe because `process_event`
returns early once `processed_at` is set: the in-process background task, a
Pub/Sub push, and the cron sweep. The sweep alone was always correct — push is
an accelerator on a mechanism that already worked, and everything on that path
is written so that losing push costs latency and nothing else.

**The push endpoint authenticates with OIDC, not a shared secret.** Pub/Sub
push cannot send custom headers, so the only alternatives were a token in the
URL — which lands in every access log — or verifying the signed token Google
attaches. This endpoint dispatches handlers that create payment links and place
phone calls, so it gets the real one: the token is verified against the
subscription's audience, and the request is refused unless the claims carry the
expected service-account `email` *and* `email_verified`.

It also refuses to run **unauthenticated by omission**. If
`PUBSUB_PUSH_SERVICE_ACCOUNT` is unset the endpoint answers 503 rather than
accepting the request — a push endpoint that dispatches handlers is not
something to leave open because a setting was forgotten. And it answers 204 to
anything that must not be redelivered, including a message it cannot parse,
because Pub/Sub retries until acknowledged and a permanently malformed message
would otherwise come back forever.

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

`decide(case, customer, now, settings)` is a **pure function** — no database, no
clock, no network. `now` and `settings` are arguments, which is what makes every
rule below a plain unit test.

It returns exactly one decision, and the ladder is ordered: every **STOP** is
checked before any **WAIT**, so a dead case gets closed rather than parked
forever outside the calling window. First match wins.

| # | Condition | Outcome |
|---|---|---|
| 1 | Case already recovered / declined / stopped | **STOP** `already_closed` |
| 2 | Customer flagged `do_not_contact` | **STOP** `do_not_contact` |
| 3 | `attempt_count >= max_attempts` | **STOP** `max_attempts_reached` |
| 4 | `max_delivery_failures` consecutive send failures | **STOP** `undeliverable` |
| 5 | Below `MIN_RECOVERABLE_AMOUNT_PAISE` (₹50) | **STOP** `below_min_amount` |
| 6 | Root cause is `configuration` — our bug, not theirs | **STOP** `needs_human` |
| 7 | No usable contact details | **STOP** `no_contact_details` |
| 8 | Last attempt within `RETRY_BACKOFF_HOURS` (24h) | **WAIT** `within_backoff` |
| 9 | Razorpay still retrying, inside `BANK_RETRY_GRACE_HOURS` (72h) | **WAIT** `awaiting_bank_retry` |
| 10 | This *person* contacted within `CUSTOMER_CONTACT_COOLDOWN_HOURS` (24h) | **WAIT** `customer_recently_contacted` |
| 11 | Outside 09:00–21:00 in `CONTACT_TIMEZONE` | **WAIT** `outside_contact_window` |
| — | Otherwise | **act** — see below |

Three rules are worth singling out:

- **Rule 11 is the compliance rule.** TRAI restricts commercial calls to
  09:00–21:00, and the window is evaluated in the customer's local time rather
  than UTC. Getting that wrong means calling people at 3am.
- **Rule 2 lives on the customer, not the case.** The obligation follows the
  person: a second failed charge opens a second case, and a case-scoped
  suppression would happily dial a wrong number all over again.
- **Rules 8 and 10 are different on purpose.** Rule 8 bounds one *case*; rule 10
  bounds one *person*. A customer with three failed charges should hear from us
  once, not three times — one subscription produced four cases in two hours
  during testing.

### What "act" means — priority decides order, not channel

A common misreading of the tier table below is that tier 2 gets phoned second.
It does not. Three decisions are separate:

| Decision | Made by | Answers |
|---|---|---|
| **Order** | `priority_tier` + `score()` | Who is worked first in this batch |
| **Channel** | `_intervention_for()` → root cause | Link, call, or mandate re-charge |
| **Timing** | the WAIT gates above | Whether *now* is allowed at all |

Only `CUSTOMER_INSTRUMENT` — a dead or expired card, a revoked mandate — is in
`NEEDS_A_CONVERSATION`, so **only a dead instrument earns a phone call on the
first attempt.** A link cannot fix an expired card: it would recover today's
money and leave the subscription to break again next cycle, so somebody has to
talk to them about the instrument.

Everything else opens with a **payment link**, which is cheap and interrupts
nobody, and reaches the phone only on a second attempt — which cannot happen
inside `RETRY_BACKOFF_HOURS`. So a customer who was about to retry the payment
themselves is never rung mid-attempt.

| Tier | Name | First action | Why |
|---|---|---|---|
| 1 | `MANDATE_BROKEN` | **Call** | The only tier where *every future charge* fails too |
| 2 | `PAYMENT_ATTEMPTED` | Link | They were trying to pay. Warm intent, but they may retry themselves |
| 3 | `CHECKOUT_ABANDONED` | Link | *Reserved* — Razorpay sends no webhook for an abandoned checkout |
| 4 | `BACKGROUND` | Link | Bank-side, transient, or undiagnosed |

`score()` weighs the tier against the size of the debt and whether Razorpay has
stopped retrying, with age breaking ties so nothing starves. A large enough debt
can outrank a higher tier; a marginal one cannot.

Contact goes through the `ContactChannel` protocol, so the transport is a
deployment choice rather than a rewrite. Production runs `TwilioChannel`. If no
telephony is configured the loop degrades to `LoggingChannel`, which is named
`logging` in the audit trail precisely so a run of it can never be mistaken for
evidence that a customer was actually called.

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
uv sync --group pipecat --group voice
cp .env.example .env          # then fill in your Razorpay test keys
docker compose up -d db       # Postgres on localhost:5433
uv run alembic upgrade head
```

> The two groups are not optional for a full checkout. A bare `uv sync`
> installs the main dependencies and `dev` only, and the test suite imports
> `pipecat` and `mcp` (voice-agent tests) and `livekit` (the older dispatch
> path). Miss either and pytest aborts during collection, which reads as
> "everything is broken" rather than "two groups are missing". The deployed
> webhook image still installs neither — it has no use for a voice stack.

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
   `subscription.halted`, `subscription.charged`, `payment_link.paid`.

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

636 tests. The signature, policy and conversation-graph tests are pure unit tests;
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

## Running more than one worker

`FOR UPDATE SKIP LOCKED` is what lets the loop scale out, and the failure it
prevents is not a slow batch — it is a customer whose payment failed being rung
twice about the same debt by two machines a millisecond apart.

That is checked rather than asserted. `tests/test_concurrency.py` runs real
ticks concurrently against real Postgres, and reads the guarantee from both
sides: what the workers think they did, and what the audit trail says happened.
Comment out the `SKIP LOCKED` clause and three of those tests fail.

To watch it instead of reading it:

```bash
uv run python -m scripts.scale_demo --cases 200 --workers 1   # baseline
uv run python -m scripts.scale_demo --cases 600 --workers 6
uv run python -m scripts.scale_demo --clear
```

Real `run_once`, real policy, real priority ordering. Only the telephony is
faked, and every row it writes is stamped `source='seed'` — a column, not a
naming convention — so simulated work can never be reported as recovered money.

Measured on one laptop against local Postgres:

| Cases | Workers | Wall time | Contacted twice |
| ----- | ------- | --------- | --------------- |
| 200   | 1       | 1.07s     | 0               |
| 200   | 3       | 1.31s     | 0               |
| 600   | 6       | 4.20s     | 0               |

The even split is the interesting part, not the wall time: six workers took a
hundred cases each without coordinating with one another, because the database
did the coordinating.

Two pieces make N workers safe rather than merely fast, and both are optional —
everything degrades to the single-worker behaviour when they are absent:

- **Redis** (`REDIS_URL`) holds the contact cooldown. A Postgres row lock
  serialises one transaction, not a fleet, so without this two workers on two
  machines can both read "not contacted recently" and both dial. `SET NX EX` is
  atomic across all of them. It is a fast path and never the authority:
  `customers.last_contacted_at` remains the rule, and a miss falls through to it.
- **Pub/Sub** (`PUBSUB_TOPIC`) delivers webhook envelopes by push instead of
  waiting for the next sweep. Worth knowing what it does *not* do: the handlers
  it triggers only move case state, so pushing does not make a call happen
  sooner. The five-minute floor on "failure → phone rings" is the scheduler's,
  not the webhook path's.

## Running the whole voice stack on our own GPU

On a payments call the audio *is* customer data. "The recording never leaves
your infrastructure" is a procurement answer before it is an engineering one,
and RBI data-localisation rules make it a requirement rather than a talking
point. It is also insurance: a vendor quota ran out mid-call once already, and
the agent talked to nobody for two minutes while every log said the call was
fine.

So the same conversation runs on three self-hosted models, behind
`LLM_PROVIDER=local` and `LOCAL_SPEECH_URL`. Everything above them — the graph,
the walker, the policy, the outcome — is unchanged and does not know which pair
is running. Deployed with `modal deploy scripts/modal_llm.py` and
`scripts/modal_speech.py`; vLLM and the weights live in the Modal image, never
on a developer machine.

**The models were chosen against hard requirements, not benchmarks.**

| Role | Model | Why this one |
| ---- | ----- | ------------ |
| LLM | Gemma 4 12B (vLLM) | **Tool calling** is not negotiable — the whole state machine rides on the model calling `transition(label)`, and Gemma 4 is the first Gemma with tool calling in its own chat template with a matching vLLM parser. **Devanagari at 1.39 tokens per word**, the best of eleven tokenizers measured, and roughly a third of what Qwen spends on the same Hindi. |
| STT | SraVaani-1.0 (ARTPARK/IISc, MIT) | 430M parameters, 65 Indian languages, and — the reason it is here rather than Whisper — trained with **code-switching tagging**. Real customers on an Indian line speak Hindi and English in the same sentence, and our own transcripts are full of it. |
| TTS | VoxCPM2 | Generates **progressively**, so the customer waits for the first chunk rather than the last one. Native 48 kHz, resampled into the 8 kHz phone leg. |

**What was tested, and what was not.** Each model works in isolation: Devanagari
out of the LLM, tool calls parsed, and a full text→speech→text round trip
through the pair. Warm, it measured **TTS first byte 0.86s and STT 4.1s** on a
ten-second clip, against the cloud path's 0.512s per complete turn. **No phone
call has run end to end on it.**

The honest summary is that this is a *data-residency option with a real latency
cost*, not a faster alternative. STT is the floor: SraVaani transcribes a
complete utterance, so unlike the TTS side it cannot stream its way out of the
problem, and closing that gap means a different model or a faster serving stack
rather than tuning. A cold GPU container also takes about 264 seconds to start
— the client timeout was raised to 300s after 120s left the agent silent on a
live call.

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
- [x] **Phase 4** — Razorpay Payment Links for the `retry_now` intent, dual-key
      recovery attribution
- [x] **Phase 5** — Batch metrics: cases, ₹ at risk, ₹ recovered, recovery rate
      (`uv run python -m app.report`)

## Design notes

- [The live pipeline, end to end](docs/pipeline.md) — what actually runs, in
  order, with real component names and configured values. Anything unbuilt is
  marked as such rather than drawn as if it worked.

- [Architecture](docs/architecture.md) — the two planes in full: webhook
  intake and recovery on one side, the voice pipeline on the other, the two
  seams between them, and the VAD/STT settings with the reasoning behind each.

- [Fully local voice stack](docs/local-voice-architecture.md) — the proposed
  self-hosted STT/LLM/TTS design, why the entity-dense ASR gap matters more
  than headline WER, and what it would cost to close it. Proposed, not built.

## Known gaps

- **A customer who was mid-payment gets no grace period.** An abandoned
  checkout waits `checkout_grace_minutes` (30) before we chase it, on the
  reasoning that someone who retries with another card two minutes later has
  not abandoned anything. A *subscription* charge that failed while the
  customer was in the flow — an OTP screen, a bank redirect, `error_step` of
  `payment_authentication` or `payment_authorization` — has no equivalent wait,
  so it is picked up on the next five-minute tick. The blast radius is small
  because the first intervention for those causes is a payment link rather than
  a call, and a link is cheap and interrupts nobody. But it is inconsistent: a
  customer who is *still on the OTP screen* may well complete the payment in the
  next two minutes, and an SMS landing mid-attempt is noise at best and
  confusing at worst.

  The fix is a `payment_attempted_grace_minutes` gate alongside the checkout
  one. It has to key on the priority tier and not on `error_step`: an expired
  card also fails at `payment_authorization`, and that case is tier 1 and
  deliberately calls immediately — deferring it would be a regression, not a
  refinement.

- **`subscription.pending` with no open case — still open.** If Razorpay's
  `payment.failed` never reaches us (endpoint down past its retry window), the
  subscription goes pending with nothing to attach to. Still only logged as a
  warning: the reconciler that would backfill from
  `GET /invoices?subscription_id=` is not built. The replay sweep does not cover
  this, because it only retries events we actually received.
- **Three paths reach the same webhook envelope, and one of them is
  redundant.** Processing runs in-process via FastAPI `BackgroundTasks`
  (~1s), announced on Pub/Sub for push delivery (~1s), and swept by the worker
  for anything whose handler died. All three are safe because `process_event`
  returns early once `processed_at` is set, and events that fail permanently are
  dead-lettered rather than retried forever. But push and the background task do
  the same job at the same speed: the only thing push adds is surviving the API
  process being killed mid-handler, and even that is mostly covered because the
  sweep runs before the orchestrator in the same tick. The genuine hole it
  closes is about sixty seconds wide.
- **Calls that end with no intent recorded.** The graph refuses any label that
  is not a legal move from the current node, which is what stops a model
  inventing an outcome for someone's money — but a conversation that ends
  before reaching a terminal node records `unclear`, and the case simply stays
  open. Two live calls stalled this way: at `explain` the model kept talking
  instead of transitioning, and at `ask_intent` it sent the payment link, said
  so, and treated that as having finished the step. Both are fixed by naming
  the last step explicitly; neither is proven fixed across a run of calls.
- **The mandate link lands on a dead page.** `send_mandate_link` creates the
  subscription, fetches its `short_url` and texts it, and the SMS is delivered
  — but Razorpay returns *"Hosted page is not available"* for it. Payment-link
  pages render fine on the same key, so this is Subscriptions not being enabled
  on the account rather than anything in the code. Account setting, not a fix.
- **Two voice implementations.** `app/voice/agent.py` (LiveKit) predates
  `app/voice/pipecat_agent.py` (Twilio) and is not what production runs. It is
  kept for the browser-demo path but has not tracked recent fixes; read the
  Pipecat one.
- **A self-hosted model path exists and is unproven on a call.**
  `scripts/modal_llm.py` and `scripts/modal_speech.py` serve Gemma 4 12B,
  SraVaani-1.0 and VoxCPM2 on a rented GPU, behind `LOCAL_SPEECH_URL` and
  `llm_provider=local`. Each model works in isolation — Devanagari out,
  tool calls parsed, a full text→speech→text round trip — but no phone call has
  completed on that path. Warm, it measured TTS first byte 0.86s and STT 4.1s
  on a ten-second clip against the cloud path's 0.512s per turn. The honest
  summary is that it is a data-residency option with a real latency cost, not a
  faster alternative.
- **The app connects to Cloud SQL as the `postgres` superuser.** A dedicated
  least-privilege role is the right answer; it was skipped because setting up
  grants needs a `psql` session the deploy box did not have. The credential
  lives in Secret Manager, not in the repo, but this is a real shortcut.
- **No backups on the Cloud SQL instance** (`--no-backup`, zonal) to keep the
  hackathon cost at the floor. Do not copy that into anything real.
