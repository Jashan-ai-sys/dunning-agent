# Architecture

Two independent planes over one schema.

**Plane A — webhook intake and recovery** decides *whether and how* to
intervene. **Plane B — the voice pipeline** conducts the conversation. They
share a database and two seams, and nothing else: no process, no container, no
import from B into A.

That separation is why a webhook cold start is ~6s and not ~264s. It is also
why `dunning-voice` must be built from `Dockerfile.voice` and never from the
default `Dockerfile` — see [Deployment](#deployment).

---

## Plane A — webhook intake and recovery

### Ingress: one INSERT, then 200

`POST /webhooks/razorpay` (`app/webhooks/router.py`) does the least work that is
still safe, because Razorpay times out:

1. **Verify the HMAC** over the *raw* body against `RAZORPAY_WEBHOOK_SECRET`.
   A bad signature returns 401 with no detail — an attacker probing the
   endpoint learns nothing about why it failed.
2. **Parse and require `event`.** Malformed body → 400.
3. **Derive an event id** from `X-Razorpay-Event-Id`, falling back to
   `sha256(raw_body)` so a missing header degrades to weaker dedupe rather than
   dropping the event.
4. **Persist the envelope** via `record_event()`. Redis is asked first — it can
   only ever say "definitely seen" — and a miss falls through to the unique
   constraint, which is what actually guarantees uniqueness. A duplicate
   returns `{"status": "duplicate"}`.

Nothing is interpreted in the request. The envelope is durable, and
`processed_at IS NULL` is what puts it on the replay queue.

### Three dispatch paths, one idempotent handler

```
                    record_event()  ──►  envelope durable, processed_at IS NULL
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  BackgroundTasks      Pub/Sub push        cron sweep
  in-process, ~1s      ~1s, survives       ≤ 5 min,
                       the API being       oldest first
                       killed mid-handler
        └───────────────────┼───────────────────┘
                            ▼
                     process_event()
              returns early once processed_at is set
```

**The push endpoint authenticates with OIDC, not a shared secret.** Pub/Sub push
cannot send custom headers, so the alternatives were a token in the URL — which
lands in every access log — or verifying the signed token Google attaches. This
endpoint dispatches handlers that create payment links and place phone calls, so
it gets the real one (`app/webhooks/pubsub_router.py`):

- the bearer token is verified with `id_token.verify_oauth2_token` against the
  subscription's configured audience;
- the claims must carry the expected service-account `email` *and*
  `email_verified`, or the request is 403'd;
- if `PUBSUB_PUSH_SERVICE_ACCOUNT` is unset the endpoint answers **503 rather
  than running unauthenticated** — a push endpoint that dispatches handlers is
  not something to leave open because a setting was forgotten;
- anything that must not be redelivered gets a 204, including an unparseable
  message, because Pub/Sub retries until acknowledged.

All three are safe because `process_event` is idempotent. Push and the
background task do the same job at the same speed; the genuine hole push closes
is about sixty seconds wide — the window where the API process is killed
mid-handler before the next sweep.

**Failing forward, not forever.** A queue that only retries does not become
resilient, it jams. The sweep reads oldest-unprocessed first, so an envelope
that fails deterministically sits at the front of every sweep — and once there
are more of those than the sweep's limit, nothing newer is ever replayed. So an
envelope is either retried or dead-lettered after `webhook_max_attempts` (5),
decided by whether it could plausibly succeed next time.

### Handlers

| Event | What it does |
| --- | --- |
| `payment.failed` | Opens a recovery case — **only** when it carries an `invoice_id`, i.e. a subscription charge. A failed payment *link* has no invoice and correctly opens nothing: that is a checkout failure, not subscription recovery. |
| `payment.captured` | Money arrived. Closes the case as recovered. |
| `subscription.pending` | Razorpay is retrying. Defers by `bank_retry_grace_hours` (72). |
| `subscription.halted` | Razorpay gave up. The strongest signal to work the case ourselves. |
| `subscription.charged` | A later cycle succeeded — closes open cases for that subscription. |
| `payment_link.paid` | Attributes payment through `reference_id` (`recovery-{case}-{attempt}`). |
| `order.paid` | Checkout recovery, attributed through `order_id`. |

### Diagnosis and priority — two pure functions

Both read columns and return a value. No I/O, no model, no clock. That is what
makes every rule a plain unit test.

`app/diagnosis.py` maps `error_source` + `error_reason` to a **root cause**:

| Root cause | Meaning |
| --- | --- |
| `customer_funds` | Instrument works, no money behind it. |
| `customer_instrument` | Card dead, expired, blocked, mandate revoked. Retrying the same charge cannot work. |
| `customer_action` | OTP not entered, 3DS abandoned. |
| `bank_decline` | Issuer refused. Often temporary, not ours to fix. |
| `transient` | Razorpay or gateway broke. Ours to retry, never the customer's to hear about. |
| `configuration` | Our own setup is wrong. A human has to look. |
| `unknown` | No usable failure fields. |

`app/priority.py` turns that into a **tier**, stored as the `priority_tier`
column when the case is opened. Stored rather than computed in SQL, because
duplicating the mapping as a `CASE` would give the queue and the report two
different opinions about the same failure. It cannot go stale: the columns it
reads are written once and never touched again.

| Tier | Name | Why here |
| --- | --- | --- |
| 1 | `MANDATE_BROKEN` | The only tier where *every future charge* fails too — and no payment link fixes that. |
| 2 | `PAYMENT_ATTEMPTED` | They were sitting there trying to pay. Warm, recent intent. |
| 3 | `CHECKOUT_ABANDONED` | *Reserved.* Razorpay sends no webhook for an abandoned checkout, so nothing produces this tier yet. |
| 4 | `BACKGROUND` | Bank-side, transient or undiagnosed. |

`score()` weighs the tier against the size of the debt and whether Razorpay has
stopped retrying, with age breaking ties so nothing starves. A large enough
debt can outrank a higher tier; a marginal one cannot.

### The worker tick

Cloud Scheduler fires `*/5 * * * *` at the Cloud Run **job** `dunning-worker` —
not at the API service. One tick is `orchestrator.run_once()`:

```sql
SELECT ... FROM recovery_cases
WHERE status IN ('open','in_progress')
  AND (next_eligible_at IS NULL OR next_eligible_at <= now)
ORDER BY score() DESC, created_at
LIMIT worker_batch_size
FOR UPDATE SKIP LOCKED
```

`SKIP LOCKED` is what lets the loop scale out, and the failure it prevents is
not a slow batch — it is a customer being rung twice about the same debt by two
machines a millisecond apart. The `next_eligible_at` filter matters more than
it looks: parked cases are still open and still among the oldest rows, so
without it they sort to the front of every batch and nothing newer is ever
claimed again.

Then, per case, a read-only Redis check (`cooldown:{customer_id}`) catches the
window the row lock cannot — another worker that contacted this person in a
transaction not yet in our snapshot. It only ever *adds* a wait, never removes
one. Postgres remains the authority.

### The policy ladder

`decide(case, customer, now, settings)` is a pure function returning one
`Decision`. First match wins:

| # | Outcome | Condition |
| --- | --- | --- |
| 1 | STOP | `already_closed` |
| 2 | STOP | `do_not_contact` — held on the **customer**, because the obligation follows the person, not the debt |
| 3 | STOP | `max_attempts_reached` |
| 4 | STOP | `undeliverable` — `max_delivery_failures` consecutive send failures |
| 5 | STOP | `below_min_amount` — under ₹50 a voice call costs more than the debt |
| 6 | STOP | `needs_human` — root cause is `configuration`. Counted separately so our own bug is not buried among exhausted cases |
| 7 | STOP | `no_contact_details` |
| 8 | WAIT | case backoff — `retry_backoff_hours` (24) |
| 9 | WAIT | bank retry grace — `bank_retry_grace_hours` (72) |
| 10 | WAIT | `customer_recently_contacted` — `customer_contact_cooldown_hours` (24), per **person** |
| 11 | WAIT | `outside_contact_window` — 09:00–21:00 Asia/Kolkata, the TRAI compliance story |
| 12 | act | `LINK` on the first attempt, `CALL` thereafter; `RETRY_MANDATE` when enabled and the only problem was absent funds |

**Priority decides order, not channel or timing.** Three separate decisions,
and conflating them is what makes the tiers look wrong:

| Decision | Made by | Answers |
| --- | --- | --- |
| Order | `priority_tier` + `score()` | Who gets worked first in this batch |
| Channel | `_intervention_for()` → root cause | Link, call, or mandate re-charge |
| Timing | the WAIT gates above | Whether *now* is allowed at all |

Only `CUSTOMER_INSTRUMENT` is in `NEEDS_A_CONVERSATION`, so only a dead
instrument earns a phone call on the *first* attempt — a link cannot fix an
expired card, and would recover today's money while leaving the subscription to
break again next cycle. Everything else opens with a link and reaches the phone
only on a second attempt, which cannot happen inside `retry_backoff_hours`. So a
customer who was about to retry the payment themselves is never rung mid-attempt.

**Known gap:** there is no grace period for a subscription charge that failed
while the customer was in the flow, though an abandoned checkout gets
`checkout_grace_minutes` (30). See `Known gaps` in the README — and note that a
fix must key on the tier, not `error_step`, because an expired card also fails
at `payment_authorization` and is deliberately called immediately.

**Two cooldowns, deliberately different.** `retry_backoff_hours` bounds one
*case*. `customer_contact_cooldown_hours` bounds one *person*. A customer with
three failed charges should still hear from us once, not three times — one
subscription produced four cases in two hours during testing.

### Schema

| Table | Holds |
| --- | --- |
| `webhook_events` | The verified envelope. `processed_at` drives replay; unique event id is the dedupe authority. |
| `customers` | Identity, phone, language, `do_not_contact`, `last_contacted_at`. |
| `subscriptions` | Razorpay subscription mirror. |
| `payments` | Payment mirror. |
| `recovery_cases` | The unit metrics are computed over. Carries `source` — `'seed'` vs `'razorpay'`. |
| `recovery_actions` | The audit trail: every policy decision and every action taken. |
| `voice_calls` | One row per call, with transitions and transcript. *The seam to Plane B.* |

`source` is a column, not a naming convention. Seeded demo rows can never be
reported as recovered money — which, for a product judged on recovered rupees,
is the claim that matters most.

---

## Plane B — the voice pipeline

Its own image, its own Cloud Run service, its own scaling. Carries onnxruntime,
Silero and the smart-turn model, and holds a CPU for the length of a call —
none of which a webhook should ever pay for.

### How a call starts

`TwilioChannel.initiate()` POSTs to Twilio's `Calls.json`. The TwiML it returns
points Twilio Media Streams at `TWILIO_STREAM_URL` — `wss://dunning-voice…/ws`.
Twilio dials from its own network, so that URL must be public and `wss://`;
localhost is invisible to it.

The container runs `python -m app.voice.pipecat_agent -t twilio`, which hands
off to `pipecat.runner.run.main()`. The runner owns the `/ws` route and
constructs the transport; `bot()` is called per connection.

### The pipeline

```
Twilio ──► transport.input() ──► Sarvam STT ──► user aggregator ──► Gemini 2.5 Flash
 μ-law                             0.483s         Silero VAD          Vertex, TTFT 0.313s
 8 kHz                                            SmartTurn v3                │
                                                                              ▼
                                                                   [GuardrailProcessor]
                                                                    off by default, 3.3ms
                                                                              │
 caller ◄── transport.output() ◄── Cartesia TTS ◄─────────────────────────────┘
                    │                TTFB 0.058s
                    ▼
            assistant aggregator ──► back into context
            records what was actually SPOKEN
```

That last arrow matters. The assistant aggregator records what was *spoken*, so
anything the pipeline emits re-enters the context as an assistant turn. That is
what made the parroting bug self-reinforcing: one echoed line taught the model
the pattern for the next turn.

### Turn-taking

| Component | Setting | Note |
| --- | --- | --- |
| Silero `confidence` | `VAD_ACTIVATION_THRESHOLD` | 0.7 suits the wideband browser path; a narrowband phone leg carries 0.05–0.86% of its energy above 4 kHz, so 0.7 never fired on the customer channel. Tuned per deployment. |
| Silero `stop_secs` | 0.2 hardcoded | **Not** `vad_min_silence_duration`. In Pipecat 1.x this is a low-level detection threshold, not the wait before replying — that lives in the stop strategy. |
| Silero `start_secs` | 0.2 hardcoded | |
| Silero `min_volume` | 0.6 hardcoded | |
| Start strategy | `MinWordsUserTurnStartStrategy(min_words=3)` | *Replaces* the VAD start strategy — start strategies race, and a VAD start would fire on the first syllable. Three words because Hindi speakers backchannel constantly ("हाँ", "जी", "अच्छा") and two-word acknowledgements were still taking the floor. |
| `use_interim` | `False` | No partials arrive (`mode="transcribe"` with `vad_signals=False`), so mid-utterance barge-in is Silero's `start_secs` alone. |
| Stop strategy | `TurnAnalyzerUserTurnStopStrategy`, SmartTurn v3 ONNX, `stop_secs=3.0` | Judges completeness from audio rather than the clock. A customer explaining why they could not pay does not speak in clean sentences. |

`vad_min_silence_duration` and `vad_min_speech_duration` in `config.py` are
**not read by the Pipecat path** — they are 0.0.108 values that mean something
different there, and the LiveKit path still uses them.

### Sarvam STT

| Setting | Value |
| --- | --- |
| model | `SARVAM_STT_MODEL`, resolved through `resolve_sarvam_model()` — Pipecat 1.7.0 rejects `saaras:v4` and silently runs `v3`; 1.8.1 accepts v4 |
| language | `hi-IN` **pinned**, not auto-detect — per-utterance detection lands on English for a short Hinglish reply and returns romanised text, which the model then mirrors |
| sample_rate | 16 kHz — 8 kHz starves the model and garbles Hindi |
| mode | `transcribe` — the model sees the customer's own language, not an English translation, which is what makes mirroring possible |
| `vad_signals` | `False` — `True` suppresses `flush_signal`, so the turn waits on the p99 fallback timer instead of the transcript that was ready |
| `high_vad_sensitivity` | `True` — tunes Sarvam's own segmenter without handing it the turn |
| keepalive | 30 s timeout / 5 s interval — Sarvam kills idle sockets after ~60 s, and a customer thinking in silence is enough to hit it |

### The conversation is a graph, not a prompt

Nine nodes in `app/voice/flow.py` — four where the agent works, five terminals:

| Node | Kind | Job |
| --- | --- | --- |
| `greet` | START | Confirm identity. Nothing about the payment is said before this — it is someone's billing information. |
| `explain` | AGENT | State the amount and the reason, matter-of-fact. Most failures are bank-side. |
| `reason_inquiry` | AGENT | One short question: do they know why it failed? |
| `ask_intent` | AGENT | The decision point. Three options, no discount — the agent has no authority to change the amount. |
| `pay_now` | END | → `retry_now` |
| `pay_later` | END | → `retry_later` |
| `declined` | END | → `declined` |
| `wrong_number` | END | → `wrong_number` |
| `dispute` | END | → `dispute`, handed to a human, never argued |

The model never picks a node. It picks an **edge label**, validated against the
current node; an unknown or out-of-context label raises `InvalidTransition` and
is *refused* rather than followed. A conversation that ends before a terminal
records `unclear` and the case stays open — the honest outcome, not a bug.

Every attempt is stored as an `Observation(node_id, label, accepted, utterance,
rejection)`. Rejections are kept deliberately: a label the model reached for and
could not have is a harder negative than anything you would write by hand, and
it is already the exact shape a DSPy signature needs.

### MCP — what the agent may do

The money tools live behind an MCP server (`app/mcp_server.py`) spawned per
call over stdio, with the case injected as `DUNNING_CASE_ID`. All three tools
take **no arguments**:

| Tool | Effect |
| --- | --- |
| `send_payment_link` | Sends the Razorpay link by SMS and email. Asking twice returns the existing link rather than minting a second one for the same debt. |
| `send_mandate_link` | Re-authorises a dead mandate. |
| `get_case` | Read-only lookup: amount, failure reason, who it is for. |

**Why zero arguments.** An earlier version took `recovery_case_id` as a tool
argument and the model did what you would expect of a number it was never told
— it invented `12345` and the link silently failed to send. The hallucination
is the mild failure. The dangerous one is a *plausible* id: there is no reason a
model guessing at integers would miss by much, and the tool would then have sent
a stranger's payment link to whoever is on the call. Binding the id outside the
model's reach means the agent cannot name a case at all.

The agent can send a link and read a case. It **cannot** mark anything
recovered, change an amount, or reopen a closed case. Recovery is something only
a real Razorpay webhook may assert.

The ordering is enforced by the stage prompt and visible in the logs: call
`send_payment_link` **first**, wait for `sent: true`, only then say the link was
sent, and only then `transition(pay_now)` — because sending a link is not the
same as recording what the customer decided.

### Writing the outcome back

`app/voice/persistence.py` is the only place the agent process touches Postgres,
which keeps `pipecat_agent.py` about audio. Two rules are load-bearing:

- **A database failure must never break a live call.** Every entry point
  swallows its exceptions. Losing the record of a call is bad; dropping the
  customer mid-sentence because a connection pool hiccuped is worse.
- **A demo run must not write anything.** With no `recovery_case_id` in the job
  metadata, every function is a no-op. Nothing invents a case.

`open_call_record()` writes the `voice_calls` row at the *start*, so an
abandoned or crashed call still leaves a trace — a call that vanished is itself
a finding. `finalise()` is idempotent and applies one of seven intents:
`retry_now`, `retry_later`, `declined`, `wrong_number`, `dispute`, `no_answer`,
`unclear`.

---

## Where the two planes touch

Exactly two seams, and both are data:

1. **Out:** the worker passes `recovery_case_id` into the call metadata when it
   dials. That is the whole handoff.
2. **Back:** `apply_call_result` writes the detected intent onto the same case —
   the identical path a simulated batch uses, so nothing about a live call is
   special-cased.

There is no shared process, no shared container, no import from Plane B into
Plane A.

---

## Deployment

| Component | Kind | Plane |
| --- | --- | --- |
| `dunning-agent` | Cloud Run service | A — webhook API, scales to zero |
| `dunning-voice` | Cloud Run service | B — Media Streams websocket at `/ws` |
| `dunning-worker` | Cloud Run job, `*/5 * * * *` | A — one tick per invocation |
| `dunning-migrate` | Cloud Run job | A — alembic, same image as the API |
| `dunning-recover` | Cloud Run job | A |
| `dunning-report` | Cloud Run job | A — batch metrics |
| `dunning-seed-demo` | Cloud Run job | A — writes `source='seed'` only |
| `dunning-db` | Cloud SQL, POSTGRES_16 | shared |
| `webhook-events` | Pub/Sub topic + push subscription | A |

All secrets come from Secret Manager — `database-url`, `razorpay-key-secret`,
`twilio-auth-token`, `sarvam-api-key`, `cartesia-api-key` — never plaintext env.

### ⚠️ Two Dockerfiles

`Dockerfile` builds the **webhook API** (`uvicorn app.main:app`).
`Dockerfile.voice` builds the **voice agent**
(`python -m app.voice.pipecat_agent -t twilio`).

`gcloud run deploy --source .` builds the **default** one. Running the obvious
command against `dunning-voice` therefore deploys the webhook API to it, which
has no `/ws` route — and Starlette closing an unmatched websocket is reported by
uvicorn as `403 Forbidden`, which sends you hunting for an auth problem that
does not exist. Symptom: `/health` starts answering `200` on `dunning-voice`,
where the voice runner has no such route and correctly 404s.

Build the voice image explicitly:

```bash
gcloud builds submit --config cloudbuild.voice.yaml .   # docker build -f Dockerfile.voice
gcloud run services update dunning-voice --image <that tag> --region asia-south1
```

### Where each setting has to live

`COMPANY_NAME` is read by the **worker**, not the voice service:
`TwilioChannel._body()` calls `call_body(..., company_name=settings.company_name)`
and bakes it into the call metadata. Setting it on `dunning-voice` has no
effect on what the agent says.

Voice-side settings (`VAD_ACTIVATION_THRESHOLD`, `SARVAM_*`, `CARTESIA_*`,
`GUARDRAILS_MODE`, `LLM_PROVIDER`) belong on `dunning-voice`.
