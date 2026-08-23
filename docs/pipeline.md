# The live pipeline, end to end

What actually runs, in order, with the real component names and configured
values. Anything not yet wired is marked **[not built]** rather than drawn as if
it were.

---

## The whole path

```
 Razorpay                                                        Razorpay
 subscription                                                    Payment
 charge fails                                                    Link paid
     │                                                               │
     ▼                                                               ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  1  WEBHOOK INTAKE            Cloud Run service, asia-south1          │
 │     verify HMAC over raw body → INSERT envelope → 200 → background    │
 └─────────────────────────────────────────────────────────────────────┘
     │
     ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  2  CORRELATE + OPEN CASE                                            │
 │     payment.failed carries only invoice_id                           │
 │       → GET /invoices/{id} → subscription_id + customer_id           │
 │       → upsert customer/subscription/payment                         │
 │       → recovery_cases row (status=open)  + audit row                │
 └─────────────────────────────────────────────────────────────────────┘
     │
     ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  3  WORKER TICK               Cloud Run Job, every 5 min via Scheduler│
 │     replay sweep (stalled envelopes) → claim cases FOR UPDATE SKIP    │
 │     LOCKED, halted_at first → policy.decide() per case                │
 └─────────────────────────────────────────────────────────────────────┘
     │                    │                      │
   STOP                 WAIT                   CALL
  (closed,           (no row written,       (attempt++, status
   permanent)         re-evaluated)          in_progress)
                                               │
                                               ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  4  DISPATCH            LiveKitChannel → agent into room recovery-{id}│
 │     metadata: name, amount_spoken, failure_reason, language, phone    │
 └─────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  5  THE CALL                  agent worker (own process)              │
 │     warm Sarvam ∥ build context ∥ build session → start → [dial]      │
 └─────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  6  OUTCOME → 7 PAYMENT LINK → 8 RECOVERY (back to step 1)            │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Webhook intake

`POST /webhooks/razorpay` → [`webhooks/router.py`](../app/webhooks/router.py)

Subscribed events: `payment.failed`, `payment.captured`, `subscription.pending`,
`subscription.halted`, `subscription.charged`, `payment_link.paid`.

1. HMAC-SHA256 over the **raw body** (never re-serialised JSON), constant-time
   compare. Empty secret never validates.
2. Dedupe on `X-Razorpay-Event-Id`; a replay returns `{"status":"duplicate"}`.
3. Envelope persisted **synchronously**, then 200, then handler runs in the
   background. `processed_at IS NULL` is the replay queue.

## 2. Correlate and open a case

A `payment.failed` entity carries **no `customer_id` and no `subscription_id`** —
only `invoice_id`. So the handler fetches the invoice to resolve both before it
can open anything. Without a subscription link it is a one-off checkout failure:
recorded, no case.

Tables touched: `webhook_events`, `customers`, `subscriptions`, `payments`,
`recovery_cases`, `recovery_actions`.

## 3. Worker tick

Cloud Run Job on a 5-minute Cloud Scheduler trigger (`python -m app.worker --once`).
A service could not host this — Cloud Run scales to zero and would kill a loop.

Each tick: replay sweep, then claim a batch with `FOR UPDATE SKIP LOCKED`
(so a second worker takes the next batch rather than blocking), `halted_at`
first — Razorpay has already given up on those.

Then [`policy.decide()`](../app/policy.py), pure, no I/O:

| # | Rule | Result |
|---|---|---|
| 1 | already recovered / declined / stopped | **STOP** |
| 2 | `attempt_count >= max_attempts` (3) | **STOP** |
| 3 | amount < ₹50 | **STOP** — a call costs more than the debt |
| 4 | no phone number | **STOP** |
| 5 | within 24h of last attempt | **WAIT** |
| 6 | outside 09:00–21:00 `Asia/Kolkata` | **WAIT** — TRAI |
| — | otherwise | **CALL** |

All STOP conditions are evaluated before any WAIT, so a dead case is closed
rather than parked forever outside the calling window. STOP is permanent.

## 4. Dispatch

[`LiveKitChannel`](../app/voice/dispatch.py) creates the agent dispatch **before**
the phone rings, so there is never a moment where someone has answered and
nothing is listening. Job metadata carries the case context — including
`amount_spoken`, already rendered ("5 लाख रुपये", not "500000"), because a raw
numeral plus a Latin "Rs" is the script flip that makes Hindi voices stumble.

Refuses to construct without `LIVEKIT_SIP_TRUNK_ID` rather than dispatching into
a room nobody will join.

## 5. The call

[`agent.py`](../app/voice/agent.py), a separate worker process.

```
asyncio.create_task(warm_sarvam())      # 15-25s cold starts drop the first hello
        ↓ (overlapped)
build call context → GraphWalker(DUNNING_FLOW, context)
        ↓                                 raises here if context is incomplete,
build_session()                           rather than reading "{customer_name}"
        ↓                                 down a live phone line
await await_warmup(...)                 # bounded; proceeds cold rather than hang
        ↓
session.start(room, NodeAgent(state))
        ↓
create_sip_participant(...)             # only if metadata carries a phone
```

### The audio loop

```
customer speaks
   │
   ▼  LiveKit WebRTC / SIP
Silero VAD            activation_threshold 0.7 · min_silence 0.3s
   │                  (0.7 may never fire on narrowband SIP — see lessons doc)
   ▼
Sarvam saaras:v4      wss, 16 kHz, mode=transcribe, language=unknown
   │                  vad_signals + flush_signal (both, per Sarvam support)
   ▼  turn_detection="stt"  — Sarvam's END_SPEECH drives turns, not a timer
Gemini 2.5 Flash      Vertex AI, asia-south1, temperature 0.2, streaming
   │
   ├── text ──▶ Cartesia (Arushi, Hinglish voice) ──▶ LiveKit ──▶ customer
   │
   └── transition(label) tool call
              │
              ▼
        GraphWalker.transition()
          · validates the label against THIS node's edges
          · records an Observation (node, utterance, label, accepted)
          · returns a new NodeAgent = handoff to the next stage
```

The model never names a destination node — only an **edge label**, validated
against the current node. An unknown label, or one valid elsewhere but not here,
is rejected and the error text is handed back so it can self-correct.

### The graph

```
greet ──identity_confirmed──▶ explain ──acknowledged──▶ reason_inquiry
  │                             │                          │
  │                             │                    reason_given
  └─not_the_customer─▶ WRONG    └─disputes─▶ DISPUTE        ▼
                                                      ask_intent
                                     ┌──────────┬──────────┼──────────┐
                                  pay_now   pay_later   declined   disputes
                                     ▼          ▲▲          ▼          ▼
                                 PAY_NOW        ││      DECLINED    DISPUTE
                                                ││
                                financial_difficulty (from either stage)
```

`greet` **cannot** reference the amount — enforced by a test, so whoever answers
never hears a stranger's billing details. Every non-terminal node carries the
language-mirroring rule; terminals carry it too, so the sign-off does not switch
language at the last turn.

## 6. Outcome

[`outcomes.py`](../app/voice/outcomes.py) — the seam between the conversation and
the money. **Intent decides the case; the transcript is evidence, never input.**
No intent maps to `RECOVERED`; only a real `payment.captured` produces that.

| Intent | Case | Contact |
|---|---|---|
| `retry_now` | unchanged | → payment link |
| `retry_later` | unchanged | backoff handles it |
| `declined` | DECLINED | suppressed |
| `wrong_number` | STOPPED | suppressed |
| `dispute` | STOPPED | suppressed, flagged for a human |
| `no_answer` / `unclear` | unchanged | attempt cap bounds it |

Suppression burns the remaining attempts, so no later tick can revive the case.

## 7. Payment link

`reference_id = recovery-{case.id}` **and** `notes.recovery_case_id` — dual key,
because Razorpay's own docs show `notes` arriving as an empty list. A second
`retry_now` reuses the existing link rather than minting a second one for the
same debt.

## 8. Recovery closes the loop

`payment_link.paid` (or `subscription.charged` / `payment.captured`) re-enters at
step 1, credits the case by `reference_id`, and `recovery_cases.status` becomes
`recovered`. `python -m app.report` then reads the batch.

---

## What is not in this picture

- **[not built]** PSTN dialling — needs an outbound SIP trunk. `_dial()` is
  written; there is no trunk.
- **[not built]** Carrier IVR detection. Indian carriers answer the leg for
  voicemail, busy and DND alike; without this the agent converses with a
  recording and stores an intent.
- **[not built]** Script-detection language post-processor. Mirroring is prompt-only
  today, and prompt-only language control is documented as unreliable.
- **[not built]** `subscription.pending` reconciler — if `payment.failed` never
  arrives, that subscription has no case to attach to.
