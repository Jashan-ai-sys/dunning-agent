# Lessons from the Blostem voice backend

Read end to end from `C:\Users\WIN11\Downloads\backend` — a production Indian
telephony voice-agent backend (Pipecat + Plivo/Exotel/Smartflo). It has already
paid for the mistakes we are about to make. This records what applies to us,
what it validates, and what it contradicts.

Their stack is Pipecat; ours is LiveKit Agents. The provider behaviour beneath
both is identical, so provider-level findings transfer directly. Pipeline-level
processors have to be re-implemented against LiveKit's hooks.

---

## 1. What it validates (we already match)

| Choice | Theirs | Ours |
|---|---|---|
| STT | Sarvam `saaras:v4` / `v3` | `saaras:v4` |
| STT mode | `transcribe` | `transcribe` |
| STT language | `unknown` (auto-detect) | `unknown` |
| LLM | `gemini-2.5-flash` | `gemini-2.5-flash` |
| TTS | Cartesia `sonic-3`, VoxCPM2 | Cartesia |
| Sample rate | **16 kHz** | 16 kHz |

**`vad_signals` + `flush_signal` — both, together.** They patched Pipecat
specifically to force `flush_signal=true` *even when* server VAD is on, because
Pipecat only sent it when `vad_signals` was false. Their note: *"Sarvam support
recommends using both for 8 kHz telephony: server VAD emits speech-boundary
events, while flush lets explicit end-of-turn flushing avoid a residual wait."*

The LiveKit plugin already sends both — our connect URL is
`?...&vad_signals=true&...&flush_signal=true`. Nothing to change, and now we
know why it matters: without server VAD events, the framework falls back to a
silence heuristic, which drove **their STT p95 to 1.3 s**.

## 2. The single biggest lesson: prompts drift, normalizers do not

They have **three** deterministic text normalizers applied to the *spoken* text
just before synthesis. The LLM context, transcript and RAG memory keep the
original — only the bytes handed to the TTS socket are rewritten.

- **`tts_amount_normalizer`** — `₹500000` → `5 लाख`. Their rationale is blunt:
  *"The system prompt already asks the LLM to speak amounts in words, but LLMs
  drift and still emit raw digits. This module is the deterministic safety
  net."*
- **`tts_time_normalizer`** — `11 Aug, 7:30 PM` → `शाम साढ़े सात बजे`. Their
  note: *"A prompt rule for this was available and did not fix it."*
- **`tts_hindi_normalizer`** — transliterates Latin tokens to Devanagari,
  because Sarvam `bulbul:v3` inserts hesitations and *drops sentence tails* when
  the script flips mid-sentence. Hinglish flips script constantly.

**Why this hits us directly.** Our `pay_now` node instructs the model to say
"it is for Rs {amount_rupees}". The model will emit `Rs 499` or `499`, and a
Hindi-configured voice will read digits with Hindi digit treatment and the Latin
"Rs" as a guess. We have exactly the failure they wrote three modules to stop.

The conservatism in their amount normalizer is the part worth copying: only
rewrite a number carrying an explicit currency cue (`₹`/`Rs`/`INR`/`रुपये`),
skip anything with a decimal fraction, and only collapse clean magnitudes. That
single rule is what keeps OTPs, phone numbers, account numbers, reference IDs
and interest rates untouched.

## 3. Prompt-based language control is not reliable

`language_auto_switcher.py` exists because *"Gemini Vertex is NOT deterministic
about this: it often answers conversationally instead"* — acknowledging a
language switch in the old language, then emitting the new language without
ever calling the `switch_language` tool. Observed result on their call
63a9661d: Hindi text rendered with Tamil prosody.

Their fix watches text frames leaving the LLM, classifies the dominant Unicode
script, and mutates the TTS language setting to match.

**We just added a prompt-only mirroring rule.** Their evidence says a prompt
alone will not hold. If mirroring matters, it needs a script-detection
post-processor on our side too.

## 4. Their debt-recovery template vs our graph

`app/data/templates/collections_templates.py` → `DEBT_RECOVERY`. Independent
convergence on rules we derived separately:

- *"Confirm identity (mandatory before discussing dues)"* — our identity-before-amount rule
- *"Never call before 9 AM or after 9 PM IST"* — our contact window
- *"Be firm but respectful — NEVER threaten"*, RBI Fair Practices Code
- *"Keep responses under 30 words"*

Worth stealing:

- **Node they have that we lack: `reason_inquiry`** — ask *why* the payment
  failed before offering options. Cheap to add, and it makes the call feel less
  like a script.
- **Objection handlers** keyed by trigger: `already_paid`, `dispute_amount`,
  `financial_difficulty`, `harassment`. We cover `dispute` and `declined`; the
  other two are real and we would fall through to the wrong branch.
- **LLM params**: `temperature=0.2`, `max_tokens=120`. We set neither.
- **Prioritisation scoring**: weighted buckets over `days_past_due` (40) and
  `overdue_amount` (30). Our orchestrator orders by `halted_at` then
  `created_at` — a weighted score is strictly better for choosing who to call
  first in a batch.

## 5. Failure modes we have not thought about

- **`non_human_detector`** — in Indian telephony, carriers (Jio/Airtel/VI/BSNL)
  answer the RTP leg for *everything*: real pickup, busy IVR, switched-off
  recording, voicemail, DND rejection. All look identical to SIP. Carrier AMD
  does not help because the IVR is a human-recorded voice saying natural
  sentences. *The only reliable signal is what is being said.* Without this our
  agent holds a full conversation with a carrier IVR and records an intent.
- **`stt_drop_clarifier`** — VAD says the customer spoke, STT returns nothing
  (common on 8 kHz Hindi). Bot stays silent, customer says "hello", the bot
  re-asks its previous question. Their fix: if ≥800 ms of speech closes with no
  transcript within 1.5 s, inject a synthetic "audio unclear, ask them to
  repeat" marker; escalate after 2 consecutive drops.
- **`user_turn_coalescer`** — streaming STT emits multiple finals for one
  thought on telephony. Without coalescing, the LLM answers `"मुझे"` before the
  customer finishes `"मुझे दस लाख का EMI बताइए"`.
- **`barge_in`** — stock end-of-turn barge-in measured **2022 ms** from speech
  onset to bot audio stopping. Their warning is the useful half: the naive fix
  (evaluate on every transcript) was *tried twice and reverted twice*, because
  word count cannot separate `"अच्छा रुको"` (genuine interrupt) from `"हाँ जी"`
  (back-channel meaning *go on*).
- **`greeting_prerender`** — cold backend gave **5.5 s of dead air** after
  pickup. They pre-render greeting PCM to Redis in parallel with the dial-out.

## 6. If we self-host: VoxCPM2 notes

From `voxcpm_tts.py`, running on their own L40S:

1. **Pin the voice clone.** VoxCPM2 is zero-shot and stochastic — call it twice
   with no reference and timbre drifts between utterances, so the agent sounds
   like a different person mid-call. Encode a reference clip *once* into prompt
   latents via `/encode_latents` and send those same latents every request; the
   engine prefix-caches them, making the clone free per call.
2. **Do not send `temperature` on the cloned path** — it produces runaway,
   garbled audio. The server default is the stable one.
3. **Gain up ~8×.** VoxCPM2 emits ≈ −24 dBFS, near-inaudible on a phone line.
4. **Request raw PCM, not MP3** — no GPU encode, no local decode.

Measured: TTFB p50 **0.2–0.5 s** from 1 to 64 concurrent streams; ~48
concurrent *speaking* streams stay faster than realtime, so one GPU carries
roughly **100 concurrent calls** at a 30–45% talk ratio.

## 7. For our PSTN path specifically

- **Sarvam wants 16 kHz. Telephony is 8 kHz.** Feeding native 8 kHz *"starved
  the model and garbled Hindi"*. They upsample (`stt_audio_upsampler.py`) and
  are careful that the rate the service is *told* matches what it is *fed* —
  otherwise it transcribes pitch-shifted gibberish. Our browser path is already
  wideband, so this only bites when we add SIP.
- **Sarvam kills idle WebSockets after ~60 s.** They set `keepalive_timeout=30`,
  `keepalive_interval=5`. Our calls are short, but a customer thinking in
  silence could hit it.

## Adoption order

1. Amount + time normalizers on the spoken text (highest value, no new deps).
2. `temperature=0.2`, `max_tokens=120` on the LLM.
3. Weighted prioritisation score in the orchestrator.
4. `reason_inquiry` node and the missing objection branches.
5. Script-detection language post-processor.
6. Everything in §5, only if we go to real telephony.
