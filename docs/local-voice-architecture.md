# Design: fully local voice stack

**Status:** proposed, not built. The shipped system uses hosted STT/LLM/TTS.
**Author's note:** every number in here is either measured on this project or
cited. Where something is a guess, it says so.

---

## Why

Three arguments, in descending order of how much they should count.

1. **Data locality.** A dunning call carries a customer's name, phone number,
   the fact that they missed a payment, and what they said about it. Under the
   current design that audio and text transit Sarvam, Google and Cartesia.
   Running inference inside our own VPC in `asia-south1` means the only third
   party that sees anything is the telephony carrier, and all it sees is a
   phone number — no amount, no failure reason, no transcript.
2. **Cost at volume.** Per-minute STT and TTS pricing is fine for a demo and
   bad for a recovery product whose whole premise is that each call is worth a
   fraction of the debt. Owned GPUs invert that curve.
3. **Control over the failure mode that actually matters.** See
   [the entity problem](#the-entity-problem-is-the-real-one) — we cannot
   fine-tune a hosted API on Indian financial speech, and that is precisely
   where hosted models are weakest.

Sovereignty also happens to align with where Razorpay is going, but that is a
pitch argument, not an engineering one, and it should not be what decides this.

## What changes, and what does not

Almost nothing above the provider layer changes. The conversation graph, the
policy engine, the orchestrator, the audit trail and the metrics are all
transport- and model-agnostic already.

Concretely, the swap is **one function**:

```python
# app/voice/agent.py
def build_session() -> AgentSession:
    return AgentSession(
        stt=sarvam.STT(...),      # -> local ASR endpoint
        llm=build_llm(),          # -> local vLLM endpoint
        tts=cartesia.TTS(...),    # -> local VoxCPM endpoint
        vad=silero.VAD.load(),    # already local
        turn_detection="stt",
    )
```

`silero` VAD already runs locally. LiveKit is open source and self-hostable,
so the SFU can move in-VPC too, though that is separable and lower value.

This is the payoff from having kept traversal in `walker.py` and providers in
`build_session()`: swapping the entire model stack does not touch a single
line of the recovery logic, and the 170 tests keep passing throughout because
none of them mock a provider.

## Target topology

```
                 ap-south-1 / asia-south1 VPC
  ┌───────────────────────────────────────────────────────┐
  │                                                       │
  │   [GPU node]                     [CPU node]           │
  │   ├── ASR service  (LoRA'd)      ├── FastAPI webhooks │
  │   ├── LLM   (vLLM)               ├── worker / policy  │
  │   └── TTS   (VoxCPM 2)           └── LiveKit SFU      │
  │                                                       │
  │                    [Postgres]                         │
  └───────────────────────────────────────────────────────┘
            │                              │
      SIP carrier                    Razorpay API
      (sees: phone number)           (payments, links)
```

Public surface stays exactly two endpoints: the Razorpay webhook, and the SIP
signalling path. Everything else is private.

## Components

### TTS — decided: VoxCPM 2 + voice cloning

Already chosen, and it does double duty: it is also the data generator for the
ASR fine-tune below. Cloning gives us acoustic diversity for that synthetic
corpus, which the flywheel paper found mattered — they deliberately used
several TTS systems and held one out for unbiased evaluation.

Practical note: cache the fixed phrases. The greeting and the sign-offs are the
same every call, and pre-rendering them removes TTS from the critical path for
the two turns where latency is most noticeable.

### STT — the decision that actually matters

Candidates:

| Model | Licence | Notes |
|---|---|---|
| `ai4bharat/indic-conformer-600m-multilingual` | MIT | 22 languages, streaming-friendly, small |
| `vasista22/whisper-*-large-v2` | open | the open SOTA baseline the flywheel paper fine-tunes |
| `faster-whisper` (CTranslate2) | MIT | fastest path to streaming; wraps Whisper checkpoints |

Realistic Hindi WER today: **12–18% clean, 22–30% telephony**. Budget for the
telephony number — that is the one our calls will actually see.

Our own prior benchmark of Gemma-4 audio put Hindi at **15.6% WER**, strong for
Hindi and unusable for Dravidian languages. Consistent with the above, and a
reminder that a good aggregate WER says nothing about the failure mode we care
about.

### LLM

[Sarvam 30B and 105B are open-weight under Apache 2.0 as of March 2026](https://www.sarvam.ai/blogs/sarvam-30b-105b),
trained from scratch for Indic. We already have **Sarvam-30B running on vLLM**,
including the two footguns it ships with (`<|nothink|>` is inert; prefill
`<think></think>` instead), and measured **4.9× decode throughput versus
Gemma**. Starting from a stack we have already debugged is worth more than a
marginally better leaderboard score.

**But size is probably the wrong axis here.** The graph does the reasoning. Per
turn the model must: pick one of ~4 edge labels, and produce one or two
sentences in the customer's language. That is not a 30B-class task. It is a
latency task, and a reasoning model emitting thinking tokens is actively
working against us.

**Recommendation:** benchmark Sarvam-30B against a 4–8B (Qwen3 class) *on our
actual node prompts*, and choose on p95 first-token latency, not on benchmarks.
A small model inside a tightly constrained graph is the design we already
committed to; this is where that pays off.

## The entity problem is the real one

Standard WER hides the failure that would break this product. From
[The TTS-STT Flywheel (arXiv 2605.03073)](https://arxiv.org/html/2605.03073),
on entity-dense audio — digits, currency, addresses, brands, code-mix:

| System | Entity-Hit-Rate |
|---|---|
| Open SOTA (`vasista22` whisper-large-v2) | 0.027 |
| Deepgram Nova-3 (commercial) | 0.16 |

Both score respectably on read prose. Both fall apart on the content a billing
call is made of. **This is not fixed by buying a better hosted API** — the
commercial system is only ~6× better than open source, and both are bad.

Their method, which we can run almost as-is:

1. Generate entity-tagged carrier utterances with an LLM, seeded from curated
   dictionaries across six entity classes.
2. Render ~22k of them through **several** TTS voices for acoustic diversity.
3. LoRA fine-tune the base ASR on that corpus.
4. Hold out one TTS system entirely, to prove you did not just learn one
   vocoder's artefacts.

Reported: Hindi EHR **0.337 (7×)**, Telugu 17×, Tamil 22×, for **$16–50** in
compute versus ~$660 for equivalent human transcription. Gains held up on real
human speech (0.516 native vs 0.473 synthetic).

Cost: **+6.6pp read-prose regression** on FLEURS. We are trading general
accuracy for financial accuracy. For this product that is the right trade, but
it is a trade and should be stated in the README rather than discovered later.

### Scope it before building it

Before fine-tuning anything, measure which entities actually arrive *inbound*.

The amount flows **outbound** — TTS reads ₹499 out of Postgres; the ASR never
has to hear it. What arrives inbound is overwhelmingly intent: *haan*, *nahi*,
*baad mein*, *galat number*. Numbers only matter inbound for callback times
("kal shaam paanch baje") and possibly a card's last four digits.

That is a far narrower target than "all financial entities", and a narrower
fine-tune. **Action: label the inbound entity distribution across ~20 real
calls first.** It may turn out the flywheel is worth building only for time
expressions, which is a day of work rather than a week.

## Latency budget

Conversational voice needs first audio inside ~500ms of end-of-speech.

| Stage | Budget | Lever if over |
|---|---|---|
| VAD end-of-speech | ~100ms | silero, already local |
| ASR final transcript | ~150ms | streaming decode; smaller model |
| LLM first token | ~150ms | small model; no thinking tokens; prefix cache |
| TTS first audio | ~100ms | streaming synth; cache fixed phrases |

Every hop stays in-VPC, which removes the public-internet round trips the
hosted stack pays three times per turn. That is the one place local should be
*faster*, and it is worth measuring rather than assuming.

## Risks

- **It is days of work, and the hosted stack already works.** A working demo on
  hosted models beats a half-working sovereign one. If the deadline is close,
  ship hosted and present this document as the production design.
- **Quality regression is likely at first.** Cartesia and Sarvam's hosted models
  are strong. Expect the first local build to sound worse.
- **Ops burden moves to us**: GPU memory, model loading, restarts. We already
  learned tonight that an unsupervised worker dies on a network blip; a GPU node
  has more failure modes, not fewer.
- **Telephony stays third-party regardless.** "Fully local" is honest about
  inference, not about the carrier. Say so plainly — a judge will ask.

## Open decisions

1. ASR base: IndicConformer-600M vs `vasista22` Whisper-large-v2.
2. LLM size: Sarvam-30B (already debugged) vs a 4–8B chosen on latency.
3. Whether to self-host LiveKit, or accept LiveKit Cloud for signalling since it
   never sees decrypted audio content we care about. Lower priority than the
   model layer.
4. Whether the entity fine-tune is worth it at all, pending the inbound entity
   measurement above.
