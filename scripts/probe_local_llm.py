"""Three questions to ask a candidate model before building anything on it.

    uv run python scripts/probe_local_llm.py https://<your-modal-url>

Not a benchmark -- a gate. Each check corresponds to a requirement the agent
cannot work without, and a model that fails one is disqualified regardless of
how it scores elsewhere.

1. **Devanagari.** The agent speaks Hindi in Hindi script. A model that answers
   code-mixed Hindi with romanised Hinglish is wrong for this product, and no
   prompt reliably argues a model out of the script it was trained toward.
2. **Tool calling.** Every outcome this system records comes from the model
   calling `transition(label)`. A model that describes the tool call in prose
   instead of emitting one records nothing, on a call that went perfectly.
3. **Time to first token.** The incumbent is Vertex Gemini 2.5 Flash at 0.313s
   p50, measured on real calls. Local has to beat that to be worth the swap;
   the customer hears this number and nothing else about where the model runs.

The prompts are deliberately shaped like real turns rather than clean test
sentences -- code-mixed, short, and mid-conversation, because that is what
comes off an Indian phone line and it is where models actually differ.
"""

import json
import sys
import time

import httpx

TIMEOUT = 120.0

TRANSITION_TOOL = {
    "type": "function",
    "function": {
        "name": "transition",
        "description": (
            "Move the conversation to the next stage. Call this as soon as a "
            "label matches what the customer just said."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["acknowledged", "disputes_charge"],
                    "description": (
                        "acknowledged: the customer acknowledges the failed payment "
                        "or asks what to do next. "
                        "disputes_charge: the customer says the charge is wrong, "
                        "that they already paid, or that they never signed up."
                    ),
                }
            },
            "required": ["label"],
        },
    },
}

SYSTEM = (
    "You are a polite payment-recovery agent for Acme, calling an Indian "
    "customer on the phone. Reply in Hindi, in Devanagari script. Never use "
    "romanised Hindi. Keep replies to one or two short sentences."
)


def _post(base: str, payload: dict) -> tuple[dict, float]:
    started = time.perf_counter()
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(
            f"{base.rstrip('/')}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
    response.raise_for_status()
    return response.json(), time.perf_counter() - started


def _model_id(base: str) -> str:
    with httpx.Client(timeout=TIMEOUT) as client:
        return client.get(f"{base.rstrip('/')}/v1/models").json()["data"][0]["id"]


def check_devanagari(base: str, model: str) -> bool:
    """Code-mixed in, Devanagari out. The way a real customer actually talks."""
    body, _ = _post(
        base,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "haan bhai, mera payment fail ho gaya, ab kya karun?"},
            ],
            "temperature": 0.0,
            "max_tokens": 120,
        },
    )
    reply = body["choices"][0]["message"]["content"] or ""
    devanagari = sum("ऀ" <= ch <= "ॿ" for ch in reply)
    letters = sum(ch.isalpha() for ch in reply)
    share = devanagari / letters if letters else 0.0

    print(f"\n[1] DEVANAGARI\n    reply: {reply.strip()[:160]}")
    print(f"    Devanagari share of letters: {share:.0%}")
    ok = share > 0.80
    print(f"    {'PASS' if ok else 'FAIL'} (need >80%; below that it is answering in Latin script)")
    return ok


def check_tool_call(base: str, model: str) -> bool:
    """The customer acknowledges. The correct move is `acknowledged`."""
    body, _ = _post(
        base,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "assistant", "content": "आपका ४९९ रुपये का पेमेंट फेल हो गया था।"},
                {"role": "user", "content": "अच्छा, तो अब मुझे क्या करना होगा?"},
            ],
            "tools": [TRANSITION_TOOL],
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 200,
        },
    )
    message = body["choices"][0]["message"]
    calls = message.get("tool_calls") or []

    print("\n[2] TOOL CALLING")
    if not calls:
        print(f"    no tool_calls. content: {(message.get('content') or '')[:160]}")
        print("    FAIL (describing the call in prose records nothing)")
        return False

    name = calls[0]["function"]["name"]
    try:
        label = json.loads(calls[0]["function"]["arguments"]).get("label")
    except json.JSONDecodeError:
        label = f"<unparseable: {calls[0]['function']['arguments'][:60]}>"
    print(f"    called {name}(label={label!r})")
    ok = name == "transition" and label == "acknowledged"
    print(f"    {'PASS' if ok else 'FAIL'} (expected transition(label='acknowledged'))")
    return ok


def check_ttft(base: str, model: str, runs: int = 5) -> bool:
    """Streaming, because time to the first *spoken* token is the whole game."""
    latencies = []
    for _ in range(runs):
        started = time.perf_counter()
        with httpx.Client(timeout=TIMEOUT) as client:
            with client.stream(
                "POST",
                f"{base.rstrip('/')}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": "मेरा कार्ड एक्सपायर हो गया है।"},
                    ],
                    "stream": True,
                    "temperature": 0.0,
                    "max_tokens": 60,
                },
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: ") and '"content"' in line:
                        latencies.append(time.perf_counter() - started)
                        break
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    print("\n[3] TIME TO FIRST TOKEN")
    print(f"    n={len(latencies)}  p50={p50:.3f}s  max={latencies[-1]:.3f}s")
    ok = p50 < 0.313
    verdict = "PASS" if ok else "SLOWER THAN CLOUD"
    print(f"    {verdict} (Vertex Gemini 2.5 Flash: 0.313s p50 on real calls)")
    return ok


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    base = sys.argv[1]
    model = _model_id(base)
    print(f"probing {model} at {base}")

    results = {
        "devanagari": check_devanagari(base, model),
        "tool_calling": check_tool_call(base, model),
        "ttft": check_ttft(base, model),
    }

    print("\n" + "=" * 60)
    for name, ok in results.items():
        print(f"  {name:14} {'PASS' if ok else 'FAIL'}")
    # Latency is a trade you can argue about. The other two are not: a model
    # that writes Latin script or will not call the tool cannot run this agent
    # at any speed.
    blocking = results["devanagari"] and results["tool_calling"]
    print(f"\n{'USABLE' if blocking else 'DISQUALIFIED'} for the dunning agent")
    return 0 if blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
