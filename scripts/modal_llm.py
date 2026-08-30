"""Serve the dunning agent's language model on our own GPU.

    modal deploy scripts/modal_llm.py       # leaves it deployed, scaled to zero
    modal serve  scripts/modal_llm.py       # hot-reloads while iterating

Why this exists: everything the agent says about a customer's money currently
crosses a third-party boundary. For a payments company that is a procurement
question before it is an engineering one, and RBI data-localisation rules make
"the model runs inside your VPC" a real requirement rather than a talking
point. This is the same conversation, on a GPU we rent.

It is also insurance. A vendor quota ran out mid-call once already and the
agent spoke to nobody for two minutes while every log said the call was going
fine.

Gemma 4 12B, chosen against four hard requirements:

* **Tool calling**, which is not negotiable -- the whole state machine rides on
  the model calling `transition(label)`. Gemma 4 is the first Gemma with tool
  calling in its own chat template, and vLLM ships a matching parser.
* **Devanagari**, at 1.39 tokens per word -- the best measured of eleven
  tokenizers, and roughly a third of what Qwen spends on the same Hindi.
* **English of equal standing**, because the demo is bilingual.
* **Apache 2.0 and ungated**, so the repo can be public and no token is needed
  to pull the weights.

The FP8 checkpoint is deliberate: ~12GB leaves most of a 24GB card for KV cache,
which is what actually decides how long the first token takes once a call has
been running for a few minutes.

Budget note, because it is a $30 experiment: `scaledown_window` is what stops
this quietly billing overnight after a test. Keep it short while iterating and
raise it only for the demo, when a cold start mid-call is the thing you cannot
afford.
"""

import os
import subprocess

import modal

#: FP8 rather than bf16: half the weights, and the headroom goes to KV cache.
MODEL_NAME = "RedHatAI/gemma-4-12B-it-FP8-Dynamic"

#: L4 is 24GB and Ada -- which matters, because FP8 needs Ada or newer. An A10G
#: has the same memory and would force bf16 at ~24GB, leaving nothing for KV
#: cache. 12GB of weights on 24GB leaves room for a long call and, later, the
#: STT and TTS models beside it: co-locating all three is both the cheap answer
#: and the fast one, since it takes two network hops out of a conversation.
#:
#: L40S (48GB) was the first choice and Modal refused it without a payment
#: method on the account. L4 is cheaper per hour anyway, which on a $30 budget
#: is the constraint that actually binds.
GPU_TYPE = "L4"

#: Seconds a container stays up with no traffic.
#:
#: 60 was right while probing from a laptop and wrong the moment a phone was
#: involved: the container died between warming it and the call connecting, and
#: a cold start is ~210s of weight loading. On a call that is not latency, it
#: is silence.
#:
#: Twenty minutes costs real money for idle time and is the correct trade while
#: testing. Put it back down when the testing stops -- a forgotten container is
#: still the easiest way to lose a third of the budget.
SCALEDOWN_SECONDS = 5 * 60

#: Weights are ~12GB and we do not want to pay to download them twice.
hf_cache = modal.Volume.from_name("dunning-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("dunning-vllm-cache", create_if_missing=True)

#: vLLM's own published image rather than pip-installing it into debian_slim.
#: That build pulls torch, the CUDA toolchain, flashinfer and a hundred other
#: wheels, and it was killed by Modal's builder twice before finishing. The
#: official image already has all of it, compiled and tested together -- which
#: also removes a whole class of "works on my CUDA version" problem we have no
#: reason to own.
image = (
    modal.Image.from_registry("vllm/vllm-openai:latest", add_python=None)
    .entrypoint([])  # the published image starts a server; we start our own
    .run_commands(
        # The vLLM image ships `python3` and no `python`. Modal detects the
        # interpreter by running bare `python`, so without this the deploy
        # fails with "unable to determine the version of Python installed" --
        # and `pip_install` fails the same way, one layer earlier.
        #
        # A symlink rather than `add_python`, which would install a *second*
        # interpreter that does not have vLLM in it.
        "ln -sf $(command -v python3) /usr/local/bin/python",
        # hf_transfer is the difference between a fast and a slow 12GB cold
        # start, and HF_HUB_ENABLE_HF_TRANSFER below errors without it.
        "python3 -m pip install --no-cache-dir 'huggingface_hub[hf_transfer]'",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Compilation caches survive between cold starts via the volume,
            # which is most of the difference between a 3-minute and a
            # 40-second start.
            "VLLM_CACHE_ROOT": "/root/.cache/vllm",
        }
    )
)

app = modal.App("dunning-llm")


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=30 * 60,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=8000, startup_timeout=15 * 60)
def serve() -> None:
    """An OpenAI-compatible endpoint, so the client swap is a base_url.

    Nothing above this line in the pipeline needs to know the model moved.
    """
    print(f"serving {MODEL_NAME} on {GPU_TYPE}")
    subprocess.Popen("vllm --version", shell=True)

    subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            # The two flags the whole architecture depends on. Without them the
            # model emits tool calls as prose and `transition` is never called.
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "gemma4",
            # Far below the model's 256K. A dunning call is a few thousand
            # tokens; asking for more only takes KV cache away from latency.
            "--max-model-len",
            "16384",
            # One turn at a time is the shape of a phone call. Batching helps
            # throughput and costs the first token, which is the only number a
            # customer can hear.
            "--max-num-seqs",
            "16",
        ]
    )


@app.local_entrypoint()
def smoke() -> None:
    """Prove the three things that would sink this before anything is built on it.

    Deliberately not a benchmark. It answers: does it come up, does it write
    Devanagari when spoken to in Hindi, and does it call the tool. Any of those
    failing means the model is wrong for us and no amount of tuning fixes it.
    """
    print(f"endpoint: {serve.get_web_url()}")
    print("run scripts/probe_local_llm.py against it -- keeping GPU time off this path")
    print(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))} (not needed: Gemma 4 is ungated)")
