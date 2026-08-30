"""Speech in and speech out, on our own GPU.

    modal deploy scripts/modal_speech.py

The other half of the local pipeline. `modal_llm.py` runs the model that
decides what to say; this runs the ears and the voice, so that a recovery call
can happen without a customer's audio reaching anybody's API.

Two models on one GPU, deliberately:

* **SraVaani-1.0** (ARTPARK/IISc, MIT) for speech to text. 430M parameters,
  65 Indian languages, and -- the reason it is here rather than Whisper --
  trained with code-switching tagging. Real customers on an Indian line speak
  Hindi and English in the same sentence, and our own transcripts are full of
  it.
* **VoxCPM2** (OpenBMB, Apache 2.0) for text to speech. 2B, 30 languages
  including Hindi, 48kHz, and it can clone a voice from a short clip -- which
  is a thing no TTS vendor will give you: a brand voice you own rather than
  rent.

Together they are ~2.5B parameters and around 10GB, which fits one L4 with
room to spare. They share a container because they are both plain torch: the
LLM lives in its own app because vLLM's CUDA stack does not want to share an
image with anything, and finding that out the hard way costs a build.

What this does not solve: SraVaani transcribes a file, not a stream. The voice
pipeline already segments on VAD, so it can hand over whole utterances -- but
that is a round trip per turn, not continuous recognition, and it is the first
number to look at when the latency is disappointing.
"""

import io
import os

import modal

STT_MODEL = "ARTPARK-IISc/SraVaani-1.0"
TTS_MODEL = "openbmb/VoxCPM2"

#: Both models on one 24GB card. VoxCPM wants ~8GB, SraVaani under 1GB, and
#: the rest is headroom for concurrent requests during a call.
GPU_TYPE = "L4"

#: Twenty minutes, which is expensive and correct. At 120s the container died
#: between warming it and the phone ringing, so the greeting hit a cold start:
#: 264s of model loading against a client that gives up at 120. The customer
#: heard silence and the mute-call guard recorded the call as failed.
#:
#: A cold start in the middle of a call is not a slow call, it is a dead one.
#: This is the one place where paying for idle is the right trade -- but it is
#: also the number to put back down the moment testing stops.
SCALEDOWN_SECONDS = 5 * 60

#: Shared with modal_llm.py so the weights are cached once for the project.
hf_cache = modal.Volume.from_name("dunning-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch",
        "transformers",
        "voxcpm",
        "soundfile",
        "numpy",
        "fastapi[standard]",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/root/.cache/huggingface"})
)

app = modal.App("dunning-speech")


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/root/.cache/huggingface": hf_cache},
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=20 * 60,
    # SraVaani is gated: accept its terms on HuggingFace, then
    #   modal secret create huggingface HF_TOKEN=hf_...
    secrets=[modal.Secret.from_name("huggingface")],
)
@modal.concurrent(max_inputs=8)
class Speech:
    """Both models, loaded once per container rather than once per request.

    `@modal.enter` is the whole point of the class: loading 2.5B parameters
    takes tens of seconds, and doing it inside a request handler would put that
    in the middle of somebody's phone call.
    """

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModel
        from voxcpm import VoxCPM

        token = os.environ.get("HF_TOKEN")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"loading {STT_MODEL}")
        self.stt = (
            AutoModel.from_pretrained(STT_MODEL, trust_remote_code=True, token=token)
            .to(device)
            .eval()
        )
        print(f"loading {TTS_MODEL}")
        # The denoiser is for noisy reference clips we are not using, and it is
        # weight and startup time we would pay for on every cold start.
        self.tts = VoxCPM.from_pretrained(TTS_MODEL, load_denoiser=False)
        self.sample_rate = self.tts.tts_model.sample_rate
        print(f"ready: TTS sample rate {self.sample_rate}")

    def _transcribe(self, audio: bytes) -> str:
        """One utterance in, text out. The pipeline segments on VAD upstream."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(audio)
            path = handle.name
        try:
            hypotheses = self.stt.transcribe(path, return_hypotheses=True)
        finally:
            os.unlink(path)
        if not hypotheses:
            return ""
        first = hypotheses[0]
        return getattr(first, "text", str(first))

    def _pcm(self, chunk) -> bytes:
        """Whatever VoxCPM yields, as 16-bit PCM.

        It hands back float32 in [-1, 1]; the phone line wants int16. Clipping
        rather than scaling, because a chunk that momentarily exceeds 1.0
        should distort in that instant, not quietly change the volume of the
        whole utterance relative to its neighbours.
        """
        import numpy as np

        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    def _speak_stream(self, text: str):
        """Yield raw PCM as it is generated, rather than a finished clip.

        This is the whole latency story. Synthesising a sentence and then
        sending it costs the customer the entire generation time before they
        hear a syllable -- measured at 7.7s for one sentence on this GPU.
        Streaming makes them wait for the first chunk instead.

        Raw PCM and not WAV: a WAV header declares a length we do not know
        when the first chunk leaves, and the pipeline is told the sample rate
        out of band anyway.
        """
        for chunk in self.tts.generate_streaming(
            text=text, cfg_value=2.0, inference_timesteps=10
        ):
            data = self._pcm(chunk)
            if data:
                yield data

    def _speak(self, text: str) -> bytes:
        """The whole clip, for tests and for anything that cannot stream."""
        import soundfile as sf

        wav = self.tts.generate(text=text, cfg_value=2.0, inference_timesteps=10)
        buffer = io.BytesIO()
        sf.write(buffer, wav, self.sample_rate, format="WAV")
        return buffer.getvalue()

    @modal.asgi_app()
    def web(self):
        """HTTP rather than Modal's own RPC, for two reasons.

        The pipeline runs on Cloud Run and has no Modal client, so it needs a
        plain URL. And `modal run` streams over a gRPC connection held open for
        the whole call -- which on a flaky link dies during a cold start and
        takes the result with it. A request that can be retried is worth more
        here than one that is slightly faster.
        """
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, Response, StreamingResponse

        api = FastAPI()

        @api.get("/health")
        async def health():
            return {"ok": True, "stt": STT_MODEL, "tts": TTS_MODEL, "sr": self.sample_rate}

        @api.post("/transcribe")
        async def transcribe(request: Request):
            audio = await request.body()
            return JSONResponse({"text": self._transcribe(audio)})

        @api.post("/speak")
        async def speak(request: Request):
            """Raw 16-bit PCM at 48kHz, streamed. The caller resamples."""
            body = await request.json()
            text = (body or {}).get("text", "")
            if not text.strip():
                return JSONResponse({"error": "no text"}, status_code=400)
            return StreamingResponse(
                self._speak_stream(text),
                media_type="audio/L16",
                headers={"X-Sample-Rate": str(self.sample_rate)},
            )

        @api.post("/speak_wav")
        async def speak_wav(request: Request):
            """The whole clip in one response. Kept for tests and for curl."""
            body = await request.json()
            text = (body or {}).get("text", "")
            if not text.strip():
                return JSONResponse({"error": "no text"}, status_code=400)
            return Response(content=self._speak(text), media_type="audio/wav")

        return api
