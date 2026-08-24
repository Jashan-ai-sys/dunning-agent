"""Deterministic rewrites applied to spoken text just before synthesis.

Ported from the Blostem production backend, whose rationale is worth keeping
verbatim: a system prompt asking the model to speak amounts in words is not
enough, because "LLMs drift and still emit raw digits". On times they are
blunter -- "A prompt rule for this was available and did not fix it."

These rewrite only the bytes handed to the TTS socket. The LLM context, the
stored transcript and the audit trail keep the original text.

Pure functions with no Pipecat dependency, so they work under either transport.
"""

from app.voice.normalizers.amount_normalizer import normalize_amounts_for_tts
from app.voice.normalizers.hindi_normalizer import normalize_for_hindi_tts, strip_markdown
from app.voice.normalizers.time_normalizer import normalize_times_for_tts

__all__ = [
    "normalize_amounts_for_tts",
    "normalize_for_hindi_tts",
    "normalize_times_for_tts",
    "strip_markdown",
]
