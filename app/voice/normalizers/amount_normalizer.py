"""TTS amount normalizer — rewrite rupee amounts to conversational Indian units.

Why this exists
===============
Reading a bare ``500000`` aloud is slow, robotic, and fragile on 8kHz
telephony: the TTS engine recites "five zero zero zero zero zero" or a
digit-group crawl, and any downstream STT re-transcription of the recording
mangles long digit runs (Sarvam saaras mis-hears spoken numbers on telephony).
Indian customers expect "5 lakh", "50 hazaar", "2 crore".

The system prompt already asks the LLM to speak amounts in words
(``_amount_normalization_prompt_rule``), but LLMs drift and still emit raw
digits. This module is the deterministic safety net applied to the SPOKEN text
just before synthesis — the LLM context, transcript and RAG memory keep the
ORIGINAL text (same contract as ``tts_hindi_normalizer``).

Design: conservative by construction
====================================
We only rewrite a number when it carries an explicit **currency cue** — a
leading ``₹`` / ``Rs`` / ``INR`` / ``रु`` / ``रुपये`` token, or a trailing
``रुपये`` / ``rupees`` / ``/-``. That single rule is what keeps OTPs, phone
numbers, account numbers, reference IDs, PINs, dates, interest rates and loan
tenures untouched — none of those are written with a rupee marker. We also:

  * skip anything with a decimal fraction (``5,00,000.50`` → left alone) so we
    never split a rate or a paise amount;
  * only collapse "clean" magnitudes (whole thousands, ≤1-decimal lakhs,
    ≤1-decimal crores). A messy amount like ``5,43,210`` is left exactly as the
    LLM wrote it rather than risk an unnatural "5.4321 lakh".

Numerals are kept as ASCII digits followed by a Devanagari (hi) or English
unit word — e.g. ``5 लाख`` / ``5 lakh``. This mirrors the trusted
``emi_calculator._inr_hinglish`` convention: Cartesia at ``language="hi"``
reads the bare numeral in Hindi and the unit word anchors the magnitude.
"""

from __future__ import annotations

import re

# Unit words per language family.
_UNITS_HI = {"crore": "करोड़", "lakh": "लाख", "thousand": "हज़ार"}
_UNITS_EN = {"crore": "crore", "lakh": "lakh", "thousand": "thousand"}

_CRORE = 10_000_000
_LAKH = 100_000
_THOUSAND = 1_000


def _fmt_num(value: float) -> str:
    """Render a magnitude count: drop a trailing ``.0`` (5.0 -> "5"), keep one
    meaningful decimal (5.5 -> "5.5")."""
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _inr_to_words(amount: int, lang: str) -> str | None:
    """Collapse a whole-rupee integer to a conversational unit phrase.

    Returns None when the amount is not a "clean" conversational magnitude
    (caller then leaves the original text untouched).

      500000  -> "5 लाख"    (hi)  / "5 lakh"    (en)
      540000  -> "5.4 लाख"
      50000   -> "50 हज़ार"
      20000000-> "2 करोड़"
      543210  -> None       (not clean — left as digits)
      500     -> None       (too small)
    """
    units = _UNITS_HI if lang.lower().startswith("hi") else _UNITS_EN
    if amount >= _CRORE:
        # Clean to ≤1 decimal of a crore (multiples of 10 lakh).
        if amount % (_CRORE // 10) == 0:
            return f"{_fmt_num(amount / _CRORE)} {units['crore']}"
        return None
    if amount >= _LAKH:
        # Clean to ≤1 decimal of a lakh (multiples of 10 thousand).
        if amount % (_LAKH // 10) == 0:
            return f"{_fmt_num(amount / _LAKH)} {units['lakh']}"
        return None
    if amount >= _THOUSAND:
        if amount % _THOUSAND == 0:
            return f"{_fmt_num(amount / _THOUSAND)} {units['thousand']}"
        return None
    return None


# A rupee amount is ``\d[\d,]*`` (optionally Indian/Western grouped) that is NOT
# immediately followed by another digit or a decimal point — the trailing
# ``(?![\d.])`` guard is what makes us skip decimal amounts and long OTP/phone
# digit runs that we happened to anchor mid-way.
_NUM = r"(\d[\d,]*)(?![\d.])"

# Leading currency cue we CONSUME (the unit word we emit conveys "rupees", so
# a redundant ``₹`` prefix is dropped). ``Rs``/``INR`` are ASCII-word-bounded.
_LEADING_CUE = re.compile(
    r"(?:₹|(?<![A-Za-z])(?:Rs\.?|INR|रुपये|रुपए|रु\.?))\s?" + _NUM,
    re.IGNORECASE,
)

# Trailing rupee cue is a LOOKAHEAD (not consumed) so "500000 रुपये" becomes
# "5 लाख रुपये" rather than dropping the रुपये.
_TRAILING_CUE = re.compile(
    _NUM + r"(?=\s*(?:रुपये|रुपए|rupees|rupaye|/-))",
    re.IGNORECASE,
)


def _parse_int(token: str) -> int | None:
    digits = token.replace(",", "")
    if not digits.isdigit():
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def normalize_amounts_for_tts(
    text: str, lang: str = "hi"
) -> tuple[str, list[tuple[str, str]]]:
    """Return ``(normalized_text, replacements)``.

    ``replacements`` is a list of ``(original, spoken)`` pairs for logging.
    Pure and side-effect-free; safe to call on every utterance. Leaves the
    string untouched when no currency-marked, cleanly-collapsible amount is
    present.
    """
    if not text or not any(c.isdigit() for c in text):
        return text, []

    replacements: list[tuple[str, str]] = []

    def _sub_leading(m: re.Match[str]) -> str:
        amount = _parse_int(m.group(1))
        if amount is None:
            return m.group(0)
        words = _inr_to_words(amount, lang)
        if words is None:
            return m.group(0)  # not clean — leave the whole ₹NNN token as-is
        replacements.append((m.group(0), words))
        return words

    def _sub_trailing(m: re.Match[str]) -> str:
        amount = _parse_int(m.group(1))
        if amount is None:
            return m.group(0)
        words = _inr_to_words(amount, lang)
        if words is None:
            return m.group(0)
        replacements.append((m.group(1), words))
        return words  # lookahead cue (रुपये/…) is preserved by construction

    out = _LEADING_CUE.sub(_sub_leading, text)
    out = _TRAILING_CUE.sub(_sub_trailing, out)
    return out, replacements
