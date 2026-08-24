"""Render amounts the way a TTS voice should say them.

Adopted from the Blostem production backend's ``tts_amount_normalizer``, which
exists because a system prompt asking the LLM to speak amounts in words is not
enough -- models drift and emit raw digits anyway, and a Hindi-configured voice
then reads a long digit run as a digit crawl.

We have an advantage that normalizer did not: the amount comes out of our own
database, so we can hand the model the spoken form up front instead of
rewriting its output afterwards. That removes the streaming-chunk hazard
entirely -- a normalizer running over streamed LLM text can split "499" across
two chunks and mangle it.

The convention is theirs and is load-bearing: keep the ASCII numeral and append
a unit word in the target language. Cartesia at ``language="hi"`` reads a bare
numeral correctly in Hindi; it is the *magnitude* and the Latin currency token
that break it. So ``500000`` becomes ``5 लाख रुपये``, not a spelled-out
``पाँच लाख`` and not ``Rs 500000``.

Only clean magnitudes collapse. A messy amount stays as its own numeral rather
than becoming an unnatural "5.4321 लाख".
"""

from decimal import Decimal

#: Unit words per language. Keys match ``Customer.preferred_language``.
#:
#: Hinglish takes the Devanagari words, not romanised ones. Hinglish is a
#: speech register rather than a script: a Hinglish speaker says "लाख" and
#: "रुपये" like any Hindi speaker, and the voice rendering it is pinned to
#: ``language="hi"``. Feeding that voice "5 lakh rupaye" is the script flip
#: this module exists to prevent -- it would read the Latin tokens as English
#: mid-sentence, which is exactly the digit-crawl failure in a different coat.
_UNITS = {
    "hi": {"thousand": "हज़ार", "lakh": "लाख", "crore": "करोड़", "rupees": "रुपये"},
    "hinglish": {"thousand": "हज़ार", "lakh": "लाख", "crore": "करोड़", "rupees": "रुपये"},
    "en": {"thousand": "thousand", "lakh": "lakh", "crore": "crore", "rupees": "rupees"},
}

_CRORE = 10_000_000
_LAKH = 100_000
_THOUSAND = 1_000


def _trim(value: Decimal) -> str:
    """One decimal place at most, and no trailing '.0'."""
    quantised = value.quantize(Decimal("0.1"))
    whole = quantised.to_integral_value()
    return str(whole) if quantised == whole else str(quantised)


def _is_clean(rupees: int, unit: int) -> bool:
    """True if the amount collapses to at most one decimal place of `unit`."""
    return (rupees * 10) % unit == 0


def spoken_amount(paise: int, language: str = "hi") -> str:
    """Format paise as something a TTS voice reads naturally.

    >>> spoken_amount(49900)
    '499 रुपये'
    >>> spoken_amount(50000000)
    '5 लाख रुपये'
    """
    units = _UNITS.get(language or "hi", _UNITS["hi"])
    rupees, remainder = divmod(max(paise, 0), 100)

    # Paise remainders are rare on subscriptions and reading them aloud is
    # noise; the link shows the exact figure.
    if remainder:
        rupees += 1

    if rupees >= _CRORE and _is_clean(rupees, _CRORE):
        magnitude, word = Decimal(rupees) / _CRORE, units["crore"]
    elif rupees >= _LAKH and _is_clean(rupees, _LAKH):
        magnitude, word = Decimal(rupees) / _LAKH, units["lakh"]
    elif rupees >= _THOUSAND and rupees % _THOUSAND == 0:
        magnitude, word = Decimal(rupees) / _THOUSAND, units["thousand"]
    else:
        # Messy or small: the bare numeral reads fine.
        return f"{rupees} {units['rupees']}"

    return f"{_trim(magnitude)} {word} {units['rupees']}"
