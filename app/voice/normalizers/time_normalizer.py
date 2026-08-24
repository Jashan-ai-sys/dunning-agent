"""Render clock times and dates as SPOKEN words before they reach TTS.

The third member of a family. Amounts already have
``tts_amount_normalizer`` (500000 -> "5 लाख"); phone numbers have a guardrail
prompt rule. Time had nothing, and it shows: on call 8e1a5a10 the agent said

    "मैंने आपका callback 11 Aug, 7:30 PM के लिए schedule कर दिया है।"

Latin abbreviations and a colon dropped into a Hindi sentence, handed to a
Cartesia voice configured ``language=hi``. Our own tts_language_switcher notes
why that breaks: "Cartesia (and most TTS engines) pronounce DIGITS according to
the configured language" — so "7:30" and "11" get Hindi digit treatment while
"Aug"/"PM" are Latin tokens the Hindi voice has to guess at.

Nothing was going to catch it. The ElevenLabs normalizer is set on a different
provider row; ``_patch_cartesia_normalize_amounts`` excludes dates and times by
design; and the guardrails time rule is a PROMPT instruction, which tells the
model how to answer rather than rewriting the string before synthesis.

DETERMINISTIC, like the amount normalizer: the spoken bytes are rewritten, the
LLM context and stored transcript keep the original. A prompt rule for this was
available and did not fix it.

Hindi uses the natural forms rather than a digit crawl — साढ़े/सवा/पौने — because
"सात बज कर तीस मिनट" is what a clock reads, and "साढ़े सात बजे" is what a person
says. The period word is derived from the hour (सुबह/दोपहर/शाम/रात), so "7:30 PM"
becomes "शाम साढ़े सात बजे" rather than transliterating "PM".
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

#: 7:30 PM / 7:30PM / 07:30 pm — the shape an LLM writes a callback time in.
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*([APap]\.?[Mm]\.?)?")

#: "11 Aug" / "11 August" / "Aug 11" — the shape it writes a callback date in.
_DATE_DM_RE = re.compile(r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b")
_DATE_MD_RE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})\b")

_MONTHS_HI = {
    "jan": "जनवरी", "feb": "फ़रवरी", "mar": "मार्च", "apr": "अप्रैल",
    "may": "मई", "jun": "जून", "jul": "जुलाई", "aug": "अगस्त",
    "sep": "सितंबर", "oct": "अक्टूबर", "nov": "नवंबर", "dec": "दिसंबर",
}
_MONTHS_EN = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "oct": "October", "nov": "November", "dec": "December",
}

_HOURS_HI = {
    1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पाँच", 6: "छह",
    7: "सात", 8: "आठ", 9: "नौ", 10: "दस", 11: "ग्यारह", 12: "बारह",
}
_MINUTES_HI = {
    5: "पाँच", 10: "दस", 20: "बीस", 25: "पच्चीस", 35: "पैंतीस",
    40: "चालीस", 50: "पचास", 55: "पचपन",
}


def _period_hi(hour24: int) -> str:
    """सुबह / दोपहर / शाम / रात from the hour.

    Derived rather than transliterated: a Hindi speaker says "शाम साढ़े सात बजे",
    never "साढ़े सात बजे पी एम".
    """
    if 4 <= hour24 < 12:
        return "सुबह"
    if 12 <= hour24 < 16:
        return "दोपहर"
    if 16 <= hour24 < 20:
        return "शाम"
    return "रात"


def _time_hi(hour12: int, minute: int, hour24: int) -> str:
    period = _period_hi(hour24)
    nxt = _HOURS_HI.get(hour12 % 12 + 1, "")
    cur = _HOURS_HI.get(hour12, str(hour12))
    if minute == 0:
        return f"{period} {cur} बजे"
    if minute == 30:
        return f"{period} साढ़े {cur} बजे"
    if minute == 15:
        return f"{period} सवा {cur} बजे"
    if minute == 45 and nxt:
        return f"{period} पौने {nxt} बजे"
    mins = _MINUTES_HI.get(minute)
    if mins:
        return f"{period} {cur} बज कर {mins} मिनट"
    # Anything unusual (7:37) stays a digit crawl rather than an invented word.
    return f"{period} {cur} बज कर {minute} मिनट"


def _time_en(hour12: int, minute: int, meridiem: str) -> str:
    base = f"{hour12} {minute:02d}" if minute else f"{hour12} o'clock"
    return f"{base} {meridiem}".strip() if meridiem else base


def normalize_times_for_tts(text: str, lang: str = "hi") -> tuple[str, list[tuple[str, str]]]:
    """Rewrite clock times and month-dates as spoken words.

    Returns ``(text, replacements)`` so the caller can log exactly what changed
    — the amount normalizer's contract, for the same reason: a silent rewrite
    of what the customer hears is not something to discover from a recording.

    Conservative by construction. Only ``H:MM`` with an optional meridiem and
    ``DD Mon`` / ``Mon DD`` are touched. Bare numbers, amounts, phone numbers,
    OTPs and reference numbers are left exactly as they are — those belong to
    tts_amount_normalizer and the phone guardrail respectively, and overlapping
    rewrites would fight each other.
    """
    if not text:
        return text, []
    base = (lang or "hi").split("-")[0].lower()
    hindi = base == "hi"
    out: list[tuple[str, str]] = []

    def _sub_time(m: re.Match) -> str:
        raw = m.group(0)
        try:
            hour = int(m.group(1))
            minute = int(m.group(2))
        except (TypeError, ValueError):
            return raw
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return raw
        mer = (m.group(3) or "").replace(".", "").upper()
        if mer == "PM" and hour < 12:
            hour24 = hour + 12
        elif mer == "AM" and hour == 12:
            hour24 = 0
        else:
            hour24 = hour
        hour12 = hour24 % 12 or 12
        spoken = _time_hi(hour12, minute, hour24) if hindi else _time_en(hour12, minute, mer)
        out.append((raw, spoken))
        return spoken

    def _sub_dm(m: re.Match) -> str:
        raw, day, mon = m.group(0), m.group(1), m.group(2).lower()[:3]
        name = (_MONTHS_HI if hindi else _MONTHS_EN).get(mon)
        if not name:
            return raw
        spoken = f"{day} {name}"
        out.append((raw, spoken))
        return spoken

    def _sub_md(m: re.Match) -> str:
        raw, mon, day = m.group(0), m.group(1).lower()[:3], m.group(2)
        name = (_MONTHS_HI if hindi else _MONTHS_EN).get(mon)
        if not name:
            return raw
        spoken = f"{day} {name}" if hindi else f"{name} {day}"
        out.append((raw, spoken))
        return spoken

    # Dates before times: "11 Aug, 7:30 PM" must not have its "11" consumed by
    # a time pattern first.
    result = _DATE_DM_RE.sub(_sub_dm, text)
    result = _DATE_MD_RE.sub(_sub_md, result)
    result = _TIME_RE.sub(_sub_time, result)
    return result, out
