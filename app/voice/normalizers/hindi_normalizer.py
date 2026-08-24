"""Hindi TTS text normalizer — transliterate Latin tokens to Devanagari before
synthesis so Sarvam bulbul:v3 stops pausing/truncating at script boundaries.

Why this exists
===============
bulbul:v3 is a streaming model with much cleaner Hindi pronunciation than v2,
but it inserts hesitations — and sometimes drops the tail of a sentence — every
time it has to switch between Devanagari and Latin script *inside one sentence*.
Our Hinglish replies are full of such switches:

    "आपने हाल ही में Stable Money ऐप से Suryoday Bank में एक Fixed Deposit (FD)
     बनाने की शुरुआत की थी।"

That single line flips script five times. v2 papered over it (by mangling the
English words); v3 exposes it as choppiness and mid-sentence pauses.

What it does
============
A pure, reusable function that rewrites the **spoken** text only:
  1. Replaces known brand / product / banking phrases with a Devanagari
     spelling (longest match first, case-insensitive, ASCII-word-bounded).
  2. Spells out any remaining ALL-CAPS abbreviation (KYC, NEFT, IFSC...) as
     Devanagari letter-names so v3 reads "के वाई सी", not a stumbling "KYC".
  3. Leaves digits, ``%``, ``₹`` and unknown proper nouns untouched — those are
     either handled fine by the engine or safer left alone than mis-transliterated.

The LLM context, transcript and RAG memory keep the ORIGINAL text — only the
bytes handed to the TTS socket are normalized (see tts_factory wrapping).

Extending
=========
Add entries to ``GLOSSARY`` (keys are matched case-insensitively). Multi-word
keys win over single-word keys automatically because matching is longest-first.
"""

from __future__ import annotations

import re

# Devanagari letter-names for abbreviation spell-out (A B C ... Z).
_LETTER: dict[str, str] = {
    "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ़",
    "G": "जी", "H": "एच", "I": "आई", "J": "जे", "K": "के", "L": "एल",
    "M": "एम", "N": "एन", "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर",
    "S": "एस", "T": "टी", "U": "यू", "V": "वी", "W": "डब्ल्यू", "X": "एक्स",
    "Y": "वाई", "Z": "ज़ेड",
}

# Known phrases → Devanagari. Keys are case-insensitive. Order does not matter;
# matching is longest-first so "suryoday bank" beats a bare "bank".
GLOSSARY: dict[str, str] = {
    # Brand / product names from the Video KYC script
    "stable money": "स्टेबल मनी",
    "suryoday bank": "सूर्योदय बैंक",
    "shivalik small finance bank": "शिवालिक स्मॉल फाइनेंस बैंक",
    "apex bank": "एपेक्स बैंक",
    "union bank": "यूनियन बैंक",
    "fixed deposit": "फिक्स्ड डिपॉज़िट",
    "home loan": "होम लोन",
    "savings account": "सेविंग्स अकाउंट",
    "video kyc": "वीडियो के वाई सी",
    "video call": "वीडियो कॉल",
    # Abbreviations that are NOT spelled letter-by-letter (read as a word)
    "pan card": "पैन कार्ड",
    "pan": "पैन",
    "aadhaar": "आधार",
    "aadhar": "आधार",
    "cibil": "सिबिल",
    # Abbreviations spelled letter-by-letter (explicit so we control spacing)
    "fd": "एफ़ डी",
    "kyc": "के वाई सी",
    "otp": "ओ टी पी",
    "ifsc": "आई एफ़ एस सी",
    "neft": "एन ई एफ़ टी",
    "rtgs": "आर टी जी एस",
    "imps": "आई एम पी एस",
    "upi": "यू पी आई",
    "emi": "ई एम आई",
    "rbi": "आर बी आई",
    "nbfc": "एन बी एफ़ सी",
    "cvv": "सी वी वी",
    "atm": "ए टी एम",
    # Common English words that recur in Hinglish replies and break the flow
    "account": "अकाउंट",
    "bank": "बैंक",
    "balance": "बैलेंस",
    "interest": "इंटरेस्ट",
    "deposit": "डिपॉज़िट",
    "branch": "ब्रांच",
    "block": "ब्लॉक",
    "unblock": "अनब्लॉक",
    "transfer": "ट्रांसफर",
    "login": "लॉगिन",
    "app": "ऐप",
    "complete": "कम्पलीट",
    "process": "प्रोसेस",
    "verify": "वेरीफ़ाई",
    "verification": "वेरीफ़िकेशन",
    "document": "डॉक्यूमेंट",
    "customer": "कस्टमर",
    "support": "सपोर्ट",
    "code": "कोड",
    "link": "लिंक",
    "number": "नंबर",
    "mobile": "मोबाइल",
    "online": "ऑनलाइन",
    "email": "ईमेल",
    "message": "मैसेज",
    "call": "कॉल",
    # Agent / person names that recur in the script (proper nouns we DO want
    # spoken in Hindi for flow — extend per campaign).
    "karan": "करण",
    "rahul": "राहुल",
    "anjali": "अंजली",
    "blossom": "ब्लॉसम",
}

# Sort keys once, longest first, so phrases win over their component words.
_GLOSSARY_KEYS = sorted(GLOSSARY.keys(), key=len, reverse=True)

# ASCII-word boundary: a token bounded by non-letters (Devanagari / space /
# punctuation count as boundaries, so "(FD)" and "बैंक" both work) but never
# cuts inside a longer Latin word.
_BOUNDARY_L = r"(?<![A-Za-z])"
_BOUNDARY_R = r"(?![A-Za-z])"

_GLOSSARY_PATTERNS = [
    (re.compile(_BOUNDARY_L + re.escape(k) + _BOUNDARY_R, re.IGNORECASE), GLOSSARY[k])
    for k in _GLOSSARY_KEYS
]

# Remaining ALL-CAPS run of 2-6 letters → spell out (RBI, NEFT, etc.).
_ABBR = re.compile(_BOUNDARY_L + r"[A-Z]{2,6}" + _BOUNDARY_R)

# Markdown markup the LLM sometimes emits (from RAG answers) that must never
# reach the TTS socket — Sarvam vocalises stray asterisks/backticks.
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")          # [text](url) -> text
_MD_IMG = re.compile(r"!\[([^\]]*)\]\([^)]+\)")           # ![alt](url) -> alt
_MD_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3}|`+)")     # ** __ * _ ` markers
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)  # ### heading
_MD_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)    # > quote
_MD_BULLET = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)    # - bullet
_MULTISPACE = re.compile(r"[ \t]{2,}")


def strip_markdown(text: str) -> str:
    """Remove markdown markup so the TTS engine never speaks raw ``**`` / ``#``.

    Keeps the visible text (link labels, bullet contents); drops only the
    syntax characters. Safe on plain text (no-op).
    """
    if not text:
        return text
    out = _MD_IMG.sub(r"\1", text)
    out = _MD_LINK.sub(r"\1", out)
    out = _MD_HEADING.sub("", out)
    out = _MD_BLOCKQUOTE.sub("", out)
    out = _MD_BULLET.sub("", out)
    out = _MD_BOLD_ITALIC.sub("", out)
    out = _MULTISPACE.sub(" ", out)
    return out


def normalize_for_hindi_tts(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (normalized_text, replacements).

    ``replacements`` is a list of (original_token, devanagari) pairs for logging.
    Pure and side-effect-free; safe to call on every utterance.
    """
    if not text:
        return text, []

    replacements: list[tuple[str, str]] = []
    # 0) Strip markdown markup first so ``**`` / ``#`` never reach the socket.
    out = strip_markdown(text)

    # 1) Glossary phrases (longest-first via pre-sorted patterns).
    for pattern, dev in _GLOSSARY_PATTERNS:
        def _sub(m: re.Match[str], _dev=dev) -> str:
            replacements.append((m.group(0), _dev))
            return _dev

        out = pattern.sub(_sub, out)

    # 2) Any leftover ALL-CAPS abbreviation → Devanagari letter-names.
    def _spell(m: re.Match[str]) -> str:
        tok = m.group(0)
        spelled = " ".join(_LETTER.get(c, c) for c in tok)
        replacements.append((tok, spelled))
        return spelled

    out = _ABBR.sub(_spell, out)

    return out, replacements
