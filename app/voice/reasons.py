"""Why the payment failed, in the language the call is happening in.

Razorpay returns failure descriptions in English ("Your card has expired."),
and they land in ``RecoveryCase.failure_reason`` verbatim. Handing that string
to a Devanagari conversation puts a Latin sentence inside a Hindi one -- the
same script flip that ``spoken.py`` exists to prevent, and the voice reads it
in an English accent mid-sentence.

Mapped deterministically rather than translated by the model. Two reasons:
the model would paraphrase differently on every call, and this is the sentence
where we tell someone why their money did not move -- it should be one agreed
wording we can point at, not whatever the LLM produced that time.

Unmapped reasons fall back to a neutral phrase instead of leaking the English
through. Saying "the bank declined it" is honest; reading a raw gateway string
at a customer is not useful to them anyway.
"""

import re

#: Matched against the lowercased English reason. Ordered: first hit wins, so
#: the more specific patterns come first.
_PATTERNS: tuple[tuple[str, dict[str, str]], ...] = (
    (
        r"expir",
        {
            "hi": "आपका कार्ड एक्सपायर हो गया है",
            "en": "your card has expired",
        },
    ),
    (
        r"insufficient|low balance|not enough",
        {
            "hi": "खाते में पर्याप्त बैलेंस नहीं था",
            "en": "there were insufficient funds",
        },
    ),
    (
        r"limit|exceed",
        {
            "hi": "कार्ड की लिमिट पार हो गई थी",
            "en": "the card limit was exceeded",
        },
    ),
    (
        r"declin|refus|do not honour|do not honor",
        {
            "hi": "बैंक ने पेमेंट अस्वीकार कर दिया",
            "en": "the bank declined the payment",
        },
    ),
    (
        r"mandate|autopay|emandate|subscription.*cancel",
        {
            "hi": "ऑटो-पे मैंडेट अब सक्रिय नहीं है",
            "en": "the auto-pay mandate is no longer active",
        },
    ),
    (
        r"gateway|timeout|timed out|network|technical|could not process|unable to process",
        {
            "hi": "बैंक की तरफ़ से तकनीकी दिक्कत आ गई थी",
            "en": "there was a technical problem at the bank",
        },
    ),
    (
        r"invalid|incorrect|wrong",
        {
            "hi": "कार्ड की जानकारी में कुछ गड़बड़ थी",
            "en": "the card details did not go through",
        },
    ),
)

_FALLBACK = {
    "hi": "बैंक ने पेमेंट पूरा नहीं किया",
    "en": "the bank did not complete the payment",
}


def spoken_reason(failure_reason: str | None, language: str = "hi") -> str:
    """The failure, phrased for the customer in their language.

    Hinglish takes the Hindi wording: it is a register, not a script, and the
    voice rendering it is pinned to Hindi -- the same rule ``spoken.py`` follows
    for amounts.

    Trailing punctuation is stripped because the prompt supplies its own; the
    raw Razorpay strings end in a full stop and produced "expired.." otherwise.
    """
    key = "en" if language == "en" else "hi"
    text = (failure_reason or "").strip().lower()

    if text:
        for pattern, phrasing in _PATTERNS:
            if re.search(pattern, text):
                return phrasing[key]

    return _FALLBACK[key]


__all__ = ["spoken_reason"]
