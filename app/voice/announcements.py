"""Speech on the line that did not come from a person.

The carrier plays its own announcement when a call is recorded, and it is
audible on the customer's leg. Sarvam transcribes it like anything else, so it
arrives as a user turn indistinguishable from the customer speaking.

On voice_call 31 that announcement -- transliterated into Devanagari as
"दिस कॉल इज नाउ बीइंग रिकॉर्डेड।" -- satisfied the identity gate. The model read
it as the customer confirming who they were, moved to `explain`, and stated the
amount before the person had said a single word. Identity confirmation is a
compliance boundary, not a formality: it is what stands between someone's
billing detail and whoever happened to answer the phone.

Matching is deliberately narrow. A transcript is dropped only when it carries
both halves of the announcement -- something meaning "call" and something
meaning "recorded" -- because a customer who genuinely says one of those words
must still be heard. Both scripts are matched: the announcement is English, but
the STT is pinned to Hindi and transliterates it.
"""

import re

#: The two halves, in both the original English and the Devanagari
#: transliteration Sarvam produces on a Hindi-pinned model.
_CALL = re.compile(r"(\bcall\b|कॉल|काल)", re.IGNORECASE | re.UNICODE)
_RECORDED = re.compile(
    r"(\brecord(ed|ing)?\b|रिकॉर्ड|रेकॉर्ड|रिकार्ड)", re.IGNORECASE | re.UNICODE
)

#: Phrasings that are announcements on their own, without needing both halves.
#: "for quality and training purposes" is never something a customer says to a
#: dunning agent.
_UNMISTAKABLE = re.compile(
    r"(quality and training|training purposes|monitored or recorded|"
    r"मॉनिटर किया जा|गुणवत्ता के लिए)",
    re.IGNORECASE | re.UNICODE,
)


def is_carrier_announcement(text: str | None) -> bool:
    """Is this the network talking rather than the customer?

    False for anything empty or uncertain. The cost of a false positive is a
    customer's words being discarded, which is worse than the announcement
    reaching the model -- so this errs towards letting speech through.
    """
    if not text or not text.strip():
        return False

    if _UNMISTAKABLE.search(text):
        return True

    return bool(_CALL.search(text) and _RECORDED.search(text))
