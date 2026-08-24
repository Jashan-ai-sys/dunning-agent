"""Why the payment failed, phrased for the person hearing it.

Razorpay writes these in English. The call happens in Hindi. Handing the raw
string to the agent puts a Latin sentence inside a Devanagari one, which is the
same script flip `spoken.py` exists to prevent -- and this is the sentence where
we tell someone why their money did not move, so it should be one agreed
wording rather than whatever the model paraphrased that time.
"""

import pytest

from app.voice.flow import HALT_NOTES, halt_note
from app.voice.reasons import spoken_reason


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Your card has expired.", "आपका कार्ड एक्सपायर हो गया है"),
        ("Your card has insufficient funds.", "खाते में पर्याप्त बैलेंस नहीं था"),
        ("The bank could not process the request.", "बैंक की तरफ़ से तकनीकी दिक्कत आ गई थी"),
        ("payment was declined by the issuing bank", "बैंक ने पेमेंट अस्वीकार कर दिया"),
        ("Card limit exceeded", "कार्ड की लिमिट पार हो गई थी"),
    ],
)
def test_real_razorpay_reasons_come_back_in_devanagari(raw, expected):
    assert spoken_reason(raw, "hi") == expected


def test_hinglish_takes_the_hindi_wording():
    """A register, not a script -- and the voice is pinned to Hindi."""
    assert spoken_reason("Your card has expired.", "hinglish") == spoken_reason(
        "Your card has expired.", "hi"
    )


def test_english_speakers_get_english():
    assert spoken_reason("Your card has expired.", "en") == "your card has expired"


def test_an_unmapped_reason_does_not_leak_english():
    """The failure mode this exists to stop: a raw gateway string read aloud."""
    spoken = spoken_reason("ERR_PSP_7731_UPSTREAM", "hi")
    assert "ERR_PSP" not in spoken
    assert spoken == "बैंक ने पेमेंट पूरा नहीं किया"


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_a_missing_reason_still_says_something(missing):
    assert spoken_reason(missing, "hi") == "बैंक ने पेमेंट पूरा नहीं किया"


def test_no_trailing_full_stop_is_carried_through():
    """The prompt supplies its own punctuation; Razorpay's produced 'expired..'"""
    assert not spoken_reason("Your card has expired.", "hi").endswith(".")


# --- the halt signal ----------------------------------------------------


def test_a_halted_subscription_is_stated_plainly():
    note = halt_note(True)
    assert "stopped retrying" in note
    # It is a fact, not leverage: no deadline the agent cannot actually name.
    assert "cut-off date" in note


def test_an_active_subscription_is_not_dramatised():
    """Telling someone their service is at risk when it is not is the worse
    error of the two, so unknown must land here."""
    assert halt_note(False) == HALT_NOTES[False]
    assert halt_note(None) == HALT_NOTES[False]
    assert "not" in halt_note(None)
