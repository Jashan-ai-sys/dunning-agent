"""Amounts on their way into the voice.

Two different jobs, easily confused:

``spoken.py`` renders an amount we already know, from our own database, before
the model ever sees it. This module covers the other half -- what happens when
the model writes an amount *itself*, mid-sentence, in whatever form it likes.
It does that constantly: told "2499 रुपये", it will still emit "₹2499".

The case that matters is the unglamorous one. Our amounts come out of a
subscriptions table, so they are rarely round lakhs -- ₹2,499 and ₹499 are the
norm, and those do not collapse to a magnitude. Before this, the collapse path
was the only path, and a non-collapsible amount was handed to the voice with
its currency glyph still attached.
"""

import pytest

from app.voice.pipecat_agent import normalize_for_speech


@pytest.mark.parametrize(
    "written,spoken",
    [
        ("₹2499", "2499 रुपये"),
        ("₹2,499", "2499 रुपये"),
        ("₹499", "499 रुपये"),
        ("Rs 2499", "2499 रुपये"),
        ("INR 1500", "1500 रुपये"),
    ],
)
def test_a_currency_glyph_never_reaches_the_voice(written, spoken):
    """A Hindi voice has no word for "₹" -- it drops it or reads it in English."""
    assert normalize_for_speech(written, "hi") == spoken


def test_a_clean_magnitude_still_collapses():
    """The behaviour this module already had, kept: 500000 read as digits is a
    digit crawl."""
    assert normalize_for_speech("₹500000", "hi") == "5 लाख"


def test_an_amount_inside_a_sentence_is_rewritten_in_place():
    assert normalize_for_speech(
        "आपकी सब्सक्रिप्शन का ₹2499 का पेमेंट नहीं हो पाया", "hi"
    ) == "आपकी सब्सक्रिप्शन का 2499 रुपये का पेमेंट नहीं हो पाया"


# --- what must NOT be touched -------------------------------------------


def test_an_otp_is_left_alone():
    """The single rule holding this together: no currency marker, no rewrite.
    An OTP spoken as "four lakh eighty-two thousand" would be unusable."""
    assert "482913" in normalize_for_speech("आपका OTP 482913 है", "hi")


def test_a_phone_number_is_left_alone():
    assert "9876543210" in normalize_for_speech("मेरा नंबर 9876543210 है", "hi")


def test_english_calls_get_english_units():
    assert normalize_for_speech("₹2499", "en") == "2499 rupees"
