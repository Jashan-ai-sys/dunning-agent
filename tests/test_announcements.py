"""The network's own speech must never become a conversation turn.

Regression for voice_call 31. The carrier's call-recording announcement was
transcribed and fed to the model as a user turn, and it satisfied the identity
gate -- the agent moved to `explain` and stated the amount before the customer
had said a word.

Identity confirmation is a compliance boundary, not a formality: it is what
stands between someone's billing detail and whoever happened to answer.
"""

import pytest

from app.voice.announcements import is_carrier_announcement


@pytest.mark.parametrize(
    "text",
    [
        # The exact transcript from voice_call 31.
        "दिस कॉल इज नाउ बीइंग रिकॉर्डेड।",
        "This call is now being recorded.",
        "this call is being recorded",
        "This call may be monitored or recorded.",
        "Your call is recorded for quality and training purposes.",
    ],
)
def test_carrier_announcements_are_dropped(text):
    assert is_carrier_announcement(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # What the customer actually said, one turn late, on that same call.
        "और ये।",
        "हाँ जी।",
        "तो मैं क्या करूँ?",
        "सर मैं अभी पेमेंट ही कर दूंगा।",
        "यार पेमेंट मेरे ख्याल से। सर्वर डाउन था बैंक का।",
        "",
        "   ",
        None,
    ],
)
def test_customer_speech_is_never_dropped(text):
    assert is_carrier_announcement(text) is False


def test_one_half_of_the_phrase_is_not_enough():
    """Both halves are required. A customer who says one of these words must
    still be heard -- discarding their speech is worse than letting an
    announcement through."""
    assert is_carrier_announcement("मुझे कॉल मत करना।") is False       # "call", no "recorded"
    assert is_carrier_announcement("मेरा रिकॉर्ड ठीक है।") is False     # "record", no "call"
    assert is_carrier_announcement("Please don't call me again.") is False


# -- a question is not a denial ---------------------------------------------

def test_asking_who_is_calling_is_not_a_wrong_number():
    """On voice_call 34 the customer asked "सॉरी, कहाँ से बात कर रहे हो?" and the
    model took `not_the_customer`, ending the call in 15 seconds with a false
    `wrong_number` outcome. Someone checking who they are speaking to has not
    said they are the wrong person."""
    from app.voice.flow import GREET

    edge = next(e for e in GREET.edges if e.label == "not_the_customer")
    assert "QUESTION is not a denial" in edge.condition
    assert "where are you calling from" in edge.condition
    # And it must still catch the case it exists for.
    assert "wrong number" in edge.condition


def test_the_agent_sends_the_link_not_the_customer():
    """On voice_call 35 the customer asked what they had to do and was told
    "आपको पेमेंट के लिए एक नया लिंक भेजना होगा" -- that they had to send a link.
    The direction is backwards and the offer is two stages early."""
    from app.voice.flow import EXPLAIN

    assert "do NOT tell them to send anything" in EXPLAIN.prompt
    assert "You send the link, they receive it" in EXPLAIN.prompt
