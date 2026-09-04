"""What the agent may and may not say.

The rules are tested here without NeMo, and the NeMo wiring is tested once at
the bottom. That split is deliberate: the rules are the thing that will change
as real calls surface new failures, and they should not need a guardrails
runtime to exercise.
"""

import pytest

from app.voice.guardrails.checks import check_output

# -- credential solicitation ----------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "कृपया अपना ओटीपी बताइए।",
        "आप अपना कार्ड नंबर बता दीजिए।",
        "Please tell me your card number.",
        "Can you share the OTP you just received?",
        "अपना यूपीआई पिन डालिए।",
    ],
)
def test_asking_for_credentials_is_refused(text):
    """A dunning agent has no reason to know any of these, and an agent that
    asks sounds exactly like the fraud call customers are warned about."""
    violation = check_output(text)
    assert violation is not None
    assert violation.rule == "credential_solicitation"


def test_cvv_is_refused_even_without_a_request():
    """Every other term has an innocent reading; this one does not. There is no
    sentence about a customer's CVV that we should be speaking."""
    violation = check_output("The CVV is on the back of the card.")
    assert violation is not None
    assert violation.rule == "credential_solicitation"


@pytest.mark.parametrize(
    "text",
    [
        "आपका कार्ड एक्सपायर हो गया है, इसलिए पेमेंट नहीं हुआ।",
        "Your card expired, so the payment could not go through.",
        "मैं आपको लिंक भेज देता हूँ, वहाँ आप दूसरे कार्ड से पेमेंट कर सकते हैं।",
    ],
)
def test_mentioning_a_card_is_not_asking_for_one(text):
    """The remedy for an expired card has to be sayable. A rule that muted it
    would be worse than no rule -- it would break the one explanation the
    customer needs."""
    assert check_output(text) is None


# -- settlement claims -----------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "आपका पेमेंट हो गया है, धन्यवाद।",
        "हम आपका रिफंड कर देंगे।",
        "मैं यह चार्ज माफ कर दूंगा।",
        "Your payment is successful.",
        "We will waive this charge for you.",
    ],
)
def test_the_agent_may_not_settle_the_debt(text):
    """Only a Razorpay webhook may assert that money moved. A customer told
    their debt is cleared stops worrying about one that is still open."""
    violation = check_output(text)
    assert violation is not None
    assert violation.rule == "unauthorised_settlement_claim"


def test_offering_to_send_a_link_is_still_allowed():
    """Sending a link is a thing the agent genuinely does, via the MCP tool.
    Only *claiming the debt is settled* is out of bounds."""
    assert check_output("मैं आपको पेमेंट लिंक एसएमएस पर भेज देता हूँ।") is None


# -- amounts ---------------------------------------------------------------

def test_an_invented_amount_is_caught():
    violation = check_output("आपके ₹1299 बकाया हैं।", expected_amount_rupees="499.00")
    assert violation is not None
    assert violation.rule == "amount_mismatch"
    assert "1299" in violation.detail


@pytest.mark.parametrize(
    "text",
    ["आपके ₹499 बकाया हैं।", "The amount is Rs. 499.00", "499 रुपये बाकी हैं।"],
)
def test_the_real_amount_passes_in_either_script(text):
    assert check_output(text, expected_amount_rupees="499.00") is None


def test_no_expected_amount_means_the_rule_is_skipped():
    """A guard that does not know the right answer must not invent a wrong one.
    Muting a correct sentence about someone's money is its own failure."""
    assert check_output("आपके ₹1299 बकाया हैं।", expected_amount_rupees=None) is None


def test_spoken_amounts_in_words_are_not_checked():
    """`spoken.py` renders the figure into words and owns that path. This rule
    is for digits, which is the form a hallucinated amount actually takes."""
    assert check_output("चार सौ निन्यानवे रुपये बाकी हैं।", expected_amount_rupees="499.00") is None


# -- ordinary speech -------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "नमस्ते, मैं ब्लॉस्टेम से बात कर रहा हूँ।",
        "क्या आप अभी पेमेंट कर सकते हैं?",
        "कोई बात नहीं, मैं कल फिर कोशिश करूँगा।",
        "",
        "   ",
    ],
)
def test_normal_conversation_is_untouched(text):
    assert check_output(text) is None


# -- the NeMo runtime ------------------------------------------------------

async def test_nemo_output_rail_withholds_and_substitutes():
    """The rail must actually stop the sentence, not merely flag it.

    This is the only test that loads NeMo. It proves the Colang flow reaches
    our action and that `stop` replaces the turn rather than appending to it.
    """
    from app.voice.guardrails.rails import DunningRails

    guard = DunningRails(expected_amount_rupees="499.00")

    clean, violation = await guard.vet("क्या आप अभी पेमेंट कर सकते हैं?")
    assert violation is None
    assert clean == "क्या आप अभी पेमेंट कर सकते हैं?"

    withheld, violation = await guard.vet("अपना ओटीपी बताइए।")
    assert violation is not None
    assert violation.rule == "credential_solicitation"
    assert "ओटीपी" not in withheld
