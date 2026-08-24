"""Spoken amount rendering. Pure unit tests."""

import pytest

from app.voice.spoken import spoken_amount


@pytest.mark.parametrize(
    "paise,expected",
    [
        (49_900, "499 रुपये"),
        (9_900, "99 रुपये"),
        (100, "1 रुपये"),
        (0, "0 रुपये"),
    ],
)
def test_small_amounts_stay_a_bare_numeral(paise, expected):
    """Cartesia reads a bare numeral correctly in Hindi; only magnitude and the
    Latin currency token break it."""
    assert spoken_amount(paise) == expected


@pytest.mark.parametrize(
    "paise,expected",
    [
        (500_000_00, "5 लाख रुपये"),
        (150_000_00, "1.5 लाख रुपये"),
        (100_000_00, "1 लाख रुपये"),
        (5_000_00, "5 हज़ार रुपये"),
        (2_00_00_000_00, "2 करोड़ रुपये"),
    ],
)
def test_clean_magnitudes_collapse_to_indian_units(paise, expected):
    """A digit crawl is what we are avoiding: '5 लाख' not '500000'."""
    assert spoken_amount(paise) == expected


@pytest.mark.parametrize("paise", [5_43_210_00, 1_23_456_00])
def test_messy_amounts_are_left_alone(paise):
    """Better a plain numeral than an unnatural '5.4321 लाख'."""
    spoken = spoken_amount(paise)
    assert "लाख" not in spoken
    assert spoken.endswith("रुपये")


def test_no_latin_currency_token_in_hindi():
    """'Rs' is a Latin token dropped into a Devanagari sentence -- exactly the
    script flip that makes Hindi voices stumble."""
    spoken = spoken_amount(49_900, "hi")
    assert "Rs" not in spoken and "₹" not in spoken


@pytest.mark.parametrize(
    "language,expected",
    [
        ("hi", "5 लाख रुपये"),
        # Hinglish is a register, not a script: the same Devanagari words as
        # Hindi, because the voice saying them is pinned to language="hi".
        ("hinglish", "5 लाख रुपये"),
        ("en", "5 lakh rupees"),
    ],
)
def test_unit_words_follow_the_language(language, expected):
    assert spoken_amount(500_000_00, language) == expected


def test_unknown_language_falls_back_to_hindi():
    assert spoken_amount(49_900, "ta") == "499 रुपये"
    assert spoken_amount(49_900, "") == "499 रुपये"


def test_paise_remainder_rounds_up_rather_than_being_spoken():
    """Reading paise aloud is noise; the payment link shows the exact figure."""
    assert spoken_amount(49_950) == "500 रुपये"


def test_negative_amounts_do_not_produce_nonsense():
    assert spoken_amount(-100) == "0 रुपये"
