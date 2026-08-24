"""The remedy is chosen by rule, before anyone picks up.

"Here is a payment link" is only the right answer to some failures. The tests
that matter here are the ones where a naive agent would give advice that
cannot work: re-entering an expired card, or demanding money from an account
that had none this morning.
"""

import pytest

from app.voice.routes import suggested_route


def test_an_expired_card_is_not_sent_back_to_the_same_card():
    """Retrying a dead card fails again; the link has to offer a way out."""
    route = suggested_route("Your card has expired.")
    assert "different card or UPI" in route


def test_no_balance_is_treated_as_timing_not_refusal():
    """The recovery here is a date, not pressure -- someone paid on the wrong
    day of the month, which is the most recoverable failure there is."""
    route = suggested_route("Your card has insufficient funds.")
    assert "which day suits them" in route


def test_a_bank_side_failure_does_not_blame_the_customer():
    route = suggested_route("The bank could not process the request.")
    assert "not theirs" in route
    assert "do not imply they did anything wrong" in route


def test_a_dead_mandate_mentions_that_it_will_recur():
    """Collecting once without re-authorising just books the same call again."""
    route = suggested_route("emandate revoked by customer")
    assert "auto-pay will need setting up again" in route


def test_an_unknown_failure_does_not_invent_a_diagnosis():
    """An agent guessing why a bank declined someone is worse than one that
    simply offers to take payment."""
    route = suggested_route("ERR_PSP_7731_UPSTREAM")
    assert "Do not speculate" in route


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_a_missing_reason_still_routes_somewhere(missing):
    assert "payment link" in suggested_route(missing)


# --- the halt signal is additive ----------------------------------------


def test_halting_raises_urgency_without_changing_the_remedy():
    """A halted subscription does not make an expired card a different problem."""
    plain = suggested_route("Your card has expired.")
    halted = suggested_route("Your card has expired.", halted=True)

    assert halted.startswith(plain)
    assert "only way it clears" in halted


def test_an_active_subscription_gets_no_urgency_language():
    assert "only way it clears" not in suggested_route("Your card has expired.")


def test_every_route_is_an_instruction_not_speech():
    """Routes are paraphrased by the agent in the call's language, unlike
    reasons.py which is read aloud and therefore stored in Devanagari."""
    for reason in ["expired", "insufficient funds", "gateway error", "limit exceeded", None]:
        route = suggested_route(reason)
        assert not any("ऀ" <= ch <= "ॿ" for ch in route), route
