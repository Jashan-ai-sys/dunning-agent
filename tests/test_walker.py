"""Graph traversal. Pure state machine, no LiveKit involved.

This is where a routing bug would send a real customer down the wrong branch, so
it is tested without needing a telephony provider.
"""

import pytest

from app.voice.flow import DUNNING_FLOW
from app.voice.graph import GraphError
from app.voice.intents import CallIntent
from app.voice.walker import GraphWalker, InvalidTransition

CONTEXT = {
    "company_name": "Acme",
    "customer_name": "Asha",
    "amount_spoken": "499 रुपये",
    "failure_reason": "insufficient funds",
    "language_hint": "in Hinglish",
}


def walker(**overrides) -> GraphWalker:
    return GraphWalker(DUNNING_FLOW, {**CONTEXT, **overrides})


def test_walker_starts_at_the_greeting():
    w = walker()
    assert w.node.id == "greet"
    assert not w.finished
    assert w.intent is None


def test_missing_context_is_caught_before_the_call_starts():
    """Better to fail at dispatch than to read '{customer_name}' down a phone."""
    with pytest.raises(GraphError, match="missing call context"):
        GraphWalker(DUNNING_FLOW, {"company_name": "Acme"})


def test_happy_path_reaches_pay_now():
    w = walker()
    w.transition("identity_confirmed")
    assert w.node.id == "explain"
    w.transition("acknowledged")
    assert w.node.id == "ask_intent"
    w.transition("pay_now")

    assert w.finished
    assert w.intent is CallIntent.RETRY_NOW
    assert w.path == ("greet", "explain", "ask_intent", "pay_now")


def test_wrong_number_exits_at_the_greeting():
    """Must be reachable before any billing detail is spoken."""
    w = walker()
    w.transition("not_the_customer")

    assert w.finished
    assert w.intent is CallIntent.WRONG_NUMBER
    assert w.path == ("greet", "wrong_number")


def test_dispute_is_reachable_from_two_stages():
    for first in (["identity_confirmed", "disputes_charge"],
                  ["identity_confirmed", "acknowledged", "disputes_charge"]):
        w = walker()
        for label in first:
            w.transition(label)
        assert w.intent is CallIntent.DISPUTE


@pytest.mark.parametrize(
    "label,expected",
    [
        ("pay_now", CallIntent.RETRY_NOW),
        ("pay_later", CallIntent.RETRY_LATER),
        ("declined", CallIntent.DECLINED),
        ("disputes_charge", CallIntent.DISPUTE),
    ],
)
def test_every_branch_from_ask_intent(label, expected):
    w = walker()
    w.transition("identity_confirmed")
    w.transition("acknowledged")
    w.transition(label)
    assert w.intent is expected


def test_unknown_label_is_rejected():
    """The model does not get to invent a destination."""
    w = walker()
    with pytest.raises(InvalidTransition, match="not a valid move"):
        w.transition("hang_up_and_run")


def test_label_valid_elsewhere_is_still_rejected_here():
    """'pay_now' exists in the flow but not from the greeting."""
    w = walker()
    with pytest.raises(InvalidTransition, match="not a valid move"):
        w.transition("pay_now")


def test_error_message_lists_the_valid_labels():
    """The model sees this text and should be able to correct itself."""
    w = walker()
    with pytest.raises(InvalidTransition) as excinfo:
        w.transition("nope")
    assert "identity_confirmed" in str(excinfo.value)
    assert "not_the_customer" in str(excinfo.value)


def test_no_transitions_after_the_call_ends():
    w = walker()
    w.transition("not_the_customer")
    with pytest.raises(InvalidTransition, match="already ended"):
        w.transition("identity_confirmed")


# --- prompting ---------------------------------------------------------


def test_instructions_are_rendered_with_no_placeholders_left():
    w = walker()
    assert "{" not in w.instructions()
    assert "Asha" in w.instructions()


def test_instructions_offer_only_the_current_node_options():
    """The model sees this node's branches, not the whole script."""
    w = walker()
    text = w.instructions()
    assert "identity_confirmed" in text
    assert "not_the_customer" in text
    assert "pay_now" not in text


def test_greeting_instructions_do_not_leak_the_amount():
    """Whoever answers the phone must not hear someone else's billing detail."""
    assert "499" not in walker().instructions()


def test_amount_appears_once_identity_is_confirmed():
    w = walker()
    w.transition("identity_confirmed")
    assert "499" in w.instructions()


def test_terminal_instructions_close_the_call():
    w = walker()
    w.transition("not_the_customer")
    assert "end of the call" in w.instructions()
    assert "transition" not in w.instructions()


def test_transition_speech_is_rendered_when_present():
    w = walker()
    w.transition("identity_confirmed")
    w.transition("acknowledged")
    speech = w.transition_speech("pay_now")
    assert speech and "लिंक" in speech


def test_transition_speech_is_none_when_the_edge_has_no_line():
    w = walker()
    assert w.transition_speech("identity_confirmed") is None
