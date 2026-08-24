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
    "halt_note": "The subscription is still active.",
    "suggested_route": "Offer the payment link.",
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
    assert w.node.id == "reason_inquiry"
    w.transition("reason_given")
    assert w.node.id == "ask_intent"
    w.transition("pay_now")

    assert w.finished
    assert w.intent is CallIntent.RETRY_NOW
    assert w.path == ("greet", "explain", "reason_inquiry", "ask_intent", "pay_now")


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
    w.transition("reason_given")
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
    # Asserted on the stage, not the whole prompt: the preamble now explains the
    # transition tool in general terms, but a finished call must not be offered
    # a move out of its terminal node.
    assert "transition" not in w.stage_instructions()


def test_transition_speech_is_rendered_when_present():
    w = walker()
    w.transition("identity_confirmed")
    w.transition("acknowledged")
    w.transition("reason_given")
    speech = w.transition_speech("pay_now")
    assert speech and "लिंक" in speech


def test_transition_speech_is_none_when_the_edge_has_no_line():
    w = walker()
    assert w.transition_speech("identity_confirmed") is None


# --- observation capture (training data for offline prompt optimisation) ---


def test_accepted_transition_is_recorded_with_its_utterance():
    w = walker()
    w.transition("identity_confirmed", utterance="haan ji, Asha bol rahi hoon")

    assert len(w.observations) == 1
    obs = w.observations[0]
    assert obs.node_id == "greet"          # the node it was decided AT
    assert obs.label == "identity_confirmed"
    assert obs.accepted is True
    assert obs.utterance == "haan ji, Asha bol rahi hoon"
    assert obs.rejection is None


def test_rejected_label_is_recorded_as_a_hard_negative():
    """A label the model reached for and could not have is worth more than an
    example we would have thought to write."""
    w = walker()
    with pytest.raises(InvalidTransition):
        w.transition("pay_now", utterance="kuch bhi")

    assert len(w.observations) == 1
    obs = w.observations[0]
    assert obs.accepted is False
    assert obs.label == "pay_now"
    assert "not a valid move" in obs.rejection


def test_observations_survive_a_full_conversation_in_order():
    w = walker()
    w.transition("identity_confirmed", utterance="haan")
    w.transition("acknowledged", utterance="achha, ab kya karun")
    w.transition("reason_given", utterance="pata nahi, balance kam tha shayad")
    w.transition("pay_now", utterance="link bhej dijiye")

    assert [o.node_id for o in w.observations] == [
        "greet",
        "explain",
        "reason_inquiry",
        "ask_intent",
    ]
    assert [o.label for o in w.observations] == [
        "identity_confirmed",
        "acknowledged",
        "reason_given",
        "pay_now",
    ]
    assert all(o.accepted for o in w.observations)


def test_utterance_is_optional_so_traversal_never_depends_on_it():
    w = walker()
    w.transition("identity_confirmed")
    assert w.observations[0].utterance is None
    assert w.node.id == "explain"


def test_observations_serialise_for_storage():
    w = walker()
    w.transition("identity_confirmed", utterance="haan")
    rows = w.observations_as_dicts()

    assert rows == [
        {
            "node_id": "greet",
            "label": "identity_confirmed",
            "accepted": True,
            "utterance": "haan",
            "rejection": None,
        }
    ]


def test_transition_after_the_call_ends_is_not_recorded():
    """Nothing to learn from a move the graph never offered."""
    w = walker()
    w.transition("not_the_customer", utterance="galat number")
    before = len(w.observations)
    with pytest.raises(InvalidTransition, match="already ended"):
        w.transition("identity_confirmed", utterance="hello")
    assert len(w.observations) == before


# --- reason_inquiry and the financial-difficulty branch -----------------


def test_reason_is_asked_before_options_are_offered():
    """Offering payment options before asking why reads as a script."""
    w = walker()
    w.transition("identity_confirmed")
    w.transition("acknowledged")
    assert w.node.id == "reason_inquiry"
    assert "pay_now" not in w.instructions()


@pytest.mark.parametrize("start", ["reason_inquiry", "ask_intent"])
def test_financial_difficulty_is_not_treated_as_a_refusal(start):
    """"I cannot afford it" is a later, not a no. Routing it to `declined`
    would close the case and stop contact on someone who never refused."""
    w = walker()
    w.transition("identity_confirmed")
    w.transition("acknowledged")
    if start == "ask_intent":
        w.transition("reason_given")

    w.transition("financial_difficulty", utterance="abhi paise nahi hain")

    assert w.intent is CallIntent.RETRY_LATER
    assert w.intent is not CallIntent.DECLINED


def test_dispute_is_reachable_from_reason_inquiry():
    w = walker()
    w.transition("identity_confirmed")
    w.transition("acknowledged")
    w.transition("disputes_charge", utterance="maine to pay kar diya tha")
    assert w.intent is CallIntent.DISPUTE


def test_reason_inquiry_asks_once_and_listens():
    """A stage that lectures is the failure mode; the prompt says one question."""
    w = walker()
    w.transition("identity_confirmed")
    w.transition("acknowledged")
    text = w.instructions()
    assert "one question" in text
    assert "do not ask twice" in text
