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


def test_the_stage_prompt_asks_for_speech_and_the_tool_call_together():
    """The batching optimisation is only worth anything if the model is asked
    for it. A tool call with no speech costs a second Vertex round trip --
    measured at +0.565s (1.87x) on turns that transitioned."""
    walker = GraphWalker(DUNNING_FLOW, CONTEXT)
    prompt = walker.stage_instructions()
    assert "SAME response as the tool call" in prompt
    assert "silence on the line" in prompt
    # It must still be a constrained choice, not free rein.
    assert "Available labels:" in prompt


def test_the_stage_prompt_says_a_tool_call_is_not_a_transition():
    """Both stalls we have seen on live calls were the model doing the work of
    an outcome and stopping there.

    At `explain` it kept talking; at `ask_intent` it sent the payment link, said
    so, and never transitioned. Both calls ended `unclear` -- no intent recorded
    for a customer who had agreed to pay.
    """
    walker = GraphWalker(DUNNING_FLOW, CONTEXT)
    assert "not a substitute" in walker.stage_instructions()


def test_the_moves_are_only_the_ones_legal_here():
    """What a rejected transition hands back. It must carry the labels without
    the node's prompt: that opens with the line the model has already spoken."""
    walker = GraphWalker(DUNNING_FLOW, CONTEXT)
    walker.transition("identity_confirmed", utterance="yes")

    moves = walker.moves()

    assert "acknowledged" in moves
    assert "disputes_charge" in moves
    assert "pay_now" not in moves
    assert walker.graph.render("explain", walker.context) not in moves


def test_sending_a_link_is_not_where_the_intent_stage_ends():
    """`ask_intent` tells the model to send the link and wait for the result.
    Satisfying that felt like finishing the step, so it stopped there."""
    walker = GraphWalker(DUNNING_FLOW, CONTEXT)
    for label in ("identity_confirmed", "acknowledged", "reason_given"):
        walker.transition(label, utterance="x")
    assert walker.node.id == "ask_intent"

    prompt = walker.stage_instructions()

    assert "send_payment_link" in prompt
    assert "`transition` with `pay_now`" in prompt


def test_a_terminal_stage_asks_for_no_transition_at_all():
    """Nothing left to batch once the call is over."""
    walker = GraphWalker(DUNNING_FLOW, CONTEXT)
    for label in ("identity_confirmed", "not_the_customer"):
        try:
            walker.transition(label, utterance="x")
        except Exception:
            continue
        break
    while not walker.finished and walker.node.edges:
        walker.transition(walker.node.edges[0].label, utterance="x")
    prompt = walker.stage_instructions()
    assert "SAME response as the tool call" not in prompt
    assert "end of the call" in prompt


def test_both_link_tools_are_offered_at_every_stage():
    """A customer asks a question when they think of it, not when the script
    expects it.

    On a live call the customer asked how to set up the UPI mandate during
    reason_inquiry. send_mandate_link was only mentioned in ask_intent's
    prompt, a stage the call had not reached, so the model had no legal move
    and the conversation stalled.
    """
    walker = GraphWalker(DUNNING_FLOW, CONTEXT)
    preamble = walker.preamble()
    assert "send_mandate_link" in preamble
    assert "send_payment_link" in preamble
    assert "EVERY stage" in preamble


def test_asking_what_to_do_is_a_way_out_of_reason_inquiry():
    """That question had no matching edge, which is what left the model stuck."""
    walker = GraphWalker(DUNNING_FLOW, CONTEXT)
    walker.transition("identity_confirmed", utterance="haan ji")
    walker.transition("acknowledged", utterance="accha")
    assert walker.node.id == "reason_inquiry"
    conditions = " ".join(e.condition for e in walker.node.edges)
    assert "what they should do" in conditions
