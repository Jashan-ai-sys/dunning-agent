"""Conversation graph engine and the dunning flow. Pure unit tests."""

import pytest

from app.constants import CaseStatus
from app.voice.flow import DUNNING_FLOW, language_hint
from app.voice.graph import (
    ConversationGraph,
    Edge,
    GraphError,
    Node,
    NodeKind,
    template_variables,
)
from app.voice.intents import OUTCOMES, CallIntent, outcome_for

CONTEXT = {
    "company_name": "Acme",
    "customer_name": "Asha",
    "amount_spoken": "499 रुपये",
    "failure_reason": "insufficient funds",
    "language_hint": "in Hinglish",
}


def node(node_id="a", kind=NodeKind.AGENT, **kwargs) -> Node:
    kwargs.setdefault("prompt", "say something")
    return Node(id=node_id, kind=kind, **kwargs)


def terminal(node_id="end", intent=CallIntent.DECLINED) -> Node:
    return Node(id=node_id, kind=NodeKind.END, prompt="bye", intent=intent)


def edge(to="end", label="l") -> Edge:
    return Edge(to=to, label=label, condition="always")


# --- engine validation -------------------------------------------------


def test_minimal_valid_graph_builds():
    graph = ConversationGraph(
        nodes=(node("start", NodeKind.START, edges=(edge(),)), terminal())
    )
    assert graph.start.id == "start"


def test_graph_needs_exactly_one_start():
    with pytest.raises(GraphError, match="exactly one start node"):
        ConversationGraph(
            nodes=(
                node("s1", NodeKind.START, edges=(edge(),)),
                node("s2", NodeKind.START, edges=(edge(),)),
                terminal(),
            )
        )


def test_graph_needs_a_terminal_node():
    """Without one, a call could never end."""
    with pytest.raises(GraphError, match="no terminal node"):
        ConversationGraph(nodes=(node("start", NodeKind.START, edges=(edge(to="start"),)),))


def test_edge_to_unknown_node_is_rejected():
    with pytest.raises(GraphError, match="unknown node"):
        ConversationGraph(
            nodes=(node("start", NodeKind.START, edges=(edge(to="nowhere"),)), terminal())
        )


def test_dead_end_node_is_rejected():
    with pytest.raises(GraphError, match="dead end"):
        ConversationGraph(
            nodes=(
                node("start", NodeKind.START, edges=(edge(to="mid"),)),
                node("mid"),
                terminal(),
            )
        )


def test_unreachable_node_is_rejected():
    with pytest.raises(GraphError, match="unreachable"):
        ConversationGraph(
            nodes=(
                node("start", NodeKind.START, edges=(edge(),)),
                terminal(),
                node("orphan", edges=(edge(),)),
            )
        )


def test_terminal_node_must_declare_an_intent():
    with pytest.raises(GraphError, match="must declare an intent"):
        ConversationGraph(
            nodes=(
                node("start", NodeKind.START, edges=(edge(),)),
                Node(id="end", kind=NodeKind.END, prompt="bye"),
            )
        )


def test_terminal_node_may_not_have_outgoing_edges():
    with pytest.raises(GraphError, match="no outgoing edges"):
        ConversationGraph(
            nodes=(
                node("start", NodeKind.START, edges=(edge(),)),
                Node(
                    id="end",
                    kind=NodeKind.END,
                    prompt="bye",
                    intent=CallIntent.DECLINED,
                    edges=(edge(to="start"),),
                ),
            )
        )


def test_duplicate_edge_labels_are_rejected():
    """The model picks a branch by label, so labels must be unambiguous."""
    with pytest.raises(GraphError, match="duplicate edge labels"):
        ConversationGraph(
            nodes=(
                node(
                    "start",
                    NodeKind.START,
                    edges=(edge(to="end", label="x"), edge(to="end2", label="x")),
                ),
                terminal("end"),
                terminal("end2"),
            )
        )


def test_self_loop_is_rejected():
    with pytest.raises(GraphError, match="self-loop"):
        ConversationGraph(
            nodes=(
                node("start", NodeKind.START, edges=(edge(to="start"), edge(to="end", label="z"))),
                terminal(),
            )
        )


# --- templating --------------------------------------------------------


def test_template_variables_are_discovered():
    assert template_variables("hi {name}, you owe {amount}") == {"name", "amount"}


def test_render_fills_the_prompt():
    graph = ConversationGraph(
        nodes=(
            node("start", NodeKind.START, prompt="hello {customer_name}", edges=(edge(),)),
            terminal(),
        )
    )
    assert graph.render("start", {"customer_name": "Asha"}) == "hello Asha"


def test_render_refuses_to_silently_drop_a_variable():
    """A half-rendered prompt would be read out to a real customer."""
    graph = ConversationGraph(
        nodes=(
            node("start", NodeKind.START, prompt="hello {customer_name}", edges=(edge(),)),
            terminal(),
        )
    )
    with pytest.raises(GraphError, match="missing template variables"):
        graph.render("start", {})


# --- the dunning flow --------------------------------------------------


def test_dunning_flow_is_valid():
    """It validates at import, so this asserts the shape stayed sane."""
    assert DUNNING_FLOW.start.id == "greet"


def test_every_terminal_maps_to_a_known_outcome():
    for node_ in DUNNING_FLOW.nodes:
        if node_.is_terminal:
            assert node_.intent in OUTCOMES


def test_flow_renders_with_the_documented_context():
    assert DUNNING_FLOW.missing_variables(CONTEXT) == set()
    for node_ in DUNNING_FLOW.nodes:
        rendered = DUNNING_FLOW.render(node_.id, CONTEXT)
        assert "{" not in rendered


def test_identity_is_confirmed_before_billing_details_are_revealed():
    """The greet node must not leak the amount to whoever picks up."""
    greet = DUNNING_FLOW.node("greet")
    assert "amount_spoken" not in template_variables(greet.prompt)
    assert {e.label for e in greet.edges} == {"identity_confirmed", "not_the_customer"}


def test_every_stage_can_reach_a_wrong_number_or_dispute_exit():
    """A customer must always have an exit that stops the calls."""
    labels = {
        node_.id: {e.label for e in node_.edges}
        for node_ in DUNNING_FLOW.nodes
        if not node_.is_terminal
    }
    assert "not_the_customer" in labels["greet"]
    assert "disputes_charge" in labels["explain"]
    assert "declined" in labels["ask_intent"]


# --- intent outcomes ---------------------------------------------------


def test_declining_stops_further_contact():
    outcome = outcome_for(CallIntent.DECLINED)
    assert outcome.status == CaseStatus.DECLINED
    assert outcome.suppress_contact


def test_wrong_number_stops_immediately():
    outcome = outcome_for(CallIntent.WRONG_NUMBER)
    assert outcome.status == CaseStatus.STOPPED
    assert outcome.suppress_contact


def test_dispute_escalates_to_a_human():
    outcome = outcome_for(CallIntent.DISPUTE)
    assert outcome.needs_human
    assert outcome.suppress_contact


def test_retry_now_is_the_only_intent_that_sends_a_link():
    senders = [i for i, o in OUTCOMES.items() if o.send_payment_link]
    assert senders == [CallIntent.RETRY_NOW]


def test_retry_now_does_not_close_the_case():
    """The case closes when money arrives, not when someone promises it will."""
    assert outcome_for(CallIntent.RETRY_NOW).status is None


@pytest.mark.parametrize("intent", [CallIntent.NO_ANSWER, CallIntent.UNCLEAR])
def test_inconclusive_calls_leave_the_case_open(intent):
    outcome = outcome_for(intent)
    assert outcome.status is None
    assert not outcome.suppress_contact


def test_every_intent_has_an_outcome():
    assert set(OUTCOMES) == set(CallIntent)


@pytest.mark.parametrize(
    "preference,expected_fragment",
    [("hinglish", "Hinglish"), ("hi", "Hindi"), ("en", "English"), (None, "Hindi")],
)
def test_language_hint_follows_the_customer_preference(preference, expected_fragment):
    assert expected_fragment in language_hint(preference)


def test_every_node_carries_the_language_mirroring_rule():
    """Sarvam runs in transcribe mode, so the model sees the customer's actual
    language. It must answer in that language rather than defaulting."""
    for node_ in DUNNING_FLOW.nodes:
        prompt = DUNNING_FLOW.render(node_.id, CONTEXT)
        assert "same language" in prompt
        assert "same script" in prompt


def test_language_rule_is_marked_as_overriding():
    """It has to win against the per-customer preferred-language hint."""
    prompt = DUNNING_FLOW.render("greet", CONTEXT)
    assert "overrides every other instruction" in prompt
