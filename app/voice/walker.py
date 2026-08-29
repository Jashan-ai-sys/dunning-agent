"""Pure traversal over the conversation graph.

Deliberately separate from the LiveKit binding. Everything that decides *what
happens* to the call lives here and is unit-tested; ``agent.py`` only carries
audio in and out. A bug in graph traversal is the kind that misroutes a real
customer, so it should not need a telephony provider to catch.

The model never picks a node directly -- it picks an edge *label*, which is
validated against the current node. An unknown or out-of-context label is
rejected rather than followed.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

from app.voice.graph import ConversationGraph, Edge, GraphError, Node
from app.voice.intents import CallIntent


class InvalidTransition(GraphError):
    """The model named an edge that does not exist on the current node."""


@dataclass(frozen=True)
class Observation:
    """One labelled decision: what the customer said, and which edge was taken.

    This is exactly the shape a DSPy signature needs -- input is the node plus
    the customer's utterance, output is the label. Collecting it during real
    calls turns prompt optimisation (GEPA/MIPROv2) into a job that runs against
    our own traffic instead of against invented examples.

    Rejected attempts are kept too. A label the model reached for and could not
    have is a harder negative than anything we would think to write by hand.
    """

    node_id: str
    label: str
    accepted: bool
    utterance: str | None = None
    rejection: str | None = None


@dataclass
class GraphWalker:
    graph: ConversationGraph
    context: dict[str, Any]
    _node: Node = field(init=False)
    _visited: list[str] = field(init=False, default_factory=list)
    _observations: list[Observation] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        missing = self.graph.missing_variables(self.context)
        if missing:
            raise GraphError(f"missing call context: {', '.join(sorted(missing))}")
        self._node = self.graph.start
        self._visited = [self._node.id]

    # -- state ----------------------------------------------------------

    @property
    def node(self) -> Node:
        return self._node

    @property
    def observations(self) -> tuple[Observation, ...]:
        """Labelled turns from this call, in order."""
        return tuple(self._observations)

    def observations_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(o) for o in self._observations]

    @property
    def path(self) -> tuple[str, ...]:
        """Nodes visited, in order. Recorded on the call for the audit trail."""
        return tuple(self._visited)

    @property
    def finished(self) -> bool:
        return self._node.is_terminal

    @property
    def intent(self) -> CallIntent | None:
        """Only set once a terminal node is reached."""
        return self._node.intent if self._node.is_terminal else None

    @property
    def options(self) -> tuple[Edge, ...]:
        return self._node.edges

    # -- movement -------------------------------------------------------

    def transition(self, label: str, utterance: str | None = None) -> Node:
        """Follow the named edge. Raises if it is not valid here.

        ``utterance`` is what the customer said to prompt this move. It is only
        recorded, never used to decide -- the label alone drives traversal.
        """
        origin = self._node.id
        if self.finished:
            raise InvalidTransition(f"call already ended at '{origin}'")

        for edge in self._node.edges:
            if edge.label == label:
                self._observations.append(
                    Observation(node_id=origin, label=label, accepted=True, utterance=utterance)
                )
                self._node = self.graph.node(edge.to)
                self._visited.append(self._node.id)
                return self._node

        valid = ", ".join(e.label for e in self._node.edges)
        message = f"'{label}' is not a valid move from '{origin}'; expected one of: {valid}"
        self._observations.append(
            Observation(
                node_id=origin,
                label=label,
                accepted=False,
                utterance=utterance,
                rejection=message,
            )
        )
        raise InvalidTransition(message)

    # -- prompting ------------------------------------------------------

    def preamble(self) -> str:
        """The half of the prompt that never changes during the call.

        Split out so a provider can cache it. Everything here is identical on
        every turn; put anything stage-dependent in here and the cached prefix
        breaks on the next transition.
        """
        return self.graph.render_preamble(self.context)

    def stage_instructions(self, *, override_prompt: str | None = None) -> str:
        """The half that changes: this node's prompt and its available moves.

        Append-only by design -- the caller adds this to the conversation
        rather than rewriting what came before, so the prefix keeps growing
        instead of shifting.

        ``override_prompt`` replaces the node's own text while keeping its
        edges. It exists for one case: when we have spoken a line the node
        would otherwise have asked the model to produce. Negating the original
        instead of replacing it does not work -- "do not greet again" directly
        above "Open the call. Greet them" is a contradiction, and the model
        resolves it by greeting. That happened on a live call.
        """
        rendered = (
            override_prompt
            if override_prompt is not None
            else self.graph.render(self._node.id, self.context)
        )
        if self.finished:
            return f"{rendered}\n\nThis is the end of the call. Do not ask any further questions."

        return (
            f"{rendered}\n\n"
            "Call the `transition` tool as soon as one of the labels below "
            "matches what the customer just said. They describe this moment in "
            "the call, not their final decision -- staying here to keep talking "
            "is how a call ends with nothing recorded. If none of them matches "
            "yet, ask one short clarifying question instead.\n\n"
            "Say your reply in the SAME response as the tool call -- speak "
            "first, then call the tool. A tool call on its own is silence on "
            "the line while you are asked again, and the customer is waiting.\n\n"
            f"Available labels:\n{self.moves()}"
        )

    def moves(self) -> str:
        """The labels that are legal right now, and what earns each one.

        Split out because a rejected transition needs the moves without the
        node's prompt -- re-sending the prompt would have the model deliver its
        opening line a second time.
        """
        return "\n".join(f"- {edge.label}: {edge.condition}" for edge in self._node.edges)

    def instructions(self) -> str:
        """Preamble and stage as one block, for callers that want a single prompt.

        The LiveKit path hands a fresh Agent its whole instruction set on every
        handoff, so it wants this. The Pipecat path splits the two halves.
        """
        return f"{self.preamble()}\n\n{self.stage_instructions()}"

    def transition_speech(self, label: str) -> str | None:
        """Optional line spoken while moving, rendered with call context."""
        for edge in self._node.edges:
            if edge.label == label and edge.speech:
                return edge.speech.format(**self.context)
        return None
