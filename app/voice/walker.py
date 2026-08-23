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

    def instructions(self) -> str:
        """Rendered node prompt plus the menu of moves available right now.

        The menu is regenerated per node so the model only ever sees the
        branches that apply, instead of the whole script.
        """
        rendered = self.graph.render(self._node.id, self.context)
        if self.finished:
            return f"{rendered}\n\nThis is the end of the call. Do not ask any further questions."

        menu = "\n".join(
            f"- {edge.label}: {edge.condition}" for edge in self._node.edges
        )
        return (
            f"{rendered}\n\n"
            "When, and only when, the customer's position is clear, call the "
            "`transition` tool with the matching label. Do not guess: if it is "
            "still ambiguous, ask one short clarifying question instead.\n\n"
            f"Available labels:\n{menu}"
        )

    def transition_speech(self, label: str) -> str | None:
        """Optional line spoken while moving, rendered with call context."""
        for edge in self._node.edges:
            if edge.label == label and edge.speech:
                return edge.speech.format(**self.context)
        return None
