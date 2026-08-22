"""Pure traversal over the conversation graph.

Deliberately separate from the LiveKit binding. Everything that decides *what
happens* to the call lives here and is unit-tested; ``agent.py`` only carries
audio in and out. A bug in graph traversal is the kind that misroutes a real
customer, so it should not need a telephony provider to catch.

The model never picks a node directly -- it picks an edge *label*, which is
validated against the current node. An unknown or out-of-context label is
rejected rather than followed.
"""

from dataclasses import dataclass, field
from typing import Any

from app.voice.graph import ConversationGraph, Edge, GraphError, Node
from app.voice.intents import CallIntent


class InvalidTransition(GraphError):
    """The model named an edge that does not exist on the current node."""


@dataclass
class GraphWalker:
    graph: ConversationGraph
    context: dict[str, Any]
    _node: Node = field(init=False)
    _visited: list[str] = field(init=False, default_factory=list)

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

    def transition(self, label: str) -> Node:
        """Follow the named edge. Raises if it is not valid here."""
        if self.finished:
            raise InvalidTransition(f"call already ended at '{self._node.id}'")

        for edge in self._node.edges:
            if edge.label == label:
                self._node = self.graph.node(edge.to)
                self._visited.append(self._node.id)
                return self._node

        valid = ", ".join(e.label for e in self._node.edges)
        raise InvalidTransition(
            f"'{label}' is not a valid move from '{self._node.id}'; expected one of: {valid}"
        )

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
