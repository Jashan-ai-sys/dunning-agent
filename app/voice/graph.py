"""A node/edge conversation graph, in the Dograh style.

The call is a small state machine, not one long prompt. Each node carries the
instructions for one stage of the conversation; each edge carries a
natural-language condition the LLM evaluates to decide where to go next. That
buys three things a monolithic prompt cannot:

* the flow is inspectable and diffable without reading a wall of prose,
* the model chooses between a handful of named transitions rather than being
  trusted to hold the whole script in its head,
* every terminal state maps to exactly one recorded outcome, so a call can
  never end in an intent the orchestrator does not know how to act on.

The graph is validated at import time, so a malformed flow fails at startup
rather than halfway through a call to a customer.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from string import Formatter
from typing import Any

from app.voice.intents import CallIntent


class NodeKind(StrEnum):
    START = "start"
    AGENT = "agent"
    END = "end"


class GraphError(ValueError):
    """Raised when a flow definition is structurally invalid."""


@dataclass(frozen=True)
class Edge:
    """A possible transition. ``condition`` is evaluated by the LLM."""

    to: str
    condition: str
    label: str
    #: Optional line spoken while transitioning, for a natural handoff.
    speech: str | None = None


@dataclass(frozen=True)
class Node:
    id: str
    kind: NodeKind
    prompt: str
    edges: tuple[Edge, ...] = ()
    #: Facts to pull out of the customer's replies at this stage.
    extracts: tuple[str, ...] = ()
    #: Only on END nodes: the outcome this branch resolves to.
    intent: CallIntent | None = None

    @property
    def is_terminal(self) -> bool:
        return self.kind is NodeKind.END


@dataclass(frozen=True)
class ConversationGraph:
    nodes: tuple[Node, ...]
    _by_id: dict[str, Node] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_id", {node.id: node for node in self.nodes})
        self.validate()

    # -- lookup ---------------------------------------------------------

    def node(self, node_id: str) -> Node:
        try:
            return self._by_id[node_id]
        except KeyError:
            raise GraphError(f"unknown node: {node_id}") from None

    @property
    def start(self) -> Node:
        return next(n for n in self.nodes if n.kind is NodeKind.START)

    @property
    def variables(self) -> set[str]:
        """Every template variable the flow needs to be rendered."""
        found: set[str] = set()
        for node in self.nodes:
            found |= template_variables(node.prompt)
            for edge in node.edges:
                if edge.speech:
                    found |= template_variables(edge.speech)
        return found

    # -- validation -----------------------------------------------------

    def validate(self) -> None:
        if not self.nodes:
            raise GraphError("graph has no nodes")

        if len(self._by_id) != len(self.nodes):
            raise GraphError("duplicate node ids")

        starts = [n for n in self.nodes if n.kind is NodeKind.START]
        if len(starts) != 1:
            raise GraphError(f"expected exactly one start node, found {len(starts)}")

        terminals = [n for n in self.nodes if n.is_terminal]
        if not terminals:
            raise GraphError("graph has no terminal node, so a call could never end")

        for node in self.nodes:
            self._validate_node(node)

        self._validate_reachability()

    def _validate_node(self, node: Node) -> None:
        if node.is_terminal:
            if node.edges:
                raise GraphError(f"terminal node '{node.id}' must have no outgoing edges")
            if node.intent is None:
                raise GraphError(f"terminal node '{node.id}' must declare an intent")
            return

        if node.intent is not None:
            raise GraphError(f"non-terminal node '{node.id}' must not declare an intent")
        if not node.edges:
            raise GraphError(f"node '{node.id}' is a dead end")

        for edge in node.edges:
            if edge.to not in self._by_id:
                raise GraphError(f"edge '{node.id}' -> '{edge.to}' points at an unknown node")
            if edge.to == node.id:
                raise GraphError(f"node '{node.id}' has a self-loop")

        labels = [edge.label for edge in node.edges]
        if len(set(labels)) != len(labels):
            raise GraphError(f"node '{node.id}' has duplicate edge labels")

    def _validate_reachability(self) -> None:
        seen: set[str] = set()
        queue = [self.start.id]
        while queue:
            current = queue.pop()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(edge.to for edge in self.node(current).edges)

        orphans = sorted(set(self._by_id) - seen)
        if orphans:
            raise GraphError(f"unreachable nodes: {', '.join(orphans)}")

    # -- rendering ------------------------------------------------------

    def render(self, node_id: str, context: dict[str, Any]) -> str:
        """Fill a node's prompt with call context."""
        return render_template(self.node(node_id).prompt, context)

    def missing_variables(self, context: dict[str, Any]) -> set[str]:
        return self.variables - set(context)


def template_variables(text: str) -> set[str]:
    """Field names referenced by a ``str.format`` template."""
    return {name for _, name, _, _ in Formatter().parse(text) if name}


def render_template(text: str, context: dict[str, Any]) -> str:
    missing = template_variables(text) - set(context)
    if missing:
        raise GraphError(f"missing template variables: {', '.join(sorted(missing))}")
    return text.format(**context)
