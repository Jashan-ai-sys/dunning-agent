# The conversation graph

The call is a small state machine, not one long prompt. Nine nodes: one start,
three agent nodes, five terminals. Defined in [`app/voice/flow.py`](../app/voice/flow.py),
traversed by [`app/voice/walker.py`](../app/voice/walker.py), validated at import
time by [`app/voice/graph.py`](../app/voice/graph.py) so a malformed flow fails at
startup rather than halfway through a call to a customer.

---

## The shape

```
                            ┌──────────────────────────────┐
                            │            greet             │  START
                            │  confirm identity. nothing    │
                            │  about the payment before it  │
                            └───────┬──────────────┬────────┘
                    identity_confirmed        not_the_customer
                                    │              │  ↯ speaks a line
                                    ▼              ▼
                            ┌───────────────┐   ┌──────────────┐
                            │    explain    │   │ wrong_number │ END
                            │ amount + why  │   │              │→ wrong_number
                            └──┬─────────┬──┘   └──────────────┘
                    acknowledged    disputes_charge
                               │         └──────────────┐
                               ▼                        │
                       ┌────────────────┐               │
                       │ reason_inquiry │               │
                       │ one question:  │               │
                       │ do they know?  │               │
                       └──┬────┬────┬───┘               │
                 reason_given │    └ disputes_charge ───┤
                          │   │                         │
                          │   └ financial_difficulty ─┐ │
                          ▼                           │ │
                  ┌──────────────────┐                │ │
                  │    ask_intent    │                │ │
                  │ exactly three    │                │ │
                  │ options, no push │                │ │
                  └─┬───┬───┬───┬──┬─┘                │ │
            pay_now  │   │   │   │  └ disputes_charge─┼─┤
                     │   │   │   └ financial_difficulty┤ │
             pay_later│   │   └ declined               │ │
                     │   │                             │ │
        ┌────────────┘   │        ┌────────────────────┘ │
        │                │        │                      │
        ▼                ▼        ▼                      ▼
  ┌──────────┐  ┌────────────┐ ┌──────────┐      ┌──────────┐
  │ pay_now  │  │ pay_later  │ │ declined │      │ dispute  │  all END
  │→retry_now│  │→retry_later│ │→declined │      │→dispute  │
  │ ↯ speaks │  └────────────┘ └──────────┘      └──────────┘
  └──────────┘
```

`↯` marks an edge carrying its own `speech` — a line rendered locally and spoken
during the move, so the handoff does not wait on the model.

---

## Nodes

| Node | Kind | Job | Extracts |
|---|---|---|---|
| `greet` | START | Confirm identity. **Nothing about the payment is said before this** — it is someone's billing information. | `identity_confirmed` |
| `explain` | AGENT | State the amount and the reason, matter-of-fact. Most failures are bank-side, not the customer's fault. | — |
| `reason_inquiry` | AGENT | **One** short question: do they know why it failed? Do not lecture, do not offer options yet, do not ask twice. | `failure_cause` |
| `ask_intent` | AGENT | Offer exactly three options and let them choose. No pushing. **No discount — the agent has no authority to change the amount.** | `preferred_time` |
| `pay_now` | END | → `retry_now` | — |
| `pay_later` | END | → `retry_later` | — |
| `declined` | END | → `declined` | — |
| `wrong_number` | END | → `wrong_number` | — |
| `dispute` | END | → `dispute`. Handed to a human, never argued with. | — |

---

## Edges

Every edge carries a `condition` — plain English describing what earns that
label. Only the edges legal from the *current* node are rendered into the turn,
so the model is never shown a move it cannot make.

### From `greet`

| Label | To | Condition |
|---|---|---|
| `identity_confirmed` | `explain` | The person confirms they are the customer, or clearly indicates it is them. |
| `not_the_customer` | `wrong_number` | The person says this is the wrong number, that they are someone else, or that they do not know the customer. |

`not_the_customer` carries speech: *"माफ़ कीजिए, मैं नंबर की जाँच करवा लेता हूँ।"* —
rendered locally, because a wrong number should be closed politely and at once
rather than after a model round trip.

### From `explain`

| Label | To | Condition |
|---|---|---|
| `acknowledged` | `reason_inquiry` | The customer acknowledges the failed payment or asks what to do next. |
| `disputes_charge` | `dispute` | The customer says the charge is wrong, that they already paid, that they cancelled the subscription, or that they never signed up. |

### From `reason_inquiry`

| Label | To | Condition |
|---|---|---|
| `reason_given` | `ask_intent` | The customer gives any reason, says they do not know, asks to get on with it, or asks what they should do about it. |
| `financial_difficulty` | `pay_later` | The customer says they cannot afford it right now, are short of money, have lost work, or are in financial trouble. |
| `disputes_charge` | `dispute` | As above. |

### From `ask_intent`

| Label | To | Condition |
|---|---|---|
| `pay_now` | `pay_now` | The customer wants to pay now, or agrees to receive a payment link. |
| `pay_later` | `pay_later` | The customer wants to pay later, asks us to call back, or names a specific day or time. |
| `declined` | `declined` | The customer refuses to pay, wants to cancel the subscription, or asks not to be contacted again. |
| `financial_difficulty` | `pay_later` | *"...This is **not a refusal** — do not treat it as one."* |
| `disputes_charge` | `dispute` | As above. |

---

## Three properties the shape enforces

### 1. `financial_difficulty` never reaches `declined`

It appears on two nodes and both times routes to `pay_later`. The node docstring
says why:

> *"I cannot afford it" is a later, not a no. Routing it to `declined` would
> close the case and stop contact on someone who never refused.*

`declined` is permanent — it stops contact. Someone short of money this week has
not refused to pay, and the graph makes that misrouting structurally impossible
rather than relying on the model to be careful.

### 2. `disputes_charge` is reachable from every agent node

A customer disputes a charge whenever they think of it, not when the script
expects it. So the edge exists on `explain`, `reason_inquiry` and `ask_intent`,
and always lands on `dispute`, which is handed to a human. The agent never
argues about a disputed charge.

### 3. The amount cannot be spoken before identity is confirmed

`greet` is the only path in, and its prompt forbids mentioning the payment. The
amount lives in `explain`'s prompt, which the model does not see until it has
already moved there. This is enforced by a test:

```
test_greeting_instructions_do_not_leak_the_amount
  "Whoever answers the phone must not hear someone else's billing detail."
```

That test is also why the whole graph is *not* shown to the model up front,
which would otherwise let one inference both move and speak — see
[architecture.md](architecture.md).

---

## How a move actually happens

The model never picks a node. It picks an **edge label**, and the walker
validates it against the node currently occupied:

```python
for edge in self._node.edges:
    if edge.label == label:
        ...move...
        return self._node

raise InvalidTransition(
    f"'{label}' is not a valid move from '{origin}'; expected one of: {valid}"
)
```

The tool schema declares **every** label in the flow as its enum, not just the
current node's — deliberately:

> *A tool schema that changes mid-call changes the request prefix and costs the
> provider's cache, and a schema that disagrees with the node's own instructions
> is worse than one that is merely broad. Which labels are legal here stays with
> the walker, which rejects the rest.*

**Broad schema, narrow enforcement.** The model can emit `pay_now` while standing
at `greet`; the walker refuses it.

### Rejections are kept

Every attempt becomes an `Observation`, accepted or not:

```python
Observation(node_id=origin, label=label, accepted=False,
            utterance=utterance, rejection=message)
```

> *A label the model reached for and could not have is a harder negative than
> anything we would think to write by hand.*

It is already the shape a DSPy signature needs, so prompt optimisation runs
against real traffic rather than invented examples.

---

## How the intent is decided

The intent is **not written by the model**. It is a property of where the
conversation ended:

1. model calls `transition(label)`
2. walker validates the label against the current node
3. walker moves; if the destination is terminal, `finished` becomes true
4. `finalise()` reads `walker.intent` — a fixed attribute of that node
5. `apply_call_result` writes it to the case

To record `retry_now`, the conversation has to actually reach `pay_now`, through
`greet → explain → reason_inquiry → ask_intent`, every step validated.

Two intents have no node, because they describe a call that did not finish:

| Intent | Meaning |
|---|---|
| `no_answer` | The call never connected |
| `unclear` | Connected, but ended before any terminal node |

`unclear` is deliberate. A call that fizzled is not guessed at — it records
nothing conclusive and the case stays open for another attempt. The alternative
is inventing an outcome about someone's money.

---

## Validation at import

`ConversationGraph.validate()` runs when the module loads, and refuses:

- no nodes, or duplicate node ids
- anything other than exactly one START node
- no terminal node — a call that could never end
- a terminal node with outgoing edges, or without an `intent`
- a non-terminal node with an `intent`, or with no edges (a dead end)
- an edge pointing at an unknown node, or a self-loop
- duplicate edge labels on one node
- unreachable nodes

So a malformed flow fails at startup, not halfway through a call.

---

## Comparison with Dograh

The graph is Dograh-style, and [their docs](https://docs.dograh.com/core-concepts/context-and-variables)
describe extraction happening at a node during the call — the same choice made
here. They also leave a variable empty rather than guessing when the
conversation does not determine it, which is what `unclear` is.

The difference is who writes the outcome. In Dograh the LLM *"reads the
transcript so far and fills in each variable"* — the model produces the value.
Here the model can only name an edge, and the intent is whichever terminal node
that walk arrived at. On a path that moves money, the outcome is not something
the model should be able to assert.
