"""Output guardrails for what the agent says on a live call.

The conversation graph refuses an illegal *move*; these refuse an illegal
*sentence*. Import from here rather than the submodules -- `rails` pulls in
NeMo Guardrails, and `checks` is deliberately importable without it so the
rules can be tested without a runtime.
"""

from app.voice.guardrails.checks import Violation, check_output

__all__ = ["Violation", "check_output", "build_guardrail"]


def build_guardrail(mode: str, *, expected_amount_rupees: str | None = None):
    """Return a `GuardrailProcessor`, or ``None`` when guardrails are off.

    NeMo is imported here rather than at module scope so that a deployment with
    guardrails disabled -- the webhook service, the worker, every test that
    does not exercise them -- never pays to load it.
    """
    if mode == "off":
        return None

    from app.voice.guardrails.processor import GuardrailProcessor
    from app.voice.guardrails.rails import DunningRails

    guard = DunningRails(expected_amount_rupees=expected_amount_rupees)
    return GuardrailProcessor(guard, mode=mode)
