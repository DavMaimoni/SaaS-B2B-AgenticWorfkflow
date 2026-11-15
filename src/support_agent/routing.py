"""Routing rules: pure functions of the state, so they are unit-tested without any model.

| condition (checked in this order)                         | route        |
| --------------------------------------------------------- | ------------ |
| queue outside the SaaS desk (returns, sales, HR)           | out_of_scope |
| priority high, or queue Service Outages and Maintenance   | large        |
| prompt injection suspected                                 | large        |
| sensitive request (refund, data deletion, security...)     | large        |
| triage confidence below ROUTE_CONFIDENCE                   | large        |
| otherwise                                                  | small        |

After validation: an invalid small-model reply gets one more try on the large model; an invalid large-model reply,
a reply that asks for a human, a sensitive request or a high-priority ticket goes to escalation.
"""

from support_agent.config import OUT_OF_SCOPE_QUEUES, settings

ESCALATE_QUEUES = {"Service Outages and Maintenance"}


def route_after_cache(state: dict) -> str:
    return "finalize" if state.get("cache_hit") else "retrieve"


def choose_route(triage: dict, injection_suspected: bool = False, threshold: float | None = None,
                 sensitive: bool = False) -> str:
    threshold = settings().route_confidence_threshold if threshold is None else threshold
    if triage.get("queue") in OUT_OF_SCOPE_QUEUES:
        return "out_of_scope"
    if triage.get("priority") == "high" or triage.get("queue") in ESCALATE_QUEUES:
        return "large"
    if injection_suspected or sensitive:
        return "large"
    if not triage.get("valid", True) or triage.get("confidence", 0.0) < threshold:
        return "large"
    return "small"


def route_after_triage(state: dict) -> str:
    return {"small": "use_small_answer", "large": "generate_large", "out_of_scope": "out_of_scope"}[state["route"]]


def needs_escalation(state: dict) -> bool:
    triage = state.get("triage", {})
    return (bool(state.get("draft", {}).get("needs_human")) or bool(state.get("sensitive"))
            or triage.get("priority") == "high"
            or triage.get("queue") in ESCALATE_QUEUES)


def route_after_validate(state: dict) -> str:
    if state.get("validation_errors"):
        if state.get("model_used") == "small" and state.get("attempts_large", 0) < settings().max_retries_on_large:
            return "generate_large"
        return "escalate"
    return "escalate" if needs_escalation(state) else "cache_write"
