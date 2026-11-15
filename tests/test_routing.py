from support_agent.routing import choose_route, route_after_validate


def tri(**kw):
    return {"queue": "Product Support", "priority": "low", "type": "Request", "confidence": 1.0, "valid": True, **kw}


def test_confident_routine_ticket_goes_to_small_model():
    assert choose_route(tri(), threshold=0.75) == "small"


def test_out_of_scope_queue_is_redirected():
    assert choose_route(tri(queue="Human Resources"), threshold=0.75) == "out_of_scope"


def test_high_priority_and_outages_go_to_large_model():
    assert choose_route(tri(priority="high"), threshold=0.75) == "large"
    assert choose_route(tri(queue="Service Outages and Maintenance"), threshold=0.75) == "large"


def test_low_confidence_invalid_injection_and_sensitive_go_to_large_model():
    assert choose_route(tri(confidence=0.5), threshold=0.75) == "large"
    assert choose_route(tri(valid=False), threshold=0.75) == "large"
    assert choose_route(tri(), injection_suspected=True, threshold=0.75) == "large"
    assert choose_route(tri(), sensitive=True, threshold=0.75) == "large"


def test_after_validation():
    ok = {"triage": tri(), "draft": {"needs_human": False}, "validation_errors": [], "model_used": "small"}
    assert route_after_validate(ok) == "cache_write"
    assert route_after_validate({**ok, "validation_errors": ["placeholder"]}) == "generate_large"
    assert route_after_validate({**ok, "validation_errors": ["placeholder"], "model_used": "large",
                                 "attempts_large": 1}) == "escalate"
    assert route_after_validate({**ok, "draft": {"needs_human": True}}) == "escalate"
    assert route_after_validate({**ok, "triage": tri(priority="high")}) == "escalate"
    assert route_after_validate({**ok, "sensitive": True}) == "escalate"
