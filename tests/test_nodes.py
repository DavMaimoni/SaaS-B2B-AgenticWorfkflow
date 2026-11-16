from support_agent.nodes import execute_tools_node
from langchain_core.messages import AIMessage

def test_execute_tools_circuit_breaker():
    """Vérifie que le circuit breaker bloque les boucles d'outils infinies."""
    state = {
        "messages": [AIMessage(content="test")],
        "loop_count": 5,
        "requires_escalation": False
    }
    result = execute_tools_node(state)
    
    assert result["requires_escalation"] is True
    assert result["loop_count"] == 6
    assert "Too many automated attempts" in result["messages"][0].content
