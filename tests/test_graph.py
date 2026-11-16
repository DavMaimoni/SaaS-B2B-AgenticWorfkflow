from support_agent.graph import should_continue
from langchain_core.messages import AIMessage

def test_should_continue_escalate():
    state = {"requires_escalation": True}
    assert should_continue(state) == "escalate"

def test_should_continue_end():
    state = {"requires_escalation": False, "messages": [AIMessage(content="Done")]}
    assert should_continue(state) == "end"
    
def test_should_continue_tools():
    msg = AIMessage(content="", tool_calls=[{"name": "get_crm_ticket", "args": {}, "id": "1"}])
    state = {"requires_escalation": False, "messages": [msg]}
    assert should_continue(state) == "continue"
