from support_agent.routing import route_intent
from langchain_core.messages import HumanMessage

def test_route_intent_billing():
    state = {"messages": [HumanMessage(content="I have a problem with my invoice")]}
    assert route_intent(state) == "billing_agent"

def test_route_intent_technical():
    state = {"messages": [HumanMessage(content="The API returns a 500 error")]}
    assert route_intent(state) == "technical_agent"

def test_route_intent_general():
    state = {"messages": [HumanMessage(content="How do I change my profile picture?")]}
    assert route_intent(state) == "general_agent"
