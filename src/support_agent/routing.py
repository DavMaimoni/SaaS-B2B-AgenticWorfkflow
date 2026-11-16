from support_agent.state import SupportAgentState

def route_intent(state: SupportAgentState) -> str:
    """Fast semantic routing based on keywords or local vector search."""
    messages = state.get("messages", [])
    if not messages:
        return "general_agent"
        
    last_user_message = messages[-1].content.lower()
    
    if any(word in last_user_message for word in ["invoice", "billing", "credit card", "charge", "facture"]):
        return "billing_agent"
    elif any(word in last_user_message for word in ["bug", "error", "api", "crash", "timeout", "500"]):
        return "technical_agent"
        
    return "general_agent"
