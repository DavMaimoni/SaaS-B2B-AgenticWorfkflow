from src.support_agent.state import SupportAgentState

def route_intent(state: SupportAgentState) -> str:
    """
    Fast semantic routing based on keywords or local vector search.
    Acts as a triage layer before invoking expensive LLM calls.
    Returns the specific workflow branch to take.
    """
    messages = state.get("messages", [])
    if not messages:
        return "general_agent"
        
    last_user_message = messages[-1].content.lower()
    
    # In a production implementation, this should be a vector similarity search (FAISS/Chroma)
    # against predefined category embeddings, rather than hardcoded keywords.
    if any(word in last_user_message for word in ["invoice", "billing", "credit card", "charge", "facture"]):
        return "billing_agent"
    elif any(word in last_user_message for word in ["bug", "error", "api", "crash", "timeout", "500"]):
        return "technical_agent"
        
    return "general_agent"
