from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.support_agent.state import SupportAgentState
from src.support_agent.nodes import (
    agent_node,
    execute_tools_node,
    human_escalation_node
)

def should_continue(state: SupportAgentState) -> str:
    """Routing logic based on the agent's output and circuit breaker status."""
    if state.get("requires_escalation"):
        return "escalate"
    
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    
    return "end"

builder = StateGraph(SupportAgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", execute_tools_node)
builder.add_node("human_escalation", human_escalation_node)

builder.set_entry_point("agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "escalate": "human_escalation",
        "end": END
    }
)

builder.add_edge("tools", "agent")
builder.add_edge("human_escalation", END)

memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["human_escalation"] 
)
