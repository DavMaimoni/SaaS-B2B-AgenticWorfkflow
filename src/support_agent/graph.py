from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.support_agent.state import SupportAgentState
from src.support_agent.nodes import (
    agent_node,
    execute_tools_node,
    human_escalation_node
)

def should_continue(state: SupportAgentState) -> str:
    """
    Routing logic to determine the next node based on the agent's output
    and the circuit breaker status.
    """
    if state.get("requires_escalation"):
        return "escalate"
    
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    # If the LLM makes a tool call, route to the tool execution node
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    
    # Otherwise, the interaction is complete
    return "end"

# Initialize the workflow graph with the defined State
builder = StateGraph(SupportAgentState)

# Define the nodes
builder.add_node("agent", agent_node)
builder.add_node("tools", execute_tools_node)
builder.add_node("human_escalation", human_escalation_node)

# Set the entry point
builder.set_entry_point("agent")

# Define the edges and routing logic
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "escalate": "human_escalation",
        "end": END
    }
)

# Return to the agent after tool execution to evaluate the tool's output
builder.add_edge("tools", "agent")
# End the workflow after human intervention (or route back to agent if needed)
builder.add_edge("human_escalation", END)

# Initialize checkpointer for Time-Travel and Human Validation mode
memory = MemorySaver()

# Compile the graph with an interruption breakpoint before human escalation
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["human_escalation"] 
)
