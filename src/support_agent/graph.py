from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from support_agent.state import SupportAgentState
from support_agent.nodes import (
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

def build_graph(checkpointer=None):
    """Builds and compiles the StateGraph. Required by the evaluation pipeline."""
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

    # Use the provided checkpointer if given, otherwise fall back to MemorySaver
    memory = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["human_escalation"] 
    )

# Global instance for normal app usage (API, CLI, etc.)
graph = build_graph()

def run_ticket(graph_instance, row: dict) -> dict:
    """
    Helper function for the evaluation pipeline.
    Executes the graph on a single mocked ticket and extracts the result.
    """
    subject = row.get("subject", "")
    body = row.get("body", "")
    
    # Use a default tenant_id for evaluation mocks if not provided
    tenant_id = row.get("tenant_id", "eval_tenant")
    
    initial_message = HumanMessage(content=f"Subject: {subject}\n\n{body}")
    
    initial_state = {
        "messages": [initial_message],
        "tenant_id": tenant_id,
        "requires_escalation": False,
        "loop_count": 0,
        "summary": None
    }
    
    # Required for MemorySaver checkpointer
    config = {"configurable": {"thread_id": row.get("ticket_id", "eval_thread")}}
    
    final_state = graph_instance.invoke(initial_state, config=config)
    
    # Extract the agent's final answer from the messages
    messages = final_state.get("messages", [])
    answer = messages[-1].content if messages else ""
    
    return {
        "answer": answer,
        "escalated": final_state.get("requires_escalation", False)
    }
