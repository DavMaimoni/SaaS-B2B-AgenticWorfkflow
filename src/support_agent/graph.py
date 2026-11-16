from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from support_agent.state import SupportAgentState
from support_agent.nodes import (
    agent_node,
    execute_tools_node,
    human_escalation_node,
    _parse_sft
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

    memory = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["human_escalation"] 
    )

def run_ticket(graph_instance, subject, body=None, customer_email=None, thread_id=None, **kwargs) -> dict:
    """
    Executes the graph on a single ticket and safely maps the internal StateGraph
    format into the dictionary structure explicitly expected by eval/run_eval.py.
    """
    # Gère les cas où la fonction est appelée avec un dictionnaire entier
    if isinstance(subject, dict):
        subject_text = subject.get("subject", "")
        body_text = subject.get("body", "")
        tid = subject.get("ticket_id", "eval_thread")
    else:
        subject_text = subject
        body_text = body or ""
        tid = thread_id or "eval_thread"

    initial_message = HumanMessage(content=f"Subject: {subject_text}\n\n{body_text}")
    
    initial_state = {
        "messages": [initial_message],
        "tenant_id": "eval_tenant",
        "requires_escalation": False,
        "loop_count": 0,
        "summary": None
    }
    
    config = {"configurable": {"thread_id": tid}}
    
    # Exécution du graphe LangGraph
    final_state = graph_instance.invoke(initial_state, config=config)
    
    # Extraction de la réponse finale
    messages = final_state.get("messages", [])
    raw_answer = messages[-1].content if messages else ""
    
    # Analyse de la réponse générée par le modèle affiné (mock/lora)
    try:
        parsed = _parse_sft(raw_answer)
    except Exception:
        parsed = None

    if parsed:
        labels, answer = parsed
        status = "answered"
        validation_errors = []
    else:
        labels = {"queue": None, "priority": None, "type": None}
        answer = raw_answer
        status = "invalid"
        validation_errors = ["invalid_json"]

    # Remplace le statut si le circuit-breaker du graphe a exigé une escalade humaine
    if final_state.get("requires_escalation"):
        status = "escalated"

    # Retourne exactement le format attendu par le script de benchmark (qui contourne les schémas stricts)
    return {
        "triage": labels,
        "status": status,
        "model_used": "small",
        "route": "graph",
        "cache_hit": False,
        "answer": answer,
        "validation_errors": validation_errors,
    }
