from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class SupportAgentState(TypedDict):
    """
    Centralized state of the LangGraph workflow.
    Tracks conversation history, B2B context, and execution guardrails.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    tenant_id: str  # Strict B2B data isolation: identifies the client company
    requires_escalation: bool  # Flag to trigger human-in-the-loop intervention
    loop_count: int  # Circuit breaker tracking to prevent infinite Tool Calling Loops
    summary: Optional[str]  # Storage of compressed context for long conversation histories
