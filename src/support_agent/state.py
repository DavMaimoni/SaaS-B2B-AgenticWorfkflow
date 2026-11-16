from typing import Annotated, TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class SupportAgentState(TypedDict, total=False):
    """
    Centralized state of the LangGraph workflow.
    Tracks conversation history, B2B context, triage metadata, and execution guardrails.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    tenant_id: str
    requires_escalation: bool
    loop_count: int
    summary: Optional[str]
    triage: Optional[Dict[str, Any]]
    status: Optional[str]
    model_used: Optional[str]
    route: Optional[str]
    cache_hit: Optional[bool]
    redirected: Optional[bool]
    validation_errors: Optional[List[str]]
