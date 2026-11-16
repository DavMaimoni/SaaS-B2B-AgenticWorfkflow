from langchain_core.messages import ToolMessage, AIMessage
from src.support_agent.state import SupportAgentState
from src.support_agent.tools import get_crm_ticket, update_crm_ticket
from src.support_agent.llm import get_llm

tools = [get_crm_ticket, update_crm_ticket]
# Fix : passed the missing 'name' positional argument
llm_with_tools = get_llm(name="agent").bind_tools(tools)

def agent_node(state: SupportAgentState) -> dict:
    """Core reasoning node. Invokes the LLM to decide on the next action."""
    messages = state.get("messages", [])
    tenant_id = state.get("tenant_id")
    
    system_instruction = AIMessage(
        content=f"You are a B2B SaaS support agent. You are currently serving tenant_id: {tenant_id}. "
                f"Always use this tenant_id when calling CRM or KB tools."
    )
    
    response = llm_with_tools.invoke([system_instruction] + messages)
    return {"messages": [response]}

def execute_tools_node(state: SupportAgentState) -> dict:
    """Executes the tools requested by the LLM with a circuit breaker."""
    current_loops = state.get("loop_count", 0)
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    if current_loops >= 5:
        return {
            "requires_escalation": True,
            "messages": [AIMessage(content="System error: Too many automated attempts. Transferring to a human.")],
            "loop_count": current_loops + 1
        }
        
    tool_responses = []
    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            try:
                if tool_name == "get_crm_ticket":
                    result = get_crm_ticket.invoke(tool_args)
                elif tool_name == "update_crm_ticket":
                    result = update_crm_ticket.invoke(tool_args)
                else:
                    result = f"Error: Tool {tool_name} not recognized."
            except Exception as e:
                result = f"Error executing {tool_name}: {str(e)}"
                
            tool_responses.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            
    return {"messages": tool_responses, "loop_count": current_loops + 1}

def human_escalation_node(state: SupportAgentState) -> dict:
    """Placeholder node for human intervention."""
    return {"requires_escalation": False, "loop_count": 0}
