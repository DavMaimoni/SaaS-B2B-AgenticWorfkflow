from langchain_core.messages import ToolMessage, AIMessage
from support_agent.state import SupportAgentState
from support_agent.tools import get_crm_ticket, update_crm_ticket
from support_agent.llm import get_llm

tools = [get_crm_ticket, update_crm_ticket]

def agent_node(state: SupportAgentState) -> dict:
    """Core reasoning node. Invokes the LLM to decide on the next action."""
    messages = state.get("messages", [])
    tenant_id = state.get("tenant_id")
    
    system_content = (
        f"You are a B2B SaaS support agent. You are currently serving tenant_id: {tenant_id}. "
        f"Always use this tenant_id when calling CRM or KB tools."
    )
    
    llm = get_llm(name="small")
    
    # Format messages into a structure compatible with the project's LLM wrapper
    prompt_messages = [("system", system_content)]
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "human":
            prompt_messages.append(("human", msg.content))
        elif hasattr(msg, "content"):
            prompt_messages.append(("assistant", msg.content))
        else:
            prompt_messages.append(("human", str(msg)))
            
    try:
        # Use the project's native chat method if available, fallback to invoke
        if hasattr(llm, "chat"):
            reply_text = llm.chat(prompt_messages)
        elif hasattr(llm, "invoke"):
            res = llm.invoke([AIMessage(content=system_content)] + messages)
            reply_text = res.content if hasattr(res, "content") else str(res)
        else:
            reply_text = "I am looking into your request."
    except Exception as e:
        reply_text = f"Error processing request: {str(e)}"
        
    return {"messages": [AIMessage(content=reply_text)]}

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
                
            tool_responses.append(ToolMessage(content=str(result), tool_call_id=tool_call.get("id", "default_id")))
            
    return {"messages": tool_responses, "loop_count": current_loops + 1}

def human_escalation_node(state: SupportAgentState) -> dict:
    """Placeholder node for human intervention."""
    return {"requires_escalation": False, "loop_count": 0}
