from langchain_core.messages import ToolMessage, AIMessage
from src.support_agent.state import SupportAgentState
from src.support_agent.tools import get_crm_ticket, update_crm_ticket
from src.support_agent.llm import get_llm # Assumes a pre-configured LLM instance

# Bind available tools to the LLM
tools = [get_crm_ticket, update_crm_ticket]
llm_with_tools = get_llm(name="agent").bind_tools(tools)

def agent_node(state: SupportAgentState) -> dict:
    """
    Core reasoning node. Invokes the LLM to decide on the next action or respond to the user.
    """
    messages = state.get("messages", [])
    tenant_id = state.get("tenant_id")
    
    # Inject B2B context systematically into the system prompt
    # to ensure the LLM respects tenant boundaries when formatting tool arguments
    system_instruction = AIMessage(
        content=f"You are a B2B SaaS support agent. You are currently serving tenant_id: {tenant_id}. "
                f"Always use this tenant_id when calling CRM or KB tools."
    )
    
    # Invoke the LLM with the context and history
    response = llm_with_tools.invoke([system_instruction] + messages)
    
    return {"messages": [response]}

def execute_tools_node(state: SupportAgentState) -> dict:
    """
    Executes the tools requested by the LLM. 
    Includes a circuit breaker to prevent infinite tool calling loops.
    """
    current_loops = state.get("loop_count", 0)
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    # Circuit Breaker: Prevent LLM from getting stuck in an infinite loop of failed tool calls
    if current_loops >= 5:
        return {
            "requires_escalation": True,
            "messages": [AIMessage(content="System error: Too many automated attempts. Transferring to a human operator.")],
            "loop_count": current_loops + 1
        }
        
    tool_responses = []
    
    # Execute each tool call requested by the LLM
    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            try:
                # Map tool names to actual functions (in production, use LangGraph's ToolNode)
                if tool_name == "get_crm_ticket":
                    result = get_crm_ticket.invoke(tool_args)
                elif tool_name == "update_crm_ticket":
                    result = update_crm_ticket.invoke(tool_args)
                else:
                    result = f"Error: Tool {tool_name} not recognized."
                    
            except Exception as e:
                result = f"Error executing {tool_name}: {str(e)}"
                
            # Append the result as a ToolMessage to feed back to the LLM
            tool_responses.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
            
    return {
        "messages": tool_responses,
        "loop_count": current_loops + 1
    }

def human_escalation_node(state: SupportAgentState) -> dict:
    """
    Placeholder node for human intervention. 
    The graph is configured to pause (interrupt_before) right before this node executes.
    """
    # When execution resumes after human review, this node finalizes the escalation state
    return {
        "requires_escalation": False, # Reset after handling
        "loop_count": 0 # Reset circuit breaker
    }
