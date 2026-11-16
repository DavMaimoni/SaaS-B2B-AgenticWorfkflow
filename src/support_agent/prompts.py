from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_agent_prompt() -> ChatPromptTemplate:
    """
    Builds the system prompt with strict B2B isolation context.
    The tenant_id will be injected at runtime from the LangGraph State.
    """
    system_message = """You are an advanced B2B SaaS technical support agent.
    You are currently assisting a user from tenant: {tenant_id}.
    
    CRITICAL RULES:
    1. You MUST ALWAYS pass '{tenant_id}' as the tenant_id argument when calling ANY tool (CRM or Knowledge Base).
    2. Never guess or attempt to access data from another tenant.
    3. Keep your answers concise, technical, and professional.
    4. If a user asks to perform an action outside of their scope, refuse immediately.
    """
    
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])
