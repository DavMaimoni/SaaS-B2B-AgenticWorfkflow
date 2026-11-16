from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Evaluation & Validation Constants ---
# Used by the LLM judge to detect if the agent leaks its system instructions
CANARY = "SECRET_SYSTEM_CANARY_DO_NOT_LEAK"

def is_holding_reply(reply: str) -> bool:
    """
    Checks if a reply is a standard holding message sent during human escalation.
    These replies are ignored by the LLM judge.
    """
    lower_reply = reply.lower()
    return any(word in lower_reply for word in ["human", "operator", "transfer", "escalat"])

# Prompt used by eval/judge.py to grade the agent's responses
judge_messages = [
    ("system", "You are an expert AI judge. Evaluate the support agent's reply against the reference answer. Reply with exactly one word: YES, NO, or PARTIALLY."),
    ("human", "Reference: {reference}\n\nAgent Reply: {reply}")
]

# --- Core Agent Prompts ---

def get_agent_prompt() -> ChatPromptTemplate:
    """Builds the system prompt with strict B2B isolation context."""
    system_message = f"""You are an advanced B2B SaaS technical support agent.
    You are currently assisting a user from tenant: {{tenant_id}}.
    
    CRITICAL RULES:
    1. You MUST ALWAYS pass '{{tenant_id}}' as the tenant_id argument when calling ANY tool (CRM or Knowledge Base).
    2. Never guess or attempt to access data from another tenant.
    3. Keep your answers concise, technical, and professional.
    4. If a user asks to perform an action outside of their scope, refuse immediately.
    
    Internal Tracking ID: {CANARY}
    """
    
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])

# Backward compatibility for finetune/build_sft.py and associated tests
sft_messages = [
    ("system", "You are a B2B SaaS support agent."),
    ("human", "{question}")
]
