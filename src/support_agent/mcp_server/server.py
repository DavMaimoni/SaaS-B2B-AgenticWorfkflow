from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="B2B Support MCP Server")

class ToolRequest(BaseModel):
    tenant_id: str
    ticket_id: str

def verify_tenant_access(tenant_id: str, ticket_id: str) -> None:
    """
    Security middleware to ensure the requested CRM entity belongs to the provided tenant.
    Prevents cross-tenant data leaks (IDOR vulnerability) if the LLM hallucinates arguments.
    """
    # Mock database validation
    # In production: SELECT tenant_id FROM tickets WHERE id = ticket_id
    mock_ticket_db = {
        "TICK-123": "tenant_A",
        "TICK-456": "tenant_B"
    }
    
    # If the ticket exists but belongs to someone else, reject immediately
    if ticket_id in mock_ticket_db and mock_ticket_db[ticket_id] != tenant_id:
        raise HTTPException(status_code=403, detail="Security Error: Cross-tenant access denied.")

@app.post("/tools/crm/get_ticket")
def mcp_get_ticket(request: ToolRequest):
    """
    MCP Endpoint to fetch a ticket, strictly protected by B2B isolation.
    """
    # 1. Enforce tenant boundary
    verify_tenant_access(request.tenant_id, request.ticket_id)
    
    # 2. Proceed with database fetch
    return {
        "status": "success", 
        "data": f"Detailed history for {request.ticket_id} (Tenant verified)"
    }
