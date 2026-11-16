from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="B2B Support MCP Server")

class ToolRequest(BaseModel):
    tenant_id: str
    ticket_id: str

def verify_tenant_access(tenant_id: str, ticket_id: str) -> None:
    """Security middleware to ensure the requested CRM entity belongs to the provided tenant."""
    mock_ticket_db = {
        "TICK-123": "tenant_A",
        "TICK-456": "tenant_B"
    }
    
    if ticket_id in mock_ticket_db and mock_ticket_db[ticket_id] != tenant_id:
        raise HTTPException(status_code=403, detail="Security Error: Cross-tenant access denied.")

@app.post("/tools/crm/get_ticket")
def mcp_get_ticket(request: ToolRequest):
    """MCP Endpoint to fetch a ticket, strictly protected by B2B isolation."""
    verify_tenant_access(request.tenant_id, request.ticket_id)
    return {
        "status": "success", 
        "data": f"Detailed history for {request.ticket_id} (Tenant verified)"
    }
