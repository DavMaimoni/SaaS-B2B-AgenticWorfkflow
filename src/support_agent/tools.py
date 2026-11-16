from pydantic import BaseModel, Field
from langchain_core.tools import tool
from typing import Any

# --- Legacy / Evaluation Compatibility Stubs ---
# These allow eval/run_eval.py to inject a fresh in-memory CRM for each test run
# to prevent data leakage between evaluation triples.

class ToolGateway:
    """Gateway to access external systems like the CRM. Required for the eval pipeline."""
    def __init__(self, crm: Any = None):
        self.crm = crm

_gateway = None

def set_gateway(gateway: ToolGateway) -> None:
    """Sets the global tool gateway. Used by eval/run_eval.py to inject isolated CRMs."""
    global _gateway
    _gateway = gateway

# --- B2B Support Tools ---

class GetTicketSchema(BaseModel):
    tenant_id: str = Field(description="Client company ID (mandatory for data isolation).")
    ticket_id: str = Field(description="Unique identifier of the CRM ticket.")

class UpdateTicketSchema(BaseModel):
    tenant_id: str = Field(description="Client company ID (mandatory for data isolation).")
    ticket_id: str = Field(description="Unique identifier of the CRM ticket.")
    new_status: str = Field(description="New status to apply (e.g., 'resolved', 'escalated').")
    resolution_summary: str = Field(description="Technical summary of the AI agent's intervention.")

@tool(args_schema=GetTicketSchema)
def get_crm_ticket(tenant_id: str, ticket_id: str) -> str:
    """Retrieves the details and history of a specific support ticket."""
    return f"Mock Ticket Data for {ticket_id} (Tenant: {tenant_id}): Customer reported issues."

@tool(args_schema=UpdateTicketSchema)
def update_crm_ticket(tenant_id: str, ticket_id: str, new_status: str, resolution_summary: str) -> str:
    """Updates the status and adds a resolution summary to a ticket in the CRM."""
    return f"Success: Ticket {ticket_id} has been updated to '{new_status}' for tenant {tenant_id}."
