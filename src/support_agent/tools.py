from pydantic import BaseModel, Field
from langchain_core.tools import tool
from typing import Optional

class GetTicketSchema(BaseModel):
    tenant_id: str = Field(description="Client company ID (mandatory for data isolation).")
    ticket_id: str = Field(description="Unique identifier of the CRM ticket.")

class UpdateTicketSchema(BaseModel):
    tenant_id: str = Field(description="Client company ID (mandatory for data isolation).")
    ticket_id: str = Field(description="Unique identifier of the CRM ticket.")
    new_status: str = Field(description="New status to apply (e.g., 'resolved', 'escalated', 'in_progress').")
    resolution_summary: str = Field(description="Technical summary of the AI agent's intervention.")

@tool(args_schema=GetTicketSchema)
def get_crm_ticket(tenant_id: str, ticket_id: str) -> str:
    """
    Retrieves the details and history of a specific support ticket.
    Must be used to gather context before attempting to resolve an issue.
    """
    # TODO: Integrate with MCP Server / actual CRM database
    # Enforce tenant_id verification here to prevent cross-tenant data leaks
    return f"Mock Ticket Data for {ticket_id} (Tenant: {tenant_id}): Customer reported login issues."

@tool(args_schema=UpdateTicketSchema)
def update_crm_ticket(tenant_id: str, ticket_id: str, new_status: str, resolution_summary: str) -> str:
    """
    Updates the status and adds a resolution summary to a ticket in the CRM.
    Should be called only when an issue is definitively resolved or requires escalation.
    """
    # TODO: Integrate with MCP Server / actual CRM database
    # Enforce tenant_id verification here to prevent unauthorized writes
    return f"Success: Ticket {ticket_id} has been updated to '{new_status}' for tenant {tenant_id}. Summary attached."
