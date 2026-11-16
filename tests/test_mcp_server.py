import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from support_agent.mcp_server.server import app, verify_tenant_access

client = TestClient(app)

def test_verify_tenant_access_success():
    # Ne doit lever aucune exception
    verify_tenant_access("tenant_A", "TICK-123")

def test_verify_tenant_access_forbidden():
    with pytest.raises(HTTPException) as exc:
        verify_tenant_access("tenant_B", "TICK-123")
    assert exc.value.status_code == 403

def test_mcp_get_ticket_endpoint():
    response = client.post(
        "/tools/crm/get_ticket",
        json={"tenant_id": "tenant_A", "ticket_id": "TICK-123"}
    )
    assert response.status_code == 200
    assert "Detailed history" in response.json()["data"]
