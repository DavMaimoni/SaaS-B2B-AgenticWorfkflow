"""Create (or reset) the simulated CRM in data/crm.sqlite: 20 fictional accounts and their invoices.

    uv run python scripts/seed_crm.py --reset
"""

import argparse
import json

from support_agent.config import CRM_PATH
from support_agent.mcp_server.crm import CRM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete the database first")
    args = ap.parse_args()
    if args.reset and CRM_PATH.exists():
        CRM_PATH.unlink()
    crm = CRM(CRM_PATH)
    accounts = [dict(r) for r in crm.db.execute("SELECT id, email, company, plan, billing_cycle, status FROM accounts")]
    print(json.dumps(accounts[:5], indent=2), f"\n... {len(accounts)} accounts in {CRM_PATH}")


if __name__ == "__main__":
    main()
