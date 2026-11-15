"""The simulated CRM: accounts, invoices, support tickets and escalations in SQLite.

Every write takes an `idempotency_key`: replaying the same key returns the first result and changes nothing,
so the graph can retry a write whose answer was lost without creating a duplicate ticket (course lab 2).
All companies and people are fictional; emails use the reserved example.com domain.
"""

from __future__ import annotations

import json
import random
import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY, email TEXT UNIQUE, company TEXT, plan TEXT, billing_cycle TEXT,
    seats INTEGER, status TEXT, renewal_date TEXT, region TEXT);
CREATE TABLE IF NOT EXISTS invoices (
    id TEXT PRIMARY KEY, account_id TEXT, issued_on TEXT, amount_usd REAL, status TEXT);
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY, account_email TEXT, subject TEXT, summary TEXT, queue TEXT, priority TEXT,
    status TEXT, created_at TEXT, idempotency_key TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY, ticket_id TEXT, reason TEXT, team TEXT, created_at TEXT, idempotency_key TEXT UNIQUE);
"""

COMPANIES = ["Northwind Analytics", "Bluepeak Studio", "Cedar & Lane", "Orbital Freight", "Kestrel Health Ops",
             "Lumen Retail Group", "Harbor Legal", "Quanta Labs", "Mosaic Events", "Tidewater Energy",
             "Pinecrest Schools", "Atlas Robotics", "Juniper Media", "Redwood Advisory", "Sable Logistics",
             "Vantage Clinics", "Zephyr Travel", "Granite Construction Co", "Ivory Finance", "Willow Nonprofit"]
PLAN_PRICE = {"Free": 0, "Pro": 12, "Enterprise": 25}
ESCALATION_TEAMS = {"Billing and Payments": "billing", "IT Support": "security",
                    "Service Outages and Maintenance": "sre", "Technical Support": "engineering"}


class CRM:
    def __init__(self, path: Path | str = ":memory:"):
        self.path = str(path)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        if not self.db.execute("SELECT 1 FROM accounts LIMIT 1").fetchone():
            self.seed()

    # -- seed ------------------------------------------------------------------------------------------------------
    def seed(self, seed: int = 7) -> None:
        rng = random.Random(seed)
        today = date(2026, 11, 1)
        with self._lock, self.db:
            for i, company in enumerate(COMPANIES):
                plan = ["Free", "Pro", "Pro", "Enterprise"][i % 4]
                cycle = "annual" if plan == "Enterprise" or i % 3 == 0 else "monthly"
                seats = 5 if plan == "Free" else rng.choice([8, 12, 20, 35, 60, 120])
                status = "read_only" if i == 5 else "active"
                slug = company.lower().replace(" & ", "-").replace(" ", "-")
                acc_id = f"ACC-{1001 + i}"
                renewal = today + timedelta(days=rng.randint(3, 330))
                self.db.execute("INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?,?)",
                                (acc_id, f"admin@{slug}.example.com", company, plan, cycle, seats, status,
                                 renewal.isoformat(), "EU" if i % 2 else "US"))
                for k in range(3):
                    months = 12 if cycle == "annual" else 1
                    amount = PLAN_PRICE[plan] * seats * (10 if cycle == "annual" else 1)   # annual = 2 months free
                    issued = today - timedelta(days=30 * k * months)
                    inv_status = "failed" if (i == 5 and k == 0) else "paid"
                    self.db.execute("INSERT INTO invoices VALUES (?,?,?,?,?)",
                                    (f"INV-{acc_id[4:]}-{k + 1}", acc_id, issued.isoformat(), round(amount, 2),
                                     inv_status))

    # -- reads -----------------------------------------------------------------------------------------------------
    def get_account(self, email: str) -> dict | None:
        row = self.db.execute("SELECT * FROM accounts WHERE lower(email) = lower(?)", (email,)).fetchone()
        return dict(row) if row else None

    def list_invoices(self, account_id: str) -> list[dict]:
        rows = self.db.execute("SELECT * FROM invoices WHERE account_id = ? ORDER BY issued_on DESC", (account_id,))
        return [dict(r) for r in rows]

    def tickets(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM tickets ORDER BY created_at")]

    def escalations(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM escalations ORDER BY created_at")]

    # -- writes (idempotent) ---------------------------------------------------------------------------------------
    def create_ticket(self, subject: str, summary: str, queue: str, priority: str, account_email: str | None,
                      idempotency_key: str) -> dict:
        with self._lock, self.db:
            prev = self.db.execute("SELECT * FROM tickets WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
            if prev:
                return {**dict(prev), "replayed": True}
            n = self.db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
            row = {"id": f"TCK-{n + 1:05d}", "account_email": account_email, "subject": subject, "summary": summary,
                   "queue": queue, "priority": priority, "status": "open", "created_at": _now(),
                   "idempotency_key": idempotency_key}
            self.db.execute("INSERT INTO tickets VALUES (:id,:account_email,:subject,:summary,:queue,:priority,"
                            ":status,:created_at,:idempotency_key)", row)
            return {**row, "replayed": False}

    def escalate(self, ticket_id: str, reason: str, queue: str, idempotency_key: str) -> dict:
        with self._lock, self.db:
            prev = self.db.execute("SELECT * FROM escalations WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
            if prev:
                return {**dict(prev), "replayed": True}
            if not self.db.execute("SELECT 1 FROM tickets WHERE id = ?", (ticket_id,)).fetchone():
                raise KeyError(f"unknown ticket {ticket_id}")
            n = self.db.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
            row = {"id": f"ESC-{n + 1:05d}", "ticket_id": ticket_id, "reason": reason,
                   "team": ESCALATION_TEAMS.get(queue, "support-l2"), "created_at": _now(),
                   "idempotency_key": idempotency_key}
            self.db.execute("INSERT INTO escalations VALUES (:id,:ticket_id,:reason,:team,:created_at,"
                            ":idempotency_key)", row)
            self.db.execute("UPDATE tickets SET status = 'escalated' WHERE id = ?", (ticket_id,))
            return {**row, "replayed": False}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    print(json.dumps(CRM().get_account("admin@northwind-analytics.example.com"), indent=2))
