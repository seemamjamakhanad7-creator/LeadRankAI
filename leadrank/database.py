import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "leadrank.db"

def connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with connect() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            company TEXT NOT NULL,
            industry TEXT NOT NULL,
            company_size TEXT NOT NULL,
            budget REAL NOT NULL,
            website_visits INTEGER NOT NULL,
            email_opens INTEGER NOT NULL,
            demo_requested INTEGER NOT NULL,
            source TEXT NOT NULL,
            score REAL NOT NULL,
            probability REAL NOT NULL,
            priority TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            assigned_to TEXT NOT NULL DEFAULT 'Unassigned',
            status TEXT NOT NULL DEFAULT 'New',
            approved INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_id INTEGER,
            action TEXT NOT NULL,
            details TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """)

def add_lead(data):
    fields = ("name","company","industry","company_size","budget","website_visits",
              "email_opens","demo_requested","source","score","probability",
              "priority","recommendation","assigned_to","status","approved")
    vals = [data.get(k) for k in fields]
    with connect() as con:
        cur = con.execute(
            f"INSERT INTO leads ({','.join(fields)}) VALUES ({','.join(['?']*len(fields))})",
            vals
        )
        lead_id = cur.lastrowid
        con.execute("INSERT INTO activity_log(lead_id,action,details) VALUES(?,?,?)",
                    (lead_id, "Lead created", f"Initial score: {data['score']:.1f}"))
        return lead_id

def get_leads():
    with connect() as con:
        return pd.read_sql_query("SELECT * FROM leads ORDER BY created_at DESC, id DESC", con)

def update_lead(lead_id, **kwargs):
    allowed = {"assigned_to", "status", "approved"}
    updates = {k:v for k,v in kwargs.items() if k in allowed}
    if not updates:
        return
    with connect() as con:
        con.execute(
            "UPDATE leads SET " + ", ".join(f"{k}=?" for k in updates) + " WHERE id=?",
            list(updates.values()) + [lead_id]
        )
        con.execute("INSERT INTO activity_log(lead_id,action,details) VALUES(?,?,?)",
                    (lead_id, "Lead updated", str(updates)))

def log_event(lead_id, action, details=""):
    with connect() as con:
        con.execute("INSERT INTO activity_log(lead_id,action,details) VALUES(?,?,?)",
                    (lead_id, action, details))

def get_activity(limit=100):
    with connect() as con:
        return pd.read_sql_query(
            "SELECT * FROM activity_log ORDER BY created_at DESC LIMIT ?",
            con, params=(limit,)
        )
