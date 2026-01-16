import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class StateManager:
    """
    Achilles State Manager (SQLite Edition).
    Provides "Risk & State" compliance via:
    1. Persistent Key-Value Store (State).
    2. Immutable Audit Log (Audit Trail).
    """
    def __init__(self, db_path="achilles_state.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Table 1: State (Key-Value)
        c.execute('''CREATE TABLE IF NOT EXISTS state
                     (key TEXT PRIMARY KEY, value TEXT, updated_at TIMESTAMP)''')
        
        # Table 2: Audit Log (Immutable Events)
        c.execute('''CREATE TABLE IF NOT EXISTS audit_log
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      event_type TEXT, 
                      payload TEXT, 
                      timestamp TIMESTAMP)''')
        
        conn.commit()
        conn.close()

    def set_state(self, key: str, value: Any):
        """Persist a state value (JSON serialized)."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        json_val = json.dumps(value)
        timestamp = datetime.now()
        
        c.execute("INSERT OR REPLACE INTO state (key, value, updated_at) VALUES (?, ?, ?)",
                  (key, json_val, timestamp))
        
        conn.commit()
        conn.close()

    def get_state(self, key: str, default: Any = None) -> Any:
        """Retrieve a state value."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT value FROM state WHERE key = ?", (key,))
        row = c.fetchone()
        conn.close()
        
        if row:
            try:
                return json.loads(row[0])
            except:
                return row[0]
        return default

    def log_event(self, event_type: str, payload: Dict[str, Any]):
        """
        [TAG: R3K_AUDIT_TRAIL]
        Log an event to the immutable audit log.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        json_payload = json.dumps(payload)
        timestamp = datetime.now()
        
        c.execute("INSERT INTO audit_log (event_type, payload, timestamp) VALUES (?, ?, ?)",
                  (event_type, json_payload, timestamp))
        
        conn.commit()
        conn.close()
        print(f"[AUDIT] {event_type} logged at {timestamp}")

    def get_audit_trail(self, limit: int = 100):
        """Retrieve recent audit logs."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        return rows
