from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from app.core.config import Settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditService:
    def __init__(self, settings: Settings):
        self.settings = settings
        with sqlite3.connect(self.settings.database_path) as conn:
            self._ensure_database(conn)

    def _ensure_database(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS platform_audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS platform_audit_events_created_at_idx
            ON platform_audit_events(created_at DESC)
            """
        )

    def record(
        self,
        *,
        action: str,
        resource_type: str,
        resource_id: str,
        status: str = "success",
        actor: str = "local",
        payload: dict | None = None,
    ) -> dict:
        with sqlite3.connect(self.settings.database_path) as conn:
            self._ensure_database(conn)
            cursor = conn.execute(
                """
                INSERT INTO platform_audit_events (
                    actor, action, resource_type, resource_id, status, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    actor,
                    action,
                    resource_type,
                    resource_id,
                    status,
                    json.dumps(payload or {}, ensure_ascii=False),
                    _now(),
                ),
            )
            row_id = int(cursor.lastrowid)
        return self.get(row_id)

    def get(self, event_id: int) -> dict:
        with sqlite3.connect(self.settings.database_path) as conn:
            self._ensure_database(conn)
            row = conn.execute(
                """
                SELECT id, actor, action, resource_type, resource_id, status, payload_json, created_at
                FROM platform_audit_events
                WHERE id = ?
                """,
                (event_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown audit event: {event_id}")
        return self._row_to_dict(row)

    def list(
        self,
        *,
        resource_type: str | None = None,
        action: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        query = """
            SELECT id, actor, action, resource_type, resource_id, status, payload_json, created_at
            FROM platform_audit_events
        """
        clauses = []
        params: list[object] = []
        if resource_type:
            clauses.append("resource_type = ?")
            params.append(resource_type)
        if action:
            clauses.append("action = ?")
            params.append(action)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, min(int(limit or 50), 500)))
        with sqlite3.connect(self.settings.database_path) as conn:
            self._ensure_database(conn)
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def clear_verification_events(self) -> dict:
        with sqlite3.connect(self.settings.database_path) as conn:
            self._ensure_database(conn)
            cursor = conn.execute(
                """
                DELETE FROM platform_audit_events
                WHERE json_extract(payload_json, '$.verification') = 1
                   OR json_extract(payload_json, '$.verification') = true
                """
            )
            deleted = int(cursor.rowcount or 0)
        return {"deleted": deleted}

    def _row_to_dict(self, row: tuple) -> dict:
        payload_raw = row[6] or "{}"
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = {"raw": payload_raw}
        return {
            "id": row[0],
            "actor": row[1],
            "action": row[2],
            "resource_type": row[3],
            "resource_id": row[4],
            "status": row[5],
            "payload": payload,
            "created_at": row[7],
        }
