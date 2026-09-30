import json
import aiosqlite
from datetime import datetime
from typing import List, Dict, Any, Optional
from config import DB_PATH

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                iteration INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                result_summary TEXT,
                workspace_dir TEXT,
                subtasks_json TEXT DEFAULT '[]'
            )
        """)
        try:
            await db.execute("ALTER TABLE tasks ADD COLUMN subtasks_json TEXT DEFAULT '[]'")
        except Exception:
            pass
        await db.execute("""
            CREATE TABLE IF NOT EXISTS task_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                sender TEXT NOT NULL,
                event_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        await db.commit()

async def create_task(title: str, description: str, workspace_dir: str) -> int:
    now = datetime.now().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            INSERT INTO tasks (title, description, status, iteration, created_at, workspace_dir)
            VALUES (?, ?, 'queued', 0, ?, ?)
        """, (title, description, now, workspace_dir))
        await db.commit()
        return cursor.lastrowid

async def get_task(task_id: int) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            try:
                d["subtasks"] = json.loads(d.get("subtasks_json") or "[]")
            except Exception:
                d["subtasks"] = []
            return d

async def get_all_tasks() -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM tasks ORDER BY id DESC") as cursor:
            rows = await cursor.fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["subtasks"] = json.loads(d.get("subtasks_json") or "[]")
                except Exception:
                    d["subtasks"] = []
                result.append(d)
            return result

async def update_task_subtasks(task_id: int, subtasks: List[Dict[str, Any]]):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE tasks SET subtasks_json = ? WHERE id = ?", (json.dumps(subtasks), task_id))
        await db.commit()

async def get_next_queued_task() -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM tasks WHERE status = 'queued' ORDER BY id ASC LIMIT 1") as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

async def update_task_status(task_id: int, status: str, result_summary: Optional[str] = None, iteration: Optional[int] = None):
    now = datetime.now().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        updates = ["status = ?"]
        params = [status]
        
        if status == 'running':
            updates.append("started_at = COALESCE(started_at, ?)")
            params.append(now)
        elif status in ('completed', 'failed'):
            updates.append("completed_at = ?")
            params.append(now)
            
        if result_summary is not None:
            updates.append("result_summary = ?")
            params.append(result_summary)
            
        if iteration is not None:
            updates.append("iteration = ?")
            params.append(iteration)
            
        params.append(task_id)
        query = f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?"
        await db.execute(query, params)
        await db.commit()

async def add_task_event(task_id: int, sender: str, event_type: str, content: str):
    now = datetime.now().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO task_events (task_id, sender, event_type, content, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (task_id, sender, event_type, content, now))
        await db.commit()

async def get_task_events(task_id: int) -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM task_events WHERE task_id = ? ORDER BY id ASC", (task_id,)) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

async def get_setting(key: str, default: str = "") -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT value FROM settings WHERE key = ?", (key,)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else default

async def set_setting(key: str, value: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
        await db.commit()

async def delete_task(task_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM task_events WHERE task_id = ?", (task_id,))
        await db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        await db.commit()

async def revert_task_events(task_id: int, event_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM task_events WHERE task_id = ? AND id > ?", (task_id, event_id))
        await db.execute("UPDATE tasks SET status = 'needs_review' WHERE id = ?", (task_id,))
        await db.commit()
