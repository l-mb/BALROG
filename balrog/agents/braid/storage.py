"""Unified SQLite storage for BRAID memory and journal.

Replaces separate JSON file memory backend and JSONL journal.
Enables multi-worker support via WAL mode.
"""

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class MemoryScope(Enum):
    EPISODE = "episode"
    PERSISTENT = "persistent"


@dataclass
class MemoryEntry:
    """Single memory entry, either episode-scoped or persistent."""

    tags: str
    content: str
    scope: MemoryScope = MemoryScope.PERSISTENT
    priority: int = 5
    source_episode: int | None = None
    source_step: int | None = None
    deleted: bool = False
    entry_id: str = ""
    created_at: str = ""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id TEXT PRIMARY KEY,
    tags TEXT NOT NULL,
    content TEXT NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('episode', 'persistent')),
    priority INTEGER NOT NULL DEFAULT 5 CHECK (priority BETWEEN 1 AND 9),
    source_episode INTEGER,
    source_step INTEGER,
    deleted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory(scope, deleted);
CREATE INDEX IF NOT EXISTS idx_memory_episode ON memory(source_episode) WHERE scope = 'episode';
CREATE INDEX IF NOT EXISTS idx_memory_priority ON memory(priority DESC) WHERE deleted = 0;

CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    episode INTEGER NOT NULL,
    step INTEGER NOT NULL,
    event TEXT NOT NULL,
    data TEXT
);

CREATE INDEX IF NOT EXISTS idx_journal_episode ON journal(episode, step);
CREATE INDEX IF NOT EXISTS idx_journal_worker ON journal(worker_id, episode);
CREATE INDEX IF NOT EXISTS idx_journal_event ON journal(event);

CREATE TABLE IF NOT EXISTS visited (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL,
    episode INTEGER NOT NULL,
    dungeon_num INTEGER NOT NULL DEFAULT 0,
    dlvl INTEGER NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    first_step INTEGER NOT NULL,
    UNIQUE(worker_id, episode, dungeon_num, dlvl, x, y)
);

CREATE INDEX IF NOT EXISTS idx_visited_lookup ON visited(worker_id, episode, dungeon_num, dlvl);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    episode INTEGER NOT NULL,
    step INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    args TEXT,
    result TEXT,
    latency_ms INTEGER,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_step ON tool_calls(worker_id, episode, step);

CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL,
    episode INTEGER NOT NULL,
    step INTEGER NOT NULL,
    content TEXT NOT NULL,
    active_form TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'in_progress', 'completed')),
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_todos_episode ON todos(worker_id, episode);
"""


class BraidStorage:
    """Unified SQLite storage for BRAID memory and journal."""

    def __init__(self, db_path: Path | str, worker_id: str | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.worker_id = worker_id or f"w{os.getpid()}"
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(_SCHEMA)
        # Migrate existing DBs: add source_step column if missing
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(memory)").fetchall()}
        if "source_step" not in cols:
            self.conn.execute("ALTER TABLE memory ADD COLUMN source_step INTEGER")
        self.conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    # -------------------------------------------------------------------------
    # Memory methods
    # -------------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> str:
        entry_id = str(uuid.uuid4())[:8]
        self.conn.execute(
            """INSERT INTO memory (id, tags, content, scope, priority, source_episode, source_step, deleted, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry_id,
                entry.tags,
                entry.content,
                entry.scope.value,
                entry.priority,
                entry.source_episode,
                entry.source_step,
                int(entry.deleted),
                self._now(),
            ),
        )
        self.conn.commit()
        return entry_id

    def retrieve(
        self,
        tags: str | set[str] | None = None,
        scope: MemoryScope | None = None,
        episode: int | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        query = "SELECT * FROM memory WHERE deleted = 0"
        params: list[Any] = []

        if scope is not None:
            query += " AND scope = ?"
            params.append(scope.value)

        if episode is not None:
            query += " AND (scope != 'episode' OR source_episode = ?)"
            params.append(episode)

        query += " ORDER BY priority DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()

        # Filter by tags in Python (SQLite doesn't have good array support)
        entries = []
        for row in rows:
            if tags is not None:
                entry_tags = {t.strip() for t in row["tags"].split(",")}
                if isinstance(tags, str):
                    if tags not in entry_tags:
                        continue
                elif not entry_tags & tags:
                    continue
            entries.append(self._row_to_entry(row))
            if len(entries) >= limit:
                break

        return entries

    def count(
        self,
        tags: str | set[str] | None = None,
        scope: MemoryScope | None = None,
        episode: int | None = None,
    ) -> int:
        if tags is not None:
            # Need to filter in Python for tag matching
            return len(self.retrieve(tags=tags, scope=scope, episode=episode, limit=10000))

        query = "SELECT COUNT(*) FROM memory WHERE deleted = 0"
        params: list[Any] = []

        if scope is not None:
            query += " AND scope = ?"
            params.append(scope.value)

        if episode is not None:
            query += " AND (scope != 'episode' OR source_episode = ?)"
            params.append(episode)

        return self.conn.execute(query, params).fetchone()[0]

    def count_by_tag(
        self,
        scope: MemoryScope | None = None,
        episode: int | None = None,
    ) -> dict[str, int]:
        entries = self.retrieve(scope=scope, episode=episode, limit=10000)
        counts: dict[str, int] = {}
        for e in entries:
            for tag in e.tags.split(","):
                tag = tag.strip()
                if tag:
                    counts[tag] = counts.get(tag, 0) + 1
        return counts

    def remove(self, entry_id: str) -> str | None:
        """Remove entry by ID. Returns scope ('episode'/'persistent') if found, None otherwise."""
        row = self.conn.execute(
            "SELECT scope FROM memory WHERE id = ? AND deleted = 0", (entry_id,)
        ).fetchone()
        if not row:
            return None
        self.conn.execute(
            "UPDATE memory SET deleted = 1 WHERE id = ? AND deleted = 0", (entry_id,)
        )
        self.conn.commit()
        return row["scope"]

    def remove_by_episode(self, episode: int) -> int:
        cursor = self.conn.execute(
            "UPDATE memory SET deleted = 1 WHERE scope = 'episode' AND source_episode = ? AND deleted = 0",
            (episode,),
        )
        self.conn.commit()
        return cursor.rowcount

    def update(self, entry_id: str, new_content: str) -> bool:
        cursor = self.conn.execute(
            "UPDATE memory SET content = ? WHERE id = ? AND deleted = 0",
            (new_content, entry_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def search(
        self,
        query: str,
        tags: set[str] | None = None,
        scope: MemoryScope | None = None,
        episode: int | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search memory by content substring and optional filters."""
        query_lower = query.lower() if query else ""
        conditions = ["deleted = 0"]
        params: list[str | int] = []

        if query_lower:
            conditions.append("LOWER(content) LIKE ?")
            params.append(f"%{query_lower}%")

        if scope is not None:
            conditions.append("scope = ?")
            params.append(scope.value)

        if episode is not None:
            conditions.append("(scope != 'episode' OR source_episode = ?)")
            params.append(episode)

        sql = f"SELECT * FROM memory WHERE {' AND '.join(conditions)} ORDER BY priority DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        entries = [self._row_to_entry(row) for row in rows]

        # Filter by tags in Python (SQLite LIKE on comma-separated is unreliable)
        if tags:
            filtered = []
            for e in entries:
                entry_tags = {t.strip() for t in e.tags.split(",") if t.strip()}
                if tags & entry_tags:  # Any tag matches
                    filtered.append(e)
            return filtered

        return entries

    def max_episode(self) -> int:
        """Get highest episode number across all tables globally.

        Episode numbers must be unique across all workers to ensure
        visited positions don't collide between different runs.
        """
        max_vals = []
        for query in [
            "SELECT MAX(source_episode) FROM memory WHERE source_episode IS NOT NULL",
            "SELECT MAX(episode) FROM journal",  # Global, not per-worker
            "SELECT MAX(episode) FROM visited",  # Global, not per-worker
            "SELECT MAX(episode) FROM tool_calls",
            "SELECT MAX(episode) FROM todos",
        ]:
            row = self.conn.execute(query).fetchone()
            if row[0] is not None:
                max_vals.append(row[0])
        return max(max_vals) if max_vals else 0

    def all_tags(self, scope: MemoryScope | None = None, episode: int | None = None) -> set[str]:
        entries = self.retrieve(scope=scope, episode=episode, limit=10000)
        tags: set[str] = set()
        for e in entries:
            for tag in e.tags.split(","):
                tag = tag.strip()
                if tag:
                    tags.add(tag)
        return tags

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            tags=row["tags"],
            content=row["content"],
            scope=MemoryScope(row["scope"]),
            priority=row["priority"],
            source_episode=row["source_episode"],
            source_step=row["source_step"],
            deleted=bool(row["deleted"]),
            entry_id=row["id"],
            created_at=row["created_at"],
        )

    # -------------------------------------------------------------------------
    # Journal methods
    # -------------------------------------------------------------------------

    def _log(self, episode: int, step: int, event: str, data: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO journal (timestamp, worker_id, episode, step, event, data) VALUES (?, ?, ?, ?, ?, ?)",
            (self._now(), self.worker_id, episode, step, event, json.dumps(data) if data else None),
        )
        self.conn.commit()

    def log_reset(self, episode: int, model_id: str | None = None) -> None:
        data = {"model_id": model_id} if model_id else None
        self._log(episode, 0, "reset", data)

    def log_request(
        self,
        episode: int,
        step: int,
        prompt_msgs: int,
        prompt_chars: int,
        observation_text: str | None = None,
        full_prompt: list[dict[str, str]] | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "prompt_msgs": prompt_msgs,
            "prompt_chars": prompt_chars,
            "prompt_tokens_est": prompt_chars // 4,
        }
        if observation_text:
            data["obs"] = observation_text
        self._log(episode, step, "request", data)

        if full_prompt:
            self._log(episode, step, "prompt", {"messages": full_prompt})

    def log_sdk_prompt(
        self,
        episode: int,
        step: int,
        sent_content: str,
        received_content: str,
        conversation_history: list[dict[str, str]],
    ) -> None:
        """Log SDK incremental prompt (what was actually sent/received this turn).

        Args:
            episode: Current episode number.
            step: Current step number.
            sent_content: The message content sent to the SDK this turn.
            received_content: The response received from the SDK this turn.
            conversation_history: Full conversation history (newest first for display).
        """
        self._log(episode, step, "sdk_prompt", {
            "sent": sent_content,
            "received": received_content,
            "history": conversation_history,  # Already newest-first from client
        })

    def log_screen(
        self, episode: int, step: int, screen: str, dlvl: int | None = None
    ) -> None:
        data: dict[str, Any] = {"screen": screen}
        if dlvl is not None:
            data["dlvl"] = dlvl
        self._log(episode, step, "screen", data)

    def log_response(
        self,
        episode: int,
        step: int,
        action: str,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        ep_input_tokens: int,
        ep_output_tokens: int,
        total_input_tokens: int,
        total_output_tokens: int,
        reasoning: str | None = None,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        extended_thinking: str | None = None,
        action_type: str = "single",  # "single", "multi", or "queued"
        compute_requests: list[str] | None = None,
        raw_completion: str | None = None,
        ep_llm_calls: int = 0,
        total_llm_calls: int = 0,
    ) -> None:
        data: dict[str, Any] = {
            "action": action,
            "action_type": action_type,
            "latency_ms": latency_ms,
            "in_tok": input_tokens,
            "out_tok": output_tokens,
            "ep_in_tok": ep_input_tokens,
            "ep_out_tok": ep_output_tokens,
            "total_in_tok": total_input_tokens,
            "total_out_tok": total_output_tokens,
            "ep_llm_calls": ep_llm_calls,
            "total_llm_calls": total_llm_calls,
        }
        if reasoning:
            data["reasoning"] = reasoning
        if cache_creation_tokens or cache_read_tokens:
            data["cache_create"] = cache_creation_tokens
            data["cache_read"] = cache_read_tokens
        if extended_thinking:
            data["extended_thinking"] = extended_thinking
        if compute_requests:
            data["compute_requests"] = compute_requests
        if raw_completion:
            data["raw_completion"] = raw_completion
        self._log(episode, step, "response", data)

    def log_memory_update(
        self,
        episode: int,
        step: int,
        adds: int = 0,
        removes: int = 0,
        tag_changes: bool = False,
        added_entries: list[dict[str, Any]] | None = None,
        removed_ids: list[str] | None = None,
        episode_adds: int = 0,
        persistent_adds: int = 0,
        episode_removes: int = 0,
        persistent_removes: int = 0,
    ) -> None:
        if not (adds or removes or tag_changes):
            return
        data: dict[str, Any] = {
            "adds": adds,
            "removes": removes,
            "tag_changes": tag_changes,
            "ep_adds": episode_adds,
            "p_adds": persistent_adds,
            "ep_removes": episode_removes,
            "p_removes": persistent_removes,
        }
        if added_entries:
            data["added"] = added_entries
        if removed_ids:
            data["removed"] = removed_ids
        self._log(episode, step, "memory", data)

    def log_error(self, episode: int, step: int, error: str) -> None:
        self._log(episode, step, "error", {"error": error})

    def log_compute(
        self,
        episode: int,
        step: int,
        requests: list[str],
        results: list[str],
    ) -> None:
        """Log compute helper requests and results."""
        if not requests:
            return
        self._log(episode, step, "compute", {"requests": requests, "results": results})

    def log_position(
        self, episode: int, step: int, dungeon_num: int, dlvl: int, x: int, y: int
    ) -> None:
        """Log visited position (upsert - only stores first visit)."""
        self.conn.execute(
            """INSERT OR IGNORE INTO visited (worker_id, episode, dungeon_num, dlvl, x, y, first_step)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (self.worker_id, episode, dungeon_num, dlvl, x, y, step),
        )
        self.conn.commit()

    def get_visited_for_level(self, episode: int, dungeon_num: int, dlvl: int) -> set[tuple[int, int]]:
        """Get all visited (x, y) positions for current episode/level/branch."""
        rows = self.conn.execute(
            """SELECT x, y FROM visited
               WHERE worker_id = ? AND episode = ? AND dungeon_num = ? AND dlvl = ?""",
            (self.worker_id, episode, dungeon_num, dlvl),
        ).fetchall()
        return {(row[0], row[1]) for row in rows}

    # -------------------------------------------------------------------------
    # Tool call logging
    # -------------------------------------------------------------------------

    def log_tool_call(
        self,
        episode: int,
        step: int,
        tool_name: str,
        args: str | None,
        result: str | None,
        error: str | None = None,
    ) -> None:
        """Log a tool invocation for debugging and visualization."""
        self.conn.execute(
            """INSERT INTO tool_calls (timestamp, worker_id, episode, step, tool_name, args, result, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (self._now(), self.worker_id, episode, step, tool_name, args, result, error),
        )
        self.conn.commit()

    def get_recent_tool_calls(
        self, episode: int, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get recent tool calls for an episode."""
        rows = self.conn.execute(
            """SELECT id, timestamp, step, tool_name, args, result, error
               FROM tool_calls
               WHERE worker_id = ? AND episode = ?
               ORDER BY id DESC LIMIT ?""",
            (self.worker_id, episode, limit),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "step": r["step"],
                "tool_name": r["tool_name"],
                "args": json.loads(r["args"]) if r["args"] else None,
                "result": r["result"],
                "error": r["error"],
            }
            for r in rows
        ]

    # -------------------------------------------------------------------------
    # Todo tracking
    # -------------------------------------------------------------------------

    def save_todos(
        self, episode: int, step: int, todos: list[dict[str, Any]]
    ) -> None:
        """Save current todo state (replaces all todos for this episode)."""
        # Clear existing todos for this episode
        self.conn.execute(
            "DELETE FROM todos WHERE worker_id = ? AND episode = ?",
            (self.worker_id, episode),
        )
        # Insert new todos
        now = self._now()
        for todo in todos:
            self.conn.execute(
                """INSERT INTO todos (worker_id, episode, step, content, active_form, status, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.worker_id,
                    episode,
                    step,
                    todo.get("content", ""),
                    todo.get("activeForm", ""),
                    todo.get("status", "pending"),
                    now,
                ),
            )
        self.conn.commit()

    def get_todos(self, episode: int) -> list[dict[str, Any]]:
        """Get todos for an episode."""
        rows = self.conn.execute(
            """SELECT content, active_form, status, step, updated_at
               FROM todos
               WHERE worker_id = ? AND episode = ?
               ORDER BY id ASC""",
            (self.worker_id, episode),
        ).fetchall()
        return [
            {
                "content": r["content"],
                "activeForm": r["active_form"],
                "status": r["status"],
                "step": r["step"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]
