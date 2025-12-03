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
        self.conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    # -------------------------------------------------------------------------
    # Memory methods
    # -------------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> str:
        entry_id = str(uuid.uuid4())[:8]
        self.conn.execute(
            """INSERT INTO memory (id, tags, content, scope, priority, source_episode, deleted, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry_id,
                entry.tags,
                entry.content,
                entry.scope.value,
                entry.priority,
                entry.source_episode,
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

    def remove(self, entry_id: str) -> bool:
        cursor = self.conn.execute(
            "UPDATE memory SET deleted = 1 WHERE id = ? AND deleted = 0", (entry_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

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

    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        query_lower = query.lower()
        rows = self.conn.execute(
            "SELECT * FROM memory WHERE deleted = 0 AND LOWER(content) LIKE ? LIMIT ?",
            (f"%{query_lower}%", limit),
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def max_episode(self) -> int:
        row = self.conn.execute(
            "SELECT MAX(source_episode) FROM memory WHERE source_episode IS NOT NULL"
        ).fetchone()
        return row[0] if row[0] is not None else 0

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

    def log_reset(self, episode: int) -> None:
        self._log(episode, 0, "reset")

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

    def log_screen(self, episode: int, step: int, screen: str) -> None:
        self._log(episode, step, "screen", {"screen": screen})

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
    ) -> None:
        data: dict[str, Any] = {
            "action": action,
            "latency_ms": latency_ms,
            "in_tok": input_tokens,
            "out_tok": output_tokens,
            "ep_in_tok": ep_input_tokens,
            "ep_out_tok": ep_output_tokens,
            "total_in_tok": total_input_tokens,
            "total_out_tok": total_output_tokens,
        }
        if reasoning:
            data["reasoning"] = reasoning
        self._log(episode, step, "response", data)

    def log_memory_update(
        self,
        episode: int,
        step: int,
        adds: int = 0,
        removes: int = 0,
        label_changes: bool = False,
        added_entries: list[dict[str, Any]] | None = None,
        removed_ids: list[str] | None = None,
    ) -> None:
        if not (adds or removes or label_changes):
            return
        data: dict[str, Any] = {
            "adds": adds,
            "removes": removes,
            "label_changes": label_changes,
        }
        if added_entries:
            data["added"] = added_entries
        if removed_ids:
            data["removed"] = removed_ids
        self._log(episode, step, "memory", data)

    def log_error(self, episode: int, step: int, error: str) -> None:
        self._log(episode, step, "error", {"error": error})
