"""SQL queries for BRAID monitor."""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentInfo:
    worker_id: str
    episode: int
    step: int


@dataclass
class AgentStats:
    episode: int
    step: int
    total_in_tokens: int
    total_out_tokens: int
    avg_latency_ms: float
    cache_read_tokens: int
    cache_create_tokens: int
    single_actions: int
    multi_actions: int
    queued_actions: int
    # Memory stats
    episode_mem_adds: int
    episode_mem_removes: int
    persistent_mem_adds: int
    persistent_mem_removes: int
    # Reasoning stats
    think_count: int


@dataclass
class ResponseInfo:
    action: str
    reasoning: str | None
    latency_ms: int


class MonitorDB:
    """Read-only connection to BRAID SQLite database."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA query_only = ON")
        return self._conn

    def get_active_agents(self, minutes: int = 5) -> list[AgentInfo]:
        """Get agents with activity in the last N minutes."""
        rows = self.conn.execute(
            """SELECT worker_id, MAX(episode) as ep, MAX(step) as step
               FROM journal
               WHERE timestamp > datetime('now', ?)
               GROUP BY worker_id
               ORDER BY worker_id""",
            (f"-{minutes} minutes",),
        ).fetchall()
        return [AgentInfo(r["worker_id"], r["ep"] or 0, r["step"] or 0) for r in rows]

    def get_all_agents(self) -> list[AgentInfo]:
        """Get all agents ever seen."""
        rows = self.conn.execute(
            """SELECT worker_id, MAX(episode) as ep, MAX(step) as step
               FROM journal
               GROUP BY worker_id
               ORDER BY worker_id"""
        ).fetchall()
        return [AgentInfo(r["worker_id"], r["ep"] or 0, r["step"] or 0) for r in rows]

    def get_latest_screen(self, worker_id: str, max_step: int | None = None) -> str | None:
        """Get the latest screen for an agent, optionally up to a specific step."""
        if max_step is not None:
            row = self.conn.execute(
                """SELECT data FROM journal
                   WHERE worker_id = ? AND event = 'screen' AND step <= ?
                   ORDER BY id DESC LIMIT 1""",
                (worker_id, max_step),
            ).fetchone()
        else:
            row = self.conn.execute(
                """SELECT data FROM journal
                   WHERE worker_id = ? AND event = 'screen'
                   ORDER BY id DESC LIMIT 1""",
                (worker_id,),
            ).fetchone()
        if row and row["data"]:
            data = json.loads(row["data"])
            return data.get("screen")
        return None

    def get_latest_response(self, worker_id: str) -> ResponseInfo | None:
        """Get the latest response for an agent."""
        row = self.conn.execute(
            """SELECT data FROM journal
               WHERE worker_id = ? AND event = 'response'
               ORDER BY id DESC LIMIT 1""",
            (worker_id,),
        ).fetchone()
        if row and row["data"]:
            data = json.loads(row["data"])
            return ResponseInfo(
                action=data.get("action", ""),
                reasoning=data.get("reasoning"),
                latency_ms=data.get("latency_ms", 0),
            )
        return None

    def get_latest_prompt(self, worker_id: str, max_step: int | None = None) -> dict | None:
        """Get the latest full prompt for an agent."""
        if max_step is not None:
            row = self.conn.execute(
                """SELECT step, data FROM journal
                   WHERE worker_id = ? AND event = 'prompt' AND step <= ?
                   ORDER BY id DESC LIMIT 1""",
                (worker_id, max_step),
            ).fetchone()
        else:
            row = self.conn.execute(
                """SELECT step, data FROM journal
                   WHERE worker_id = ? AND event = 'prompt'
                   ORDER BY id DESC LIMIT 1""",
                (worker_id,),
            ).fetchone()
        if row and row["data"]:
            data = json.loads(row["data"])
            return {"step": row["step"], "messages": data.get("messages", [])}
        return None

    def get_latest_full_response(self, worker_id: str, max_step: int | None = None) -> dict | None:
        """Get the latest full response data for an agent."""
        if max_step is not None:
            row = self.conn.execute(
                """SELECT step, data FROM journal
                   WHERE worker_id = ? AND event = 'response' AND step <= ?
                   ORDER BY id DESC LIMIT 1""",
                (worker_id, max_step),
            ).fetchone()
        else:
            row = self.conn.execute(
                """SELECT step, data FROM journal
                   WHERE worker_id = ? AND event = 'response'
                   ORDER BY id DESC LIMIT 1""",
                (worker_id,),
            ).fetchone()
        if row and row["data"]:
            data = json.loads(row["data"])
            return {
                "step": row["step"],
                "action": data.get("action", ""),
                "reasoning": data.get("reasoning"),
                "extended_thinking": data.get("extended_thinking"),
                "latency_ms": data.get("latency_ms", 0),
                "in_tok": data.get("in_tok", 0),
                "out_tok": data.get("out_tok", 0),
            }
        return None

    def get_recent_responses(
        self, worker_id: str, limit: int = 10, max_step: int | None = None
    ) -> list[dict]:
        """Get recent responses for an agent (newest first), optionally up to a step."""
        if max_step is not None:
            rows = self.conn.execute(
                """SELECT step, data FROM journal
                   WHERE worker_id = ? AND event = 'response' AND step <= ?
                   ORDER BY id DESC LIMIT ?""",
                (worker_id, max_step, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT step, data FROM journal
                   WHERE worker_id = ? AND event = 'response'
                   ORDER BY id DESC LIMIT ?""",
                (worker_id, limit),
            ).fetchall()
        responses = []
        for row in rows:
            if row["data"]:
                data = json.loads(row["data"])
                responses.append({
                    "step": row["step"],
                    "action": data.get("action", ""),
                    "reasoning": data.get("reasoning"),
                    "action_type": data.get("action_type", "single"),
                    "latency_ms": data.get("latency_ms", 0),
                })
        return responses

    def get_max_step(self, worker_id: str) -> int:
        """Get the maximum step number for a worker."""
        row = self.conn.execute(
            "SELECT MAX(step) as max_step FROM journal WHERE worker_id = ?",
            (worker_id,),
        ).fetchone()
        return row["max_step"] or 0

    def get_stats(self, worker_id: str) -> AgentStats:
        """Get aggregate stats for an agent."""
        # Response stats
        resp = self.conn.execute(
            """SELECT MAX(episode) as episode, MAX(step) as step,
                      SUM(json_extract(data, '$.in_tok')) as total_in,
                      SUM(json_extract(data, '$.out_tok')) as total_out,
                      AVG(json_extract(data, '$.latency_ms')) as avg_latency,
                      SUM(json_extract(data, '$.cache_read')) as cache_read,
                      SUM(json_extract(data, '$.cache_create')) as cache_create,
                      SUM(CASE WHEN json_extract(data, '$.action_type') = 'single' THEN 1 ELSE 0 END) as single_ct,
                      SUM(CASE WHEN json_extract(data, '$.action_type') = 'multi' THEN 1 ELSE 0 END) as multi_ct,
                      SUM(CASE WHEN json_extract(data, '$.action_type') = 'queued' THEN 1 ELSE 0 END) as queued_ct,
                      SUM(CASE WHEN json_extract(data, '$.reasoning') IS NOT NULL THEN 1 ELSE 0 END) as think_ct
               FROM journal WHERE worker_id = ? AND event = 'response'""",
            (worker_id,),
        ).fetchone()
        # Memory stats
        mem = self.conn.execute(
            """SELECT SUM(json_extract(data, '$.ep_adds')) as ep_adds,
                      SUM(json_extract(data, '$.ep_removes')) as ep_removes,
                      SUM(json_extract(data, '$.p_adds')) as p_adds,
                      SUM(json_extract(data, '$.p_removes')) as p_removes
               FROM journal WHERE worker_id = ? AND event = 'memory'""",
            (worker_id,),
        ).fetchone()
        return AgentStats(
            episode=resp["episode"] or 0,
            step=resp["step"] or 0,
            total_in_tokens=int(resp["total_in"] or 0),
            total_out_tokens=int(resp["total_out"] or 0),
            avg_latency_ms=float(resp["avg_latency"] or 0),
            cache_read_tokens=int(resp["cache_read"] or 0),
            cache_create_tokens=int(resp["cache_create"] or 0),
            single_actions=int(resp["single_ct"] or 0),
            multi_actions=int(resp["multi_ct"] or 0),
            queued_actions=int(resp["queued_ct"] or 0),
            episode_mem_adds=int(mem["ep_adds"] or 0),
            episode_mem_removes=int(mem["ep_removes"] or 0),
            persistent_mem_adds=int(mem["p_adds"] or 0),
            persistent_mem_removes=int(mem["p_removes"] or 0),
            think_count=int(resp["think_ct"] or 0),
        )

    def get_memory_entries(
        self,
        limit: int = 50,
        scope: str | None = None,
        include_deleted: bool = False,
        episode: int | None = None,
    ) -> list[dict]:
        """Get memory entries ordered by recency.

        Args:
            limit: Max entries to return
            scope: Filter by 'persistent' or 'episode', or None for all
            include_deleted: If True, include soft-deleted entries
            episode: Filter episode-scoped memories to this episode only
        """
        conditions = []
        params: list[str | int] = []

        if not include_deleted:
            conditions.append("deleted = 0")

        if scope == "persistent":
            conditions.append("scope = 'persistent'")
        elif scope == "episode":
            conditions.append("scope = 'episode'")
            if episode is not None:
                conditions.append("source_episode = ?")
                params.append(episode)
        else:
            # "All": show persistent + episode memories for current episode only
            if episode is not None:
                conditions.append("(scope = 'persistent' OR (scope = 'episode' AND source_episode = ?))")
                params.append(episode)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = self.conn.execute(
            f"""SELECT id, tags, content, scope, priority, source_episode, source_step, created_at, deleted
               FROM memory {where}
               ORDER BY created_at DESC
               LIMIT ?""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]
