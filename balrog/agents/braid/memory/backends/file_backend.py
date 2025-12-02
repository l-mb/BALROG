import json
import uuid
from pathlib import Path
from typing import Any

from ..interface import MemoryBackend, MemoryEntry, MemoryScope


class FileMemoryBackend(MemoryBackend):
    """JSON file-based memory storage.

    Never deletes entries - uses soft-delete flag for debugging memory evolution.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

    def _entry_from_data(self, entry_id: str, data: dict[str, Any]) -> MemoryEntry:
        scope_str = data.get("scope", "persistent")
        scope = MemoryScope(scope_str) if scope_str else MemoryScope.PERSISTENT
        return MemoryEntry(
            tags=str(data["tags"]),
            content=str(data["content"]),
            scope=scope,
            priority=int(data.get("priority", 5)),
            source_episode=int(data["source_episode"]) if data.get("source_episode") else None,
            deleted=bool(data.get("deleted", False)),
            entry_id=entry_id,
        )

    def _matches_filter(
        self,
        data: dict[str, Any],
        tags: str | set[str] | None,
        scope: MemoryScope | None,
        episode: int | None,
    ) -> bool:
        """Check if entry data matches the given filters."""
        if data.get("deleted", False):
            return False

        # Tag filtering
        if tags is not None:
            entry_tags = {t.strip() for t in str(data["tags"]).split(",")}
            if isinstance(tags, str):
                if tags not in entry_tags:
                    return False
            else:  # set[str]
                if not entry_tags & tags:  # no intersection
                    return False

        # Scope filtering
        data_scope = data.get("scope", "persistent")
        if scope is not None and data_scope != scope.value:
            return False

        # Episode filtering
        if episode is not None:
            if data_scope == "episode" and data.get("source_episode") != episode:
                return False

        return True

    def store(self, entry: MemoryEntry) -> str:
        entry_id = str(uuid.uuid4())[:8]
        self._data[entry_id] = {
            "tags": entry.tags,
            "content": entry.content,
            "scope": entry.scope.value,
            "priority": entry.priority,
            "source_episode": entry.source_episode,
            "deleted": entry.deleted,
        }
        self._save()
        return entry_id

    def retrieve(
        self,
        tags: str | set[str] | None = None,
        scope: MemoryScope | None = None,
        episode: int | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        # Collect all matching entries with their priority for sorting
        matches: list[tuple[int, str, dict[str, Any]]] = []
        for entry_id, data in self._data.items():
            if self._matches_filter(data, tags, scope, episode):
                prio = int(data.get("priority", 5))
                matches.append((prio, entry_id, data))
        # Sort by priority descending, take top N
        matches.sort(key=lambda x: x[0], reverse=True)
        return [self._entry_from_data(eid, d) for _, eid, d in matches[:limit]]

    def count(
        self,
        tags: str | set[str] | None = None,
        scope: MemoryScope | None = None,
        episode: int | None = None,
    ) -> int:
        """Count matching entries without loading them."""
        return sum(1 for data in self._data.values() if self._matches_filter(data, tags, scope, episode))

    def count_by_tag(
        self,
        scope: MemoryScope | None = None,
        episode: int | None = None,
    ) -> dict[str, int]:
        """Count entries per tag, optionally filtered by scope/episode."""
        counts: dict[str, int] = {}
        for data in self._data.values():
            if not self._matches_filter(data, None, scope, episode):
                continue
            for tag in str(data["tags"]).split(","):
                tag = tag.strip()
                if tag:
                    counts[tag] = counts.get(tag, 0) + 1
        return counts

    def remove(self, entry_id: str) -> bool:
        """Soft-delete: marks entry as deleted but keeps it for debugging."""
        if entry_id in self._data:
            self._data[entry_id]["deleted"] = True
            self._save()
            return True
        return False

    def remove_by_episode(self, episode: int) -> int:
        """Soft-delete all episode-scoped entries for given episode."""
        count = 0
        for data in self._data.values():
            if (
                data.get("scope") == "episode"
                and data.get("source_episode") == episode
                and not data.get("deleted", False)
            ):
                data["deleted"] = True
                count += 1
        if count:
            self._save()
        return count

    def update(self, entry_id: str, new_content: str) -> bool:
        if entry_id in self._data:
            self._data[entry_id]["content"] = new_content
            self._save()
            return True
        return False

    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Simple keyword search. Future: semantic search with embeddings."""
        query_lower = query.lower()
        matches = []
        for entry_id, data in self._data.items():
            if data.get("deleted", False):
                continue
            if query_lower in str(data["content"]).lower():
                matches.append(self._entry_from_data(entry_id, data))
                if len(matches) >= limit:
                    break
        return matches

    def max_episode(self) -> int:
        """Return the highest source_episode in storage, or 0 if empty."""
        return max(
            (int(data["source_episode"]) for data in self._data.values() if data.get("source_episode")),
            default=0,
        )

    def all_tags(self, scope: MemoryScope | None = None, episode: int | None = None) -> set[str]:
        """Return all unique tags from non-deleted entries, optionally filtered by scope/episode."""
        tags: set[str] = set()
        for data in self._data.values():
            if not self._matches_filter(data, None, scope, episode):
                continue
            for tag in str(data["tags"]).split(","):
                tag = tag.strip()
                if tag:
                    tags.add(tag)
        return tags
