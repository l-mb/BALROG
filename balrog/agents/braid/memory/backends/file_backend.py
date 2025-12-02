import json
import uuid
from pathlib import Path
from typing import Any

from ..interface import MemoryBackend, MemoryEntry


class FileMemoryBackend(MemoryBackend):
    """JSON file-based persistent memory storage.

    Simple implementation for initial development.
    Interface supports future swap to postgres/FAISS.
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

    def store(self, entry: MemoryEntry) -> str:
        entry_id = str(uuid.uuid4())[:8]
        self._data[entry_id] = {
            "category": entry.category,
            "content": entry.content,
            "confidence": entry.confidence,
            "source_episode": entry.source_episode,
        }
        self._save()
        return entry_id

    def retrieve(self, category: str | None = None, limit: int = 20) -> list[MemoryEntry]:
        entries = []
        for data in self._data.values():
            if category is None or data["category"] == category:
                entries.append(
                    MemoryEntry(
                        category=str(data["category"]),
                        content=str(data["content"]),
                        confidence=float(data.get("confidence", 1.0)),
                        source_episode=int(data["source_episode"]) if data.get("source_episode") else None,
                    )
                )
                if len(entries) >= limit:
                    break
        return entries

    def remove(self, entry_id: str) -> bool:
        if entry_id in self._data:
            del self._data[entry_id]
            self._save()
            return True
        return False

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
        for data in self._data.values():
            if query_lower in str(data["content"]).lower():
                matches.append(
                    MemoryEntry(
                        category=str(data["category"]),
                        content=str(data["content"]),
                        confidence=float(data.get("confidence", 1.0)),
                        source_episode=int(data["source_episode"]) if data.get("source_episode") else None,
                    )
                )
                if len(matches) >= limit:
                    break
        return matches
