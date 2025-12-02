from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class MemoryScope(Enum):
    EPISODE = "episode"  # Per-episode, cleared on reset
    PERSISTENT = "persistent"  # Cross-episode, survives resets


@dataclass
class MemoryEntry:
    """Single memory entry, either episode-scoped or persistent."""

    tags: str  # Comma-separated tags (e.g., "plan,strategy" or "monster,dangerous")
    content: str  # LLM-formatted text
    scope: MemoryScope = MemoryScope.PERSISTENT
    priority: int = 5  # 1-9, higher = shown first when limit reached
    source_episode: int | None = None
    deleted: bool = False  # Soft-delete flag for debugging memory evolution
    entry_id: str | None = None  # Set by backend on retrieve


class MemoryBackend(ABC):
    """Abstract interface for memory storage.

    Implementations: FileMemoryBackend (JSON), future postgres/FAISS.
    """

    @abstractmethod
    def store(self, entry: MemoryEntry) -> str:
        """Store entry, return unique ID."""

    @abstractmethod
    def retrieve(
        self,
        tags: str | set[str] | None = None,
        scope: MemoryScope | None = None,
        episode: int | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Retrieve entries, optionally filtered by tags/scope/episode."""

    @abstractmethod
    def count(
        self,
        tags: str | set[str] | None = None,
        scope: MemoryScope | None = None,
        episode: int | None = None,
    ) -> int:
        """Count matching entries without loading them."""

    @abstractmethod
    def count_by_tag(
        self,
        scope: MemoryScope | None = None,
        episode: int | None = None,
    ) -> dict[str, int]:
        """Count entries per tag, optionally filtered by scope/episode."""

    @abstractmethod
    def remove(self, entry_id: str) -> bool:
        """Remove entry by ID. Returns True if found and removed."""

    @abstractmethod
    def remove_by_episode(self, episode: int) -> int:
        """Remove all episode-scoped entries for given episode. Returns count removed."""

    @abstractmethod
    def update(self, entry_id: str, new_content: str) -> bool:
        """Update existing entry content. Returns True if found and updated."""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Search entries by query. Backend-dependent (keyword or semantic)."""
