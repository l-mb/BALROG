from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MemoryEntry:
    """Single entry in persistent memory."""

    category: str  # "strategy", "monster", "item", "pitfall", etc.
    content: str  # LLM-formatted text
    confidence: float = 1.0  # 0-1, for future ranking/filtering
    source_episode: int | None = None


class MemoryBackend(ABC):
    """Abstract interface for persistent memory storage.

    Implementations: FileMemoryBackend (JSON), future postgres/FAISS.
    """

    @abstractmethod
    def store(self, entry: MemoryEntry) -> str:
        """Store entry, return unique ID."""

    @abstractmethod
    def retrieve(self, category: str | None = None, limit: int = 20) -> list[MemoryEntry]:
        """Retrieve entries, optionally filtered by category."""

    @abstractmethod
    def remove(self, entry_id: str) -> bool:
        """Remove entry by ID. Returns True if found and removed."""

    @abstractmethod
    def update(self, entry_id: str, new_content: str) -> bool:
        """Update existing entry content. Returns True if found and updated."""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Search entries by query. Backend-dependent (keyword or semantic)."""
