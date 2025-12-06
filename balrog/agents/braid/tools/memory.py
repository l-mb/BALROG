"""Memory management tools for BRAID agent.

Provides Claude with the ability to store, retrieve, and manage memory entries.
Memory persists to SQLite and supports episode vs persistent scoping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from claude_agent_sdk import tool

if TYPE_CHECKING:
    from ..storage import BraidStorage

# Module-level storage reference, injected by create_braid_mcp_server()
_storage: BraidStorage | None = None

# Module-level state for tag filtering (managed by agent, shared with tools)
_enabled_tags: set[str] | None = None  # None = all enabled
_episode_number: int = 0
_step: int = 0


def set_context(storage: BraidStorage, episode: int, step: int, enabled_tags: set[str] | None) -> None:
    """Set context for memory tools. Called by agent before generate()."""
    global _storage, _episode_number, _step, _enabled_tags
    _storage = storage
    _episode_number = episode
    _step = step
    _enabled_tags = enabled_tags


@tool(
    "memory_add",
    "Store a memory entry for learning and tracking. Use for rules learned, "
    "positions explored, items found, dangers encountered. Entries are shown "
    "with [id] prefix for later removal.",
    {
        "content": str,
        "scope": str,  # "episode" or "persistent"
        "tags": str,  # comma-separated
        "priority": int,  # 1-9, higher = shown first
    },
)
async def memory_add(args: dict[str, Any]) -> dict[str, Any]:
    """Add a memory entry."""
    if _storage is None:
        return {"content": [{"type": "text", "text": "ERROR: Storage not initialized"}], "is_error": True}

    from ..storage import MemoryEntry, MemoryScope

    content = args["content"]
    scope_str = args.get("scope", "episode")
    tags = args.get("tags", "")
    priority = args.get("priority", 5)

    # Validate scope
    try:
        scope = MemoryScope(scope_str)
    except ValueError:
        return {
            "content": [{"type": "text", "text": f"ERROR: Invalid scope '{scope_str}'. Use 'episode' or 'persistent'"}],
            "is_error": True,
        }

    # Validate priority
    if not 1 <= priority <= 9:
        priority = max(1, min(9, priority))

    entry = MemoryEntry(
        tags=tags,
        content=content,
        scope=scope,
        priority=priority,
        source_episode=_episode_number,
        source_step=_step,
    )

    entry_id = _storage.store(entry)
    return {"content": [{"type": "text", "text": f"Added [{entry_id}] ({scope_str}, prio:{priority})"}]}


@tool(
    "memory_remove",
    "Remove a memory entry by its ID (the [abc123] prefix shown in memory list). "
    "Use to clean up outdated or incorrect information.",
    {"entry_id": str},
)
async def memory_remove(args: dict[str, Any]) -> dict[str, Any]:
    """Remove a memory entry by ID."""
    if _storage is None:
        return {"content": [{"type": "text", "text": "ERROR: Storage not initialized"}], "is_error": True}

    entry_id = args["entry_id"]
    result = _storage.remove(entry_id)

    if result is None:
        return {"content": [{"type": "text", "text": f"NOT FOUND: [{entry_id}]"}]}

    return {"content": [{"type": "text", "text": f"Removed [{entry_id}] (was {result})"}]}


@tool(
    "memory_search",
    "Search memory entries by content substring. Returns matching entries with IDs.",
    {"query": str, "limit": int},
)
async def memory_search(args: dict[str, Any]) -> dict[str, Any]:
    """Search memory entries by content."""
    if _storage is None:
        return {"content": [{"type": "text", "text": "ERROR: Storage not initialized"}], "is_error": True}

    query = args["query"]
    limit = args.get("limit", 10)

    results = _storage.search(query, limit=limit)

    if not results:
        return {"content": [{"type": "text", "text": f"No matches for '{query}'"}]}

    lines = [f"[{e.entry_id}] ({e.scope.value}) {e.content}" for e in results]
    return {"content": [{"type": "text", "text": f"Found {len(results)}:\n" + "\n".join(lines)}]}


@tool(
    "tag_filter",
    "Control which memory tags are shown. Enable/disable specific tags to focus "
    "on relevant memories. Reset to show all tags.",
    {
        "action": str,  # "enable", "disable", "reset"
        "tags": str,  # comma-separated (ignored if action=reset)
    },
)
async def tag_filter(args: dict[str, Any]) -> dict[str, Any]:
    """Control tag-based memory filtering."""
    global _enabled_tags

    action = args.get("action", "").lower()
    tags_str = args.get("tags", "")
    tags = {t.strip() for t in tags_str.split(",") if t.strip()}

    if action == "reset":
        _enabled_tags = None
        return {"content": [{"type": "text", "text": "Tag filter reset - all tags now visible"}]}

    if action == "enable":
        if _enabled_tags is None:
            _enabled_tags = tags.copy()
        else:
            _enabled_tags.update(tags)
        return {"content": [{"type": "text", "text": f"Enabled tags: {', '.join(sorted(_enabled_tags))}"}]}

    if action == "disable":
        if _enabled_tags is None:
            # Start from all known tags minus the disabled ones
            if _storage:
                all_tags = set(_storage.count_by_tag().keys())
                _enabled_tags = all_tags - tags
        else:
            _enabled_tags -= tags
        enabled_list = sorted(_enabled_tags) if _enabled_tags else ["(none)"]
        return {"content": [{"type": "text", "text": f"Enabled tags: {', '.join(enabled_list)}"}]}

    return {
        "content": [{"type": "text", "text": f"ERROR: Invalid action '{action}'. Use 'enable', 'disable', or 'reset'"}],
        "is_error": True,
    }


def get_enabled_tags() -> set[str] | None:
    """Get current enabled tags. None means all tags enabled."""
    return _enabled_tags
