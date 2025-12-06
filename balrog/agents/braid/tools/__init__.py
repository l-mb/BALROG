"""BRAID Tool Definitions for Claude Agent SDK.

Exposes BRAID functionality as MCP tools that Claude can invoke directly.
Tools are organized into three modules:
- memory: Memory CRUD operations (add, remove, search)
- navigation: Map scanning and pathfinding (scan, navigate, auto_explore)
- action: Game action execution
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from claude_agent_sdk import create_sdk_mcp_server

from . import action as action_module
from . import memory as memory_module
from . import navigation as nav_module
from .action import game_action
from .memory import memory_add, memory_discover_tags, memory_remove, memory_search
from .navigation import auto_explore, navigate, scan

if TYPE_CHECKING:
    from claude_agent_sdk.types import McpSdkServerConfig

    from ..storage import BraidStorage

__all__ = [
    "create_braid_mcp_server",
    "set_tool_context",
    "get_pending_actions",
    "memory_add",
    "memory_discover_tags",
    "memory_remove",
    "memory_search",
    "scan",
    "navigate",
    "auto_explore",
    "game_action",
]


def create_braid_mcp_server(storage: BraidStorage) -> McpSdkServerConfig:
    """Create MCP server with all BRAID tools.

    Args:
        storage: BraidStorage instance for memory operations.

    Returns:
        McpSdkServerConfig for use with ClaudeAgentOptions.mcp_servers
    """
    # Inject storage into memory module (navigation gets it via set_tool_context)
    memory_module._storage = storage

    return create_sdk_mcp_server(
        name="braid",
        version="1.0.0",
        tools=[
            # Memory tools
            memory_add,
            memory_remove,
            memory_search,
            memory_discover_tags,
            # Navigation tools
            scan,
            navigate,
            auto_explore,
            # Action tool
            game_action,
        ],
    )


def set_tool_context(
    obs: dict[str, Any],
    storage: BraidStorage,
    episode: int,
    step: int,
) -> None:
    """Set context for all tools. Call before each generate().

    Args:
        obs: Current observation dict with glyphs, blstats, tty_chars
        storage: BraidStorage instance
        episode: Current episode number
        step: Current step number
    """
    memory_module.set_context(storage, episode, step)
    nav_module.set_context(obs, storage, episode)


def get_pending_actions() -> list[str]:
    """Get pending actions from tools (auto_explore and game_action).

    Returns list of action strings to execute. Clears the queue.
    """
    # Collect from both navigation (auto_explore) and action (game_action) modules
    actions = nav_module.get_pending_actions()
    actions.extend(action_module.get_pending_actions())
    return actions
