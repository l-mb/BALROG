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
from .navigation import auto_explore, navigate, scan, travel, travel_to

if TYPE_CHECKING:
    from claude_agent_sdk.types import McpSdkServerConfig

    from ..storage import BraidStorage

__all__ = [
    "create_braid_mcp_server",
    "set_tool_context",
    "get_pending_actions",
    "clear_action_tool_tracking",
    "memory_add",
    "memory_discover_tags",
    "memory_remove",
    "memory_search",
    "scan",
    "navigate",
    "travel_to",
    "travel",
    "auto_explore",
    "game_action",
]

# Action tools that can only be called once per turn
ACTION_TOOLS = frozenset({"game_action", "travel", "travel_to", "auto_explore"})


def create_braid_mcp_server(storage: BraidStorage) -> McpSdkServerConfig:
    """Create MCP server with all BRAID tools.

    Args:
        storage: BraidStorage instance for memory operations.

    Returns:
        McpSdkServerConfig for use with ClaudeAgentOptions.mcp_servers
    """
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
            travel_to,
            travel,
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
    # Clear action tool tracking for new turn
    nav_module.clear_action_tool_tracking()
    action_module.clear_action_tool_tracking()


def clear_action_tool_tracking() -> None:
    """Clear action tool tracking. Called at start of each turn."""
    nav_module.clear_action_tool_tracking()
    action_module.clear_action_tool_tracking()


def get_pending_actions() -> tuple[list[str], str | None]:
    """Get pending actions from tools (auto_explore and game_action).

    Returns:
        Tuple of (actions, error_message).
        If multiple action tools were called, actions is empty and error_message explains why.
        Otherwise, actions contains the queued actions and error_message is None.
    """
    # Check if multiple action tools were called
    nav_action_tools = nav_module.get_action_tools_called()
    action_action_tools = action_module.get_action_tools_called()
    all_action_tools = nav_action_tools + action_action_tools

    if len(all_action_tools) > 1:
        # Multiple action tools called - reject all
        nav_module.get_pending_actions()  # Clear nav pending
        action_module.get_pending_actions()  # Clear action pending
        tools_str = ", ".join(all_action_tools)
        error = (
            f"VIOLATION: Multiple action tools called in one turn: {tools_str}. "
            f"You may only use ONE of game_action/travel/travel_to/auto_explore per turn. "
            f"NO actions were executed. Choose ONE action tool and try again."
        )
        return [], error

    # Collect from both navigation (auto_explore, travel, travel_to) and action (game_action) modules
    actions = nav_module.get_pending_actions()
    actions.extend(action_module.get_pending_actions())
    return actions, None
