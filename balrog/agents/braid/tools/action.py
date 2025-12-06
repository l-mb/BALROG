"""Game action tool for BRAID agent.

Executes NetHack game commands. Actions are queued and executed one per turn.
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import tool

# Action queue for game actions - picked up by agent after tool execution
_pending_actions: list[str] = []

# Track which action tools were called this turn (for multi-action detection)
_action_tools_called: list[str] = []


def clear_action_tool_tracking() -> None:
    """Clear action tool tracking for new turn."""
    global _action_tools_called
    _action_tools_called = []


def get_action_tools_called() -> list[str]:
    """Get list of action tools called this turn."""
    return _action_tools_called.copy()


def get_pending_actions() -> list[str]:
    """Get and clear pending actions from game_action tool."""
    global _pending_actions
    actions = _pending_actions.copy()
    _pending_actions = []
    return actions


# Valid single-character commands and their expansions
COMPOUND_ACTIONS = {
    # Movement
    "north": ["north"],
    "south": ["south"],
    "east": ["east"],
    "west": ["west"],
    "northeast": ["northeast"],
    "northwest": ["northwest"],
    "southeast": ["southeast"],
    "southwest": ["southwest"],
    "n": ["north"],
    "s": ["south"],
    "e": ["east"],
    "w": ["west"],
    "ne": ["northeast"],
    "nw": ["northwest"],
    "se": ["southeast"],
    "sw": ["southwest"],
    # Directional commands - expand to command + direction
    "open north": ["open", "north"],
    "open south": ["open", "south"],
    "open east": ["open", "east"],
    "open west": ["open", "west"],
    "close north": ["close", "north"],
    "close south": ["close", "south"],
    "close east": ["close", "east"],
    "close west": ["close", "west"],
    "kick north": ["kick", "north"],
    "kick south": ["kick", "south"],
    "kick east": ["kick", "east"],
    "kick west": ["kick", "west"],
}


def expand_action(action: str) -> list[str]:
    """Expand compound action into individual commands."""
    action = action.strip().lower()

    # Check compound actions first
    if action in COMPOUND_ACTIONS:
        return COMPOUND_ACTIONS[action]

    # Handle "open <direction>" pattern dynamically
    for cmd in ["open", "close", "kick", "zap", "throw", "fire"]:
        if action.startswith(f"{cmd} "):
            direction = action[len(cmd) + 1 :].strip()
            if direction in COMPOUND_ACTIONS:
                return [cmd] + COMPOUND_ACTIONS[direction]
            return [cmd, direction]

    # Single action
    return [action]


@tool(
    "game_action",
    "Execute NetHack game action(s) in sequence. Pass multiple actions as separate arguments. "
    "Examples: game_action('north'), game_action('north', 'east', 'pickup'), game_action('open north'). "
    "Actions queue and execute one per game turn.",
    {"actions": list[str]},
)
async def game_action(args: dict[str, Any]) -> dict[str, Any]:
    """Execute game action(s)."""
    global _pending_actions, _action_tools_called

    _action_tools_called.append("game_action")

    actions_input = args.get("actions", [])

    # Handle both list and string input for robustness
    if isinstance(actions_input, str):
        raw_actions = [a.strip() for a in actions_input.strip().split("\n") if a.strip()]
    elif isinstance(actions_input, list):
        raw_actions = [str(a).strip() for a in actions_input if str(a).strip()]
    else:
        raw_actions = []

    if not raw_actions:
        return {
            "content": [{"type": "text", "text": "ERROR: No actions provided"}],
            "is_error": True,
        }

    # Parse and expand actions
    expanded: list[str] = []
    for action in raw_actions:
        expanded.extend(expand_action(action))

    if not expanded:
        return {
            "content": [{"type": "text", "text": "ERROR: No valid actions"}],
            "is_error": True,
        }

    _pending_actions = expanded
    actions_preview = ", ".join(expanded[:5])
    if len(expanded) > 5:
        actions_preview += f", ... ({len(expanded)} total)"

    return {"content": [{"type": "text", "text": f"Queued: {actions_preview}"}]}
