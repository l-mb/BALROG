"""Navigation and scanning tools for BRAID agent.

Grouped tools to reduce token overhead:
- scan: View map entities (monsters, items, traps, exits, unexplored)
- navigate: Query paths and distances
- auto_explore: Queue exploration actions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from claude_agent_sdk import tool

if TYPE_CHECKING:
    from ..storage import BraidStorage

# Module-level observation reference, set by agent before generate()
_obs: dict[str, Any] | None = None
_storage: BraidStorage | None = None
_episode: int = 0
_dlvl: int = 1

# Action queue for auto-exploration - picked up by agent after tool execution
_pending_actions: list[str] = []

# Track which action tools were called this turn (for multi-action detection)
_action_tools_called: list[str] = []


def set_context(obs: dict[str, Any], storage: BraidStorage, episode: int) -> None:
    """Set context for navigation tools. Called by agent before generate()."""
    global _obs, _storage, _episode, _dlvl
    _obs = obs
    _storage = storage
    _episode = episode
    # Handle NLE wrapper for blstats
    if obs:
        raw_obs = obs.get("obs", obs)
        if isinstance(raw_obs, dict):
            blstats = raw_obs.get("blstats", obs.get("blstats"))
            if blstats is not None:
                _dlvl = int(blstats[12])


def clear_action_tool_tracking() -> None:
    """Clear action tool tracking for new turn."""
    global _action_tools_called
    _action_tools_called = []


def get_action_tools_called() -> list[str]:
    """Get list of action tools called this turn."""
    return _action_tools_called.copy()


def get_pending_actions() -> list[str]:
    """Get and clear pending actions from auto_explore."""
    global _pending_actions
    actions = _pending_actions.copy()
    _pending_actions = []
    return actions


def _get_obs_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]] | None:
    """Extract observation data needed for navigation."""
    if _obs is None:
        return None
    try:
        # Handle NLE wrapper: obs["obs"] contains the actual data
        raw_obs = _obs.get("obs", _obs)
        if not isinstance(raw_obs, dict):
            raw_obs = _obs

        glyphs = raw_obs.get("glyphs")
        if glyphs is None:
            glyphs = _obs.get("glyphs")
        if glyphs is None:
            return None

        tty_chars = raw_obs.get("tty_chars", _obs.get("tty_chars", np.zeros_like(glyphs)))
        blstats = raw_obs.get("blstats", _obs.get("blstats"))
        if blstats is None:
            return None

        pos = (int(blstats[0]), int(blstats[1]))
        return glyphs, tty_chars, blstats, pos
    except (KeyError, TypeError, IndexError):
        return None


def _get_visited() -> set[tuple[int, int]]:
    """Get visited tiles for current episode/level."""
    if _storage is None:
        return set()
    return _storage.get_visited_for_level(_episode, _dlvl)


@tool(
    "scan",
    "Scan visible entities on the map. Returns positions and distances.",
    {
        "target": str,  # "monsters", "items", "traps", "exits", "unexplored"
    },
)
async def scan(args: dict[str, Any]) -> dict[str, Any]:
    """Scan for map entities."""
    obs_data = _get_obs_data()
    if obs_data is None:
        return {"content": [{"type": "text", "text": "ERROR: No observation available"}], "is_error": True}

    glyphs, tty_chars, blstats, pos = obs_data
    target = args.get("target", "").lower()

    from ..compute.navigation import (
        find_exits,
        find_unexplored,
        scan_items,
        scan_monsters,
        scan_traps,
    )

    result: str
    if target == "monsters":
        result = scan_monsters(glyphs, tty_chars, pos)
    elif target == "items":
        result = scan_items(glyphs, tty_chars, pos)
    elif target == "traps":
        result = scan_traps(glyphs, tty_chars, pos)
    elif target == "exits":
        result = find_exits(glyphs, pos)
    elif target == "unexplored":
        result = find_unexplored(glyphs, pos)
    else:
        return {
            "content": [
                {"type": "text", "text": f"ERROR: Unknown target '{target}'. Use: monsters, items, traps, exits, unexplored"}
            ],
            "is_error": True,
        }

    return {"content": [{"type": "text", "text": f"scan {target}: {result}"}]}


@tool(
    "navigate",
    "Query paths and distances. Commands: nearest (find feature), distance (between points), pathfind (get directions).",
    {
        "command": str,  # "nearest", "distance", "pathfind"
        "args": str,  # "stairs_down" or "@x1,y1 -> @x2,y2"
    },
)
async def navigate(args: dict[str, Any]) -> dict[str, Any]:
    """Navigation queries."""
    obs_data = _get_obs_data()
    if obs_data is None:
        return {"content": [{"type": "text", "text": "ERROR: No observation available"}], "is_error": True}

    glyphs, _tty_chars, _blstats, pos = obs_data
    command = args.get("command", "").lower()
    cmd_args = args.get("args", "")

    from ..compute.navigation import distance as calc_distance
    from ..compute.navigation import nearest as find_nearest
    from ..compute.navigation import pathfind

    if command == "nearest":
        # Find nearest feature
        feature = cmd_args.strip()
        result = find_nearest(glyphs, pos, feature)
        if result is None:
            return {"content": [{"type": "text", "text": f"nearest {feature}: NOT FOUND"}]}
        x, y, d = result
        return {"content": [{"type": "text", "text": f"nearest {feature}: @{x},{y} ({d} tiles)"}]}

    elif command == "distance":
        # Parse "@x1,y1 -> @x2,y2"
        import re

        match = re.match(r"@?(\d+),(\d+)\s*->\s*@?(\d+),(\d+)", cmd_args)
        if not match:
            return {
                "content": [{"type": "text", "text": "ERROR: Distance format: @x1,y1 -> @x2,y2"}],
                "is_error": True,
            }
        x1, y1, x2, y2 = map(int, match.groups())
        d = calc_distance(x1, y1, x2, y2)
        return {"content": [{"type": "text", "text": f"distance: @{x1},{y1} -> @{x2},{y2} = {d} tiles"}]}

    elif command == "pathfind":
        # Parse "@x1,y1 -> @x2,y2" or just "@x,y" (from current pos)
        import re

        full_match = re.match(r"@?(\d+),(\d+)\s*->\s*@?(\d+),(\d+)", cmd_args)
        single_match = re.match(r"@?(\d+),(\d+)$", cmd_args.strip())

        if full_match:
            x1, y1, x2, y2 = map(int, full_match.groups())
            start = (x1, y1)
            goal = (x2, y2)
        elif single_match:
            x2, y2 = map(int, single_match.groups())
            start = pos
            goal = (x2, y2)
        else:
            return {
                "content": [{"type": "text", "text": "ERROR: Pathfind format: @x,y or @x1,y1 -> @x2,y2"}],
                "is_error": True,
            }

        # Include visited tiles and monster positions as extra walkable
        visited = _get_visited()
        from ..compute.navigation import find_monster_positions

        extra_walkable = visited | find_monster_positions(glyphs) | {start}

        path = pathfind(glyphs, start, goal, extra_walkable)
        if path is None:
            return {"content": [{"type": "text", "text": f"pathfind: @{start[0]},{start[1]} -> @{goal[0]},{goal[1]} = NO PATH"}]}
        if not path:
            return {"content": [{"type": "text", "text": f"pathfind: @{start[0]},{start[1]} -> @{goal[0]},{goal[1]} = ALREADY THERE"}]}

        dirs_str = " ".join(path)
        return {"content": [{"type": "text", "text": f"pathfind: @{start[0]},{start[1]} -> @{goal[0]},{goal[1]} = {dirs_str} ({len(path)} moves)"}]}

    else:
        return {
            "content": [{"type": "text", "text": f"ERROR: Unknown command '{command}'. Use: nearest, distance, pathfind"}],
            "is_error": True,
        }


@tool(
    "travel_to",
    "Queue movement to a specific location. Pathfinds and queues all moves automatically. "
    "Aborts on combat, HP drop, or hunger.",
    {
        "target": str,  # "@x,y" absolute or "+dx,dy" relative
    },
)
async def travel_to(args: dict[str, Any]) -> dict[str, Any]:
    """Queue travel to a specific coordinate."""
    global _pending_actions, _action_tools_called
    import re

    _action_tools_called.append("travel_to")

    obs_data = _get_obs_data()
    if obs_data is None:
        return {"content": [{"type": "text", "text": "ERROR: No observation available"}], "is_error": True}

    glyphs, _tty_chars, _blstats, pos = obs_data
    target = args.get("target", "")
    visited = _get_visited()

    from ..compute.navigation import find_monster_positions, pathfind

    extra_walkable = visited | find_monster_positions(glyphs) | {pos}

    # Parse "@x,y" or "+dx,dy" (relative)
    abs_match = re.match(r"@?(\d+),(\d+)$", target.strip())
    rel_match = re.match(r"([+-]?\d+),([+-]?\d+)$", target.strip())

    if abs_match:
        gx, gy = map(int, abs_match.groups())
    elif rel_match:
        dx, dy = map(int, rel_match.groups())
        gx, gy = pos[0] + dx, pos[1] + dy
    else:
        return {
            "content": [{"type": "text", "text": "ERROR: travel_to format: @x,y or +dx,dy"}],
            "is_error": True,
        }

    path = pathfind(glyphs, pos, (gx, gy), extra_walkable)
    if path is None:
        return {"content": [{"type": "text", "text": f"travel_to @{gx},{gy}: NO PATH"}]}
    if not path:
        return {"content": [{"type": "text", "text": f"travel_to @{gx},{gy}: ALREADY THERE"}]}

    _pending_actions = path
    return {"content": [{"type": "text", "text": f"travel_to @{gx},{gy}: EXECUTING {len(path)} moves"}]}


@tool(
    "travel",
    "Queue movement in a direction for N steps. Faster than repeated game_action calls. "
    "Aborts on combat, HP drop, or hunger.",
    {
        "direction": str,  # "north", "NE", etc.
        "count": int,  # number of steps
    },
)
async def travel(args: dict[str, Any]) -> dict[str, Any]:
    """Queue directional travel."""
    global _pending_actions, _action_tools_called

    _action_tools_called.append("travel")

    obs_data = _get_obs_data()
    if obs_data is None:
        return {"content": [{"type": "text", "text": "ERROR: No observation available"}], "is_error": True}

    dir_name = args.get("direction", "").lower()
    count = args.get("count", 1)

    if count < 1:
        return {"content": [{"type": "text", "text": "ERROR: count must be >= 1"}], "is_error": True}

    # Normalize direction
    dir_map = {
        "n": "north",
        "s": "south",
        "e": "east",
        "w": "west",
        "ne": "northeast",
        "nw": "northwest",
        "se": "southeast",
        "sw": "southwest",
    }
    if dir_name in dir_map:
        dir_name = dir_map[dir_name]

    from ..compute.navigation import DIRS

    if dir_name not in DIRS:
        return {
            "content": [{"type": "text", "text": f"ERROR: Unknown direction '{dir_name}'. Use: north, south, east, west, northeast, northwest, southeast, southwest"}],
            "is_error": True,
        }

    actions = [dir_name] * count
    _pending_actions = actions
    return {"content": [{"type": "text", "text": f"travel {dir_name} {count}: EXECUTING {count} moves"}]}


@tool(
    "auto_explore",
    "Queue exploration actions for current room or corridor. Automatically walks to all reachable tiles. "
    "Aborts on combat, HP drop, or hunger.",
    {
        "mode": str,  # "room" or "corridor"
        "cautious": bool,  # abort on new tile discovery (for secret door hunting)
    },
)
async def auto_explore(args: dict[str, Any]) -> dict[str, Any]:
    """Queue exploration actions for room or corridor."""
    global _pending_actions, _action_tools_called

    _action_tools_called.append("auto_explore")

    obs_data = _get_obs_data()
    if obs_data is None:
        return {"content": [{"type": "text", "text": "ERROR: No observation available"}], "is_error": True}

    glyphs, _tty_chars, _blstats, pos = obs_data
    mode = args.get("mode", "").lower()
    cautious = args.get("cautious", False)

    visited = _get_visited()

    from ..compute.navigation import (
        detect_corridor,
        detect_room,
        plan_corridor_exploration,
        plan_room_exploration,
    )

    if mode == "room":
        room_tiles = detect_room(glyphs, pos)
        if room_tiles is None:
            return {"content": [{"type": "text", "text": "auto_explore room: NOT IN ROOM (try mode='corridor')"}]}

        actions = plan_room_exploration(glyphs, room_tiles, pos, visited)
        if not actions:
            return {"content": [{"type": "text", "text": "auto_explore room: FULLY EXPLORED - memorize this room as explored"}]}

        actions = actions[:100]
        _pending_actions = actions
        mode_suffix = " (cautious)" if cautious else ""
        return {"content": [{"type": "text", "text": f"auto_explore room{mode_suffix}: EXECUTING {len(actions)} actions"}]}

    elif mode == "corridor":
        corridor = detect_corridor(glyphs, pos)
        if corridor is None:
            return {"content": [{"type": "text", "text": "auto_explore corridor: NOT IN CORRIDOR (try mode='room')"}]}

        actions = plan_corridor_exploration(glyphs, corridor, pos, visited)
        if not actions:
            return {"content": [{"type": "text", "text": "auto_explore corridor: FULLY EXPLORED - memorize this corridor as explored"}]}

        actions = actions[:100]
        _pending_actions = actions
        mode_suffix = " (cautious)" if cautious else ""
        return {"content": [{"type": "text", "text": f"auto_explore corridor{mode_suffix}: EXECUTING {len(actions)} actions"}]}

    else:
        return {
            "content": [{"type": "text", "text": f"ERROR: Unknown mode '{mode}'. Use: room, corridor"}],
            "is_error": True,
        }
