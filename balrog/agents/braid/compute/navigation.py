"""Navigation and scanning helpers for BRAID agent."""

from collections import deque

import numpy as np
from nle import nethack

CMAP_OFF = nethack.GLYPH_CMAP_OFF

# Hunger state names (blstats index 21)
HUNGER_STATES = ["Satiated", "OK", "Hungry", "Weak", "Fainting", "Fainted", "Starved"]

# Direction vectors: (dy, dx) for NetHack coordinate system
# In NetHack, y increases downward (row), x increases rightward (column)
DIRS = {
    "north": (-1, 0),
    "south": (1, 0),
    "east": (0, 1),
    "west": (0, -1),
    "northeast": (-1, 1),
    "northwest": (-1, -1),
    "southeast": (1, 1),
    "southwest": (1, -1),
}

# Walkable cmap indices (from NetHack rm.h)
# 12: S_ndoor (doorway), 13-14: open doors, 19-20: floor/darkroom
# 21-22: corridor/lit corridor, 23-24: stairs, 25-26: ladders
# 27: altar, 29: throne, 30: sink, 31: fountain, 33: ice, 35-36: drawbridge
WALKABLE_CMAP = {12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 33, 35, 36}

# Feature cmap indices for "nearest" command
FEATURES = {
    "stairs_down": {24},
    "stairs_up": {23},
    "stairs": {23, 24},
    "altar": {27},
    "fountain": {31},
    "sink": {30},
    "throne": {29},
}


def get_position(blstats: np.ndarray) -> tuple[int, int]:
    """Extract (x, y) from blstats array."""
    return int(blstats[0]), int(blstats[1])


def build_walkable_mask(glyphs: np.ndarray) -> np.ndarray:
    """Build mask of walkable tiles for pathfinding.

    Only explored tiles that are floors, corridors, doors, stairs, etc.
    are considered walkable. Unexplored tiles (S_stone) are NOT walkable.
    """
    walkable = np.zeros(glyphs.shape, dtype=bool)
    for idx in WALKABLE_CMAP:
        walkable |= glyphs == CMAP_OFF + idx
    return walkable


def distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Manhattan distance between two points."""
    return abs(x2 - x1) + abs(y2 - y1)


def nearest(
    glyphs: np.ndarray, pos: tuple[int, int], feature: str
) -> tuple[int, int, int] | None:
    """Find nearest feature of given type.

    Args:
        glyphs: 2D glyph array from observation
        pos: Current (x, y) position
        feature: Feature name (stairs_down, altar, fountain, etc.)

    Returns:
        (x, y, distance) of nearest feature, or None if not found
    """
    if feature not in FEATURES:
        return None

    cmap_indices = FEATURES[feature]
    target_glyphs = {CMAP_OFF + idx for idx in cmap_indices}

    # Find all matching positions
    matches = []
    for row in range(glyphs.shape[0]):
        for col in range(glyphs.shape[1]):
            if glyphs[row, col] in target_glyphs:
                d = distance(pos[0], pos[1], col, row)
                matches.append((col, row, d))

    if not matches:
        return None
    return min(matches, key=lambda m: m[2])


def pathfind(
    glyphs: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[str] | None:
    """BFS pathfinding on explored walkable tiles.

    Args:
        glyphs: 2D glyph array from observation
        start: Starting (x, y) position
        goal: Goal (x, y) position

    Returns:
        List of direction names (north, east, etc.) or None if no path exists
    """
    walkable = build_walkable_mask(glyphs)
    sx, sy = start
    gx, gy = goal

    # Check if goal is walkable
    if not (0 <= gy < glyphs.shape[0] and 0 <= gx < glyphs.shape[1]):
        return None
    if not walkable[gy, gx]:
        return None

    # Already at goal
    if (sx, sy) == (gx, gy):
        return []

    # BFS
    queue: deque[tuple[int, int, list[str]]] = deque([(sx, sy, [])])
    visited = {(sx, sy)}

    while queue:
        x, y, path = queue.popleft()

        for dir_name, (dy, dx) in DIRS.items():
            nx, ny = x + dx, y + dy

            if 0 <= ny < glyphs.shape[0] and 0 <= nx < glyphs.shape[1]:
                if (nx, ny) not in visited and walkable[ny, nx]:
                    new_path = path + [dir_name]

                    if (nx, ny) == (gx, gy):
                        return new_path

                    visited.add((nx, ny))
                    queue.append((nx, ny, new_path))

    return None  # No path found


def format_status(blstats: np.ndarray) -> str:
    """Format blstats into compact status line.

    blstats indices (from NLE):
        0: x, 1: y, 10: hp, 11: maxhp, 12: depth, 14: pw, 15: maxpw,
        16: ac, 18: xlvl, 20: time, 21: hunger_state, 22: encumbrance
    """
    hp, maxhp = int(blstats[10]), int(blstats[11])
    pw, maxpw = int(blstats[14]), int(blstats[15])
    ac = int(blstats[16])
    xlvl = int(blstats[18])
    turn = int(blstats[20])
    hunger = int(blstats[21])
    dlvl = int(blstats[12])

    hunger_str = HUNGER_STATES[hunger] if 0 <= hunger < len(HUNGER_STATES) else f"?{hunger}"

    return f"HP:{hp}/{maxhp} Pw:{pw}/{maxpw} AC:{ac} XL:{xlvl} T:{turn} {hunger_str} Dlvl:{dlvl}"


def scan_monsters(
    glyphs: np.ndarray, tty_chars: np.ndarray, pos: tuple[int, int]
) -> str:
    """Scan for visible monsters on the map.

    Args:
        glyphs: 2D glyph array from observation
        tty_chars: 2D ASCII character array from observation
        pos: Current (x, y) position

    Returns:
        Formatted string of monsters: "d@45,12(3) G@40,8(7)" or "none"
    """
    px, py = pos
    monsters = []

    for row in range(glyphs.shape[0]):
        for col in range(glyphs.shape[1]):
            glyph = int(glyphs[row, col])
            # Skip player position
            if col == px and row == py:
                continue
            # Check if it's any kind of monster (including pets)
            if nethack.glyph_is_monster(glyph) or nethack.glyph_is_pet(glyph):
                char = chr(tty_chars[row, col]) if tty_chars[row, col] > 0 else "?"
                d = distance(px, py, col, row)
                is_pet = nethack.glyph_is_pet(glyph)
                pet_marker = "*" if is_pet else ""
                monsters.append((char, col, row, d, pet_marker))

    if not monsters:
        return "none"

    # Sort by distance
    monsters.sort(key=lambda m: m[3])
    return " ".join(f"{m[0]}{m[4]}@{m[1]},{m[2]}({m[3]})" for m in monsters[:10])


def scan_items(
    glyphs: np.ndarray, tty_chars: np.ndarray, pos: tuple[int, int]
) -> str:
    """Scan for visible items on the map.

    Args:
        glyphs: 2D glyph array from observation
        tty_chars: 2D ASCII character array from observation
        pos: Current (x, y) position

    Returns:
        Formatted string of items: ")@44,10(2) [@42,11(5)" or "none"
    """
    px, py = pos
    items = []

    for row in range(glyphs.shape[0]):
        for col in range(glyphs.shape[1]):
            glyph = int(glyphs[row, col])
            if nethack.glyph_is_object(glyph):
                char = chr(tty_chars[row, col]) if tty_chars[row, col] > 0 else "?"
                d = distance(px, py, col, row)
                items.append((char, col, row, d))

    if not items:
        return "none"

    # Sort by distance
    items.sort(key=lambda m: m[3])
    return " ".join(f"{m[0]}@{m[1]},{m[2]}({m[3]})" for m in items[:15])


# Stone glyph (unexplored) - cmap index 0
S_STONE = CMAP_OFF + 0


def find_unexplored(glyphs: np.ndarray, pos: tuple[int, int]) -> str:
    """Find exploration frontiers - walkable tiles adjacent to unexplored areas.

    Args:
        glyphs: 2D glyph array from observation
        pos: Current (x, y) position

    Returns:
        Formatted string of frontiers: "@44,5(N,3) @50,12(E,7)" or "none"
    """
    px, py = pos
    walkable = build_walkable_mask(glyphs)
    rows, cols = glyphs.shape
    frontiers = []

    # Check each walkable tile for adjacent unexplored (stone) tiles
    for row in range(rows):
        for col in range(cols):
            if not walkable[row, col]:
                continue

            # Check all 8 neighbors for unexplored tiles
            unexplored_dirs = []
            for dir_name, (dy, dx) in DIRS.items():
                ny, nx = row + dy, col + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    if glyphs[ny, nx] == S_STONE:
                        unexplored_dirs.append(dir_name[:1].upper())  # N, S, E, W, etc.

            if unexplored_dirs:
                d = distance(px, py, col, row)
                dirs_str = "".join(sorted(set(unexplored_dirs)))
                frontiers.append((col, row, dirs_str, d))

    if not frontiers:
        return "none"

    # Sort by distance
    frontiers.sort(key=lambda f: f[3])
    return " ".join(f"@{f[0]},{f[1]}({f[2]},{f[3]})" for f in frontiers[:10])


# Cmap indices for doors and corridors
S_VCDOOR = 12  # Doorway
S_HODOOR = 13  # Open door horizontal
S_VODOOR = 14  # Open door vertical
S_CORR = 21    # Corridor
S_LITCORR = 22  # Lit corridor
DOOR_CMAP = {12, 13, 14, 15, 16}  # All door types
CORRIDOR_CMAP = {21, 22}


# Floor cmap indices for room detection
FLOOR_CMAP = {19, 20}  # S_room, S_darkroom


def detect_room(
    glyphs: np.ndarray, pos: tuple[int, int]
) -> set[tuple[int, int]] | None:
    """Flood-fill to detect room from current position.

    Args:
        glyphs: 2D glyph array from observation
        pos: Current (x, y) position

    Returns:
        Set of (x, y) floor tile coordinates, or None if not in a room
    """
    px, py = pos
    rows, cols = glyphs.shape

    # Check if current position is a floor tile
    if not (0 <= py < rows and 0 <= px < cols):
        return None
    cmap_idx = int(glyphs[py, px]) - CMAP_OFF
    if cmap_idx not in FLOOR_CMAP:
        return None

    # Flood fill from current position
    room_tiles: set[tuple[int, int]] = set()
    queue = [(px, py)]

    while queue:
        x, y = queue.pop()
        if (x, y) in room_tiles:
            continue
        if not (0 <= y < rows and 0 <= x < cols):
            continue
        cmap = int(glyphs[y, x]) - CMAP_OFF
        if cmap not in FLOOR_CMAP:
            continue

        room_tiles.add((x, y))
        # Add cardinal neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            queue.append((x + dx, y + dy))

    # Check if it's actually a room (not corridor)
    if len(room_tiles) < 6:
        return None  # Too small

    xs = [t[0] for t in room_tiles]
    ys = [t[1] for t in room_tiles]
    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1

    if width < 3 or height < 3:
        return None  # Too narrow, probably corridor

    return room_tiles


def detect_corridor(
    glyphs: np.ndarray, pos: tuple[int, int]
) -> list[tuple[int, int]] | None:
    """Detect corridor tiles from current position.

    Args:
        glyphs: 2D glyph array from observation
        pos: Current (x, y) position

    Returns:
        Ordered list of corridor tiles, or None if not in a corridor
    """
    px, py = pos
    rows, cols = glyphs.shape

    # Check if current position is a corridor tile
    if not (0 <= py < rows and 0 <= px < cols):
        return None
    cmap_idx = int(glyphs[py, px]) - CMAP_OFF
    if cmap_idx not in CORRIDOR_CMAP:
        return None

    # Flood fill on corridor tiles
    corridor_tiles: set[tuple[int, int]] = set()
    queue = [(px, py)]

    while queue:
        x, y = queue.pop()
        if (x, y) in corridor_tiles:
            continue
        if not (0 <= y < rows and 0 <= x < cols):
            continue
        cmap = int(glyphs[y, x]) - CMAP_OFF
        if cmap not in CORRIDOR_CMAP:
            continue

        corridor_tiles.add((x, y))
        # Add all 8 neighbors for corridors
        for dy, dx in DIRS.values():
            queue.append((x + dx, y + dy))

    if len(corridor_tiles) < 2:
        return None

    # Order tiles by walking from one end to the other
    # Find endpoints (tiles with only 1 corridor neighbor)
    def count_neighbors(tile: tuple[int, int]) -> int:
        x, y = tile
        return sum(1 for dy, dx in DIRS.values() if (x + dx, y + dy) in corridor_tiles)

    endpoints = [t for t in corridor_tiles if count_neighbors(t) == 1]

    # If no clear endpoints, just return tiles sorted by position
    if not endpoints:
        return sorted(corridor_tiles)

    # Walk from first endpoint
    ordered = []
    visited: set[tuple[int, int]] = set()
    current: tuple[int, int] | None = endpoints[0]

    while current is not None and current not in visited:
        visited.add(current)
        ordered.append(current)
        # Find unvisited neighbor
        x, y = current
        next_tile = None
        for dy, dx in DIRS.values():
            neighbor = (x + dx, y + dy)
            if neighbor in corridor_tiles and neighbor not in visited:
                next_tile = neighbor
                break
        current = next_tile

    return ordered


def _find_perimeter(
    room_tiles: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Find perimeter tiles (room tiles adjacent to non-room)."""
    perimeter = []
    for x, y in room_tiles:
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            if (x + dx, y + dy) not in room_tiles:
                perimeter.append((x, y))
                break
    return perimeter


def _order_perimeter(
    perimeter: list[tuple[int, int]], start: tuple[int, int]
) -> list[tuple[int, int]]:
    """Order perimeter tiles for efficient walk using nearest-neighbor heuristic."""
    if not perimeter:
        return []

    # Start from nearest to current position
    remaining = set(perimeter)
    ordered = []
    current = min(remaining, key=lambda p: distance(start[0], start[1], p[0], p[1]))

    while remaining:
        ordered.append(current)
        remaining.discard(current)
        if remaining:
            current = min(remaining, key=lambda p: distance(current[0], current[1], p[0], p[1]))

    return ordered


def plan_room_exploration(
    glyphs: np.ndarray, room_tiles: set[tuple[int, int]], pos: tuple[int, int]
) -> list[str]:
    """Plan efficient perimeter walk with searches.

    Args:
        glyphs: 2D glyph array from observation
        room_tiles: Set of room floor tiles
        pos: Current (x, y) position

    Returns:
        List of actions (directions + "search")
    """
    # Find and order perimeter
    perimeter = _find_perimeter(room_tiles)
    if not perimeter:
        return []

    ordered = _order_perimeter(perimeter, pos)

    # Generate actions
    actions = []
    current = pos

    for target in ordered:
        # Path to target
        path = pathfind(glyphs, current, target)
        if path:
            actions.extend(path)
            current = target
        # Search at wall
        actions.append("search")

    return actions


def plan_corridor_exploration(
    glyphs: np.ndarray, corridor: list[tuple[int, int]], pos: tuple[int, int]
) -> list[str]:
    """Plan corridor exploration with searches at ends.

    Args:
        glyphs: 2D glyph array from observation
        corridor: Ordered list of corridor tiles
        pos: Current (x, y) position

    Returns:
        List of actions (directions + "search")
    """
    if not corridor:
        return []

    actions = []

    # Find endpoints
    start_end = corridor[0]
    far_end = corridor[-1]

    # Determine which end to explore first (nearest)
    d_start = distance(pos[0], pos[1], start_end[0], start_end[1])
    d_far = distance(pos[0], pos[1], far_end[0], far_end[1])

    if d_start <= d_far:
        first, second = start_end, far_end
    else:
        first, second = far_end, start_end

    # Go to first end
    path = pathfind(glyphs, pos, first)
    if path:
        actions.extend(path)
    # Search at dead-end (3x)
    actions.extend(["search", "search", "search", "search"])

    # Go to second end
    path = pathfind(glyphs, first, second)
    if path:
        actions.extend(path)
    # Search at dead-end (3x)
    actions.extend(["search", "search", "search"])

    return actions


def find_exits(glyphs: np.ndarray, pos: tuple[int, int]) -> str:
    """Find exits from current room/area - doors and corridor openings.

    Args:
        glyphs: 2D glyph array from observation
        pos: Current (x, y) position

    Returns:
        Formatted string: "N@45,5(door,3) E@52,10(corr,8)" or "none"
    """
    px, py = pos
    rows, cols = glyphs.shape
    exits = []

    for row in range(rows):
        for col in range(cols):
            glyph = int(glyphs[row, col])
            cmap_idx = glyph - CMAP_OFF

            exit_type = None
            if cmap_idx in DOOR_CMAP:
                exit_type = "door"
            elif cmap_idx in CORRIDOR_CMAP:
                # Only count corridor tiles at room boundaries (adjacent to floor)
                # Check if adjacent to a floor tile
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = row + dy, col + dx
                    if 0 <= ny < rows and 0 <= nx < cols:
                        adj_cmap = int(glyphs[ny, nx]) - CMAP_OFF
                        if adj_cmap in {19, 20}:  # Floor tiles
                            exit_type = "corr"
                            break

            if exit_type:
                d = distance(px, py, col, row)
                # Determine direction from player
                dy, dx = row - py, col - px
                if abs(dy) > abs(dx):
                    dir_char = "S" if dy > 0 else "N"
                elif abs(dx) > abs(dy):
                    dir_char = "E" if dx > 0 else "W"
                else:
                    dir_char = ("S" if dy > 0 else "N") + ("E" if dx > 0 else "W")
                exits.append((dir_char, col, row, exit_type, d))

    if not exits:
        return "none"

    # Sort by distance
    exits.sort(key=lambda e: e[4])
    return " ".join(f"{e[0]}@{e[1]},{e[2]}({e[3]},{e[4]})" for e in exits[:10])
