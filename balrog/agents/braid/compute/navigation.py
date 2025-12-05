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


def build_walkable_mask(glyphs: np.ndarray, avoid_traps: bool = True) -> np.ndarray:
    """Build mask of walkable tiles for pathfinding.

    Only explored tiles that are floors, corridors, doors, stairs, etc.
    are considered walkable. Unexplored tiles (S_stone) are NOT walkable.
    Discovered traps are avoided by default.
    """
    walkable = np.zeros(glyphs.shape, dtype=bool)
    for idx in WALKABLE_CMAP:
        walkable |= glyphs == CMAP_OFF + idx

    # Exclude discovered traps from walkable tiles
    if avoid_traps:
        rows, cols = glyphs.shape
        for row in range(rows):
            for col in range(cols):
                if nethack.glyph_is_trap(int(glyphs[row, col])):
                    walkable[row, col] = False

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
    glyphs: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    extra_walkable: set[tuple[int, int]] | None = None,
) -> list[str] | None:
    """BFS pathfinding on explored walkable tiles.

    Args:
        glyphs: 2D glyph array from observation
        start: Starting (x, y) position
        goal: Goal (x, y) position
        extra_walkable: Additional positions to treat as walkable (e.g., player position)

    Returns:
        List of direction names (north, east, etc.) or None if no path exists
    """
    walkable = build_walkable_mask(glyphs)

    # Mark extra positions as walkable (player/monster positions have different glyphs)
    if extra_walkable:
        for x, y in extra_walkable:
            if 0 <= y < glyphs.shape[0] and 0 <= x < glyphs.shape[1]:
                walkable[y, x] = True

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


def scan_traps(
    glyphs: np.ndarray, tty_chars: np.ndarray, pos: tuple[int, int]
) -> str:
    """Scan for discovered traps on the map.

    Args:
        glyphs: 2D glyph array from observation
        tty_chars: 2D ASCII character array from observation
        pos: Current (x, y) position

    Returns:
        Formatted string of traps: "^@44,10(2) ^@42,11(5)" or "none"
    """
    px, py = pos
    traps = []

    for row in range(glyphs.shape[0]):
        for col in range(glyphs.shape[1]):
            glyph = int(glyphs[row, col])
            if nethack.glyph_is_trap(glyph):
                char = chr(tty_chars[row, col]) if tty_chars[row, col] > 0 else "^"
                d = distance(px, py, col, row)
                traps.append((char, col, row, d))

    if not traps:
        return "none"

    # Sort by distance
    traps.sort(key=lambda t: t[3])
    return " ".join(f"{t[0]}@{t[1]},{t[2]}({t[3]})" for t in traps[:10])


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


def _is_room_tile(glyphs: np.ndarray, x: int, y: int, room_tiles: set[tuple[int, int]]) -> bool:
    """Check if tile is a floor tile or likely on floor (item/monster on floor).

    Items and monsters overlay floor glyphs, so we need to check if they're
    adjacent to known room tiles to include them in the room.
    Traps are also included as room tiles (they're on the floor).
    """
    rows, cols = glyphs.shape
    if not (0 <= y < rows and 0 <= x < cols):
        return False

    glyph = int(glyphs[y, x])
    cmap_idx = glyph - CMAP_OFF

    # Direct floor glyph
    if cmap_idx in FLOOR_CMAP:
        return True

    # Traps are on floor tiles - include them in room detection
    if nethack.glyph_is_trap(glyph):
        return True

    # Item or monster - check if adjacent to known room tile (probably on floor)
    if nethack.glyph_is_object(glyph) or nethack.glyph_is_monster(glyph) or nethack.glyph_is_pet(glyph):
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if (x + dx, y + dy) in room_tiles:
                return True
        # Also check if adjacent to floor glyph directly
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= ny < rows and 0 <= nx < cols:
                adj_cmap = int(glyphs[ny, nx]) - CMAP_OFF
                if adj_cmap in FLOOR_CMAP:
                    return True

    return False


def detect_room(
    glyphs: np.ndarray, pos: tuple[int, int]
) -> set[tuple[int, int]] | None:
    """Flood-fill to detect room from current position.

    Handles items and monsters on floor tiles by checking adjacency.

    Args:
        glyphs: 2D glyph array from observation
        pos: Current (x, y) position

    Returns:
        Set of (x, y) floor tile coordinates, or None if not in a room
    """
    px, py = pos
    rows, cols = glyphs.shape

    if not (0 <= py < rows and 0 <= px < cols):
        return None

    # Player position may not show floor glyph (player/monster/item overlays it)
    # Seed flood-fill from adjacent floor tiles instead
    seed_tiles = []
    cmap_idx = int(glyphs[py, px]) - CMAP_OFF
    if cmap_idx in FLOOR_CMAP:
        seed_tiles.append((px, py))
    else:
        # Check adjacent tiles for floor
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = px + dx, py + dy
            if 0 <= ny < rows and 0 <= nx < cols:
                adj_cmap = int(glyphs[ny, nx]) - CMAP_OFF
                if adj_cmap in FLOOR_CMAP:
                    seed_tiles.append((nx, ny))

    if not seed_tiles:
        return None  # Not adjacent to any floor tiles

    # Flood fill from seed tiles, including tiles with items/monsters
    room_tiles: set[tuple[int, int]] = set()
    # Include player position even though glyph differs (player/monster overlay)
    room_tiles.add((px, py))
    queue = list(seed_tiles)

    while queue:
        x, y = queue.pop()
        if (x, y) in room_tiles:
            continue
        if not (0 <= y < rows and 0 <= x < cols):
            continue

        # Check if this is a room tile (floor, or item/monster on floor)
        if not _is_room_tile(glyphs, x, y, room_tiles):
            continue

        room_tiles.add((x, y))
        # Add cardinal neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            queue.append((x + dx, y + dy))

    # Check if it's actually a room (not corridor)
    # Relaxed thresholds for dark rooms where only nearby tiles visible
    if len(room_tiles) < 3:
        return None  # Too small (1-2 tiles is just a doorway)

    xs = [t[0] for t in room_tiles]
    ys = [t[1] for t in room_tiles]
    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1

    # Room if either: spans 2+ in both dimensions, OR has 4+ tiles
    # This allows dark rooms (small visible area) while rejecting linear corridors
    if width < 2 or height < 2:
        if len(room_tiles) < 4:
            return None  # Linear, probably corridor

    return room_tiles


def _is_corridor_tile(glyphs: np.ndarray, x: int, y: int) -> bool:
    """Check if tile is corridor glyph."""
    rows, cols = glyphs.shape
    if not (0 <= y < rows and 0 <= x < cols):
        return False
    cmap_idx = int(glyphs[y, x]) - CMAP_OFF
    return cmap_idx in CORRIDOR_CMAP


def _is_monster_on_corridor(
    glyphs: np.ndarray, x: int, y: int, known_corridor: set[tuple[int, int]]
) -> bool:
    """Check if tile has monster/pet that's likely on a corridor.

    A monster adjacent to known corridor tiles is probably standing on corridor.
    """
    rows, cols = glyphs.shape
    if not (0 <= y < rows and 0 <= x < cols):
        return False

    glyph = int(glyphs[y, x])
    if not (nethack.glyph_is_monster(glyph) or nethack.glyph_is_pet(glyph)):
        return False

    # Check if adjacent to known corridor
    for dy, dx in DIRS.values():
        if (x + dx, y + dy) in known_corridor:
            return True
    # Also check if adjacent to corridor glyph
    for dy, dx in DIRS.values():
        if _is_corridor_tile(glyphs, x + dx, y + dy):
            return True
    return False


def detect_corridor(
    glyphs: np.ndarray, pos: tuple[int, int]
) -> list[tuple[int, int]] | None:
    """Detect corridor tiles from current position.

    Handles pets/monsters standing on corridor tiles by checking adjacency.

    Args:
        glyphs: 2D glyph array from observation
        pos: Current (x, y) position

    Returns:
        Ordered list of corridor tiles, or None if not in a corridor
    """
    px, py = pos
    rows, cols = glyphs.shape

    if not (0 <= py < rows and 0 <= px < cols):
        return None

    # First, find ALL visible corridor tiles on the map
    all_corridor_tiles: set[tuple[int, int]] = set()
    for row in range(rows):
        for col in range(cols):
            if _is_corridor_tile(glyphs, col, row):
                all_corridor_tiles.add((col, row))

    # Player position may not show corridor glyph (player/monster overlay)
    # Seed flood-fill from current or adjacent corridor tiles
    seed_tiles = []
    if _is_corridor_tile(glyphs, px, py):
        seed_tiles.append((px, py))
    else:
        # Check adjacent tiles for corridor
        for dy, dx in DIRS.values():
            nx, ny = px + dx, py + dy
            if _is_corridor_tile(glyphs, nx, ny):
                seed_tiles.append((nx, ny))

    if not seed_tiles:
        return None  # Not adjacent to any corridor tiles

    # Flood fill on corridor tiles, including monster/pet positions
    corridor_tiles: set[tuple[int, int]] = set()
    # Include player position (might be standing on corridor)
    corridor_tiles.add((px, py))
    queue = list(seed_tiles)

    while queue:
        x, y = queue.pop()
        if (x, y) in corridor_tiles:
            continue
        if not (0 <= y < rows and 0 <= x < cols):
            continue

        # Accept if corridor glyph OR monster adjacent to known corridor
        is_corridor = _is_corridor_tile(glyphs, x, y)
        is_monster_on_corr = _is_monster_on_corridor(glyphs, x, y, corridor_tiles)

        if not (is_corridor or is_monster_on_corr):
            continue

        corridor_tiles.add((x, y))
        # Add all 8 neighbors for corridors
        for dy, dx in DIRS.values():
            queue.append((x + dx, y + dy))

    # Also include any corridor tiles that are connected via known corridor
    # (handles cases where tiles are diagonally connected through player/monster)
    changed = True
    while changed:
        changed = False
        for tile in list(all_corridor_tiles - corridor_tiles):
            x, y = tile
            # Check if adjacent to any known corridor tile
            for dy, dx in DIRS.values():
                if (x + dx, y + dy) in corridor_tiles:
                    corridor_tiles.add(tile)
                    changed = True
                    break

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
    """Order perimeter tiles for efficient clockwise walk around the room.

    Uses connected-component walking to trace the perimeter rather than
    greedy nearest-neighbor which can jump around and miss tiles.
    """
    if not perimeter:
        return []

    perimeter_set = set(perimeter)

    # Start from nearest to current position
    first = min(perimeter_set, key=lambda p: distance(start[0], start[1], p[0], p[1]))

    # Walk around the perimeter by following adjacent tiles
    # Priority: clockwise order (E, SE, S, SW, W, NW, N, NE)
    clockwise_dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    ordered = [first]
    visited = {first}
    current = first

    while len(visited) < len(perimeter_set):
        # Find next unvisited perimeter tile, preferring clockwise order
        next_tile = None
        for dx, dy in clockwise_dirs:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in perimeter_set and neighbor not in visited:
                next_tile = neighbor
                break

        if next_tile is None:
            # No adjacent unvisited - jump to nearest unvisited
            unvisited = perimeter_set - visited
            if not unvisited:
                break
            next_tile = min(unvisited, key=lambda p: distance(current[0], current[1], p[0], p[1]))

        ordered.append(next_tile)
        visited.add(next_tile)
        current = next_tile

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

    # Find tiles with items/monsters that need to be walkable for pathfinding
    extra_walkable = {pos}
    rows, cols = glyphs.shape
    for x, y in room_tiles:
        if 0 <= y < rows and 0 <= x < cols:
            glyph = int(glyphs[y, x])
            if nethack.glyph_is_object(glyph) or nethack.glyph_is_monster(glyph) or nethack.glyph_is_pet(glyph):
                extra_walkable.add((x, y))

    # Generate actions
    actions = []
    current = pos

    for target in ordered:
        # Path to target
        path = pathfind(glyphs, current, target, extra_walkable)
        if path is not None:  # None = no path, [] = already there
            actions.extend(path)
            current = target
            # Search at wall
            actions.append("search")

    return actions


def _find_monsters_on_corridor(
    glyphs: np.ndarray, corridor_set: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    """Find monster/pet positions that overlap with corridor tiles."""
    monsters = set()
    rows, cols = glyphs.shape
    for x, y in corridor_set:
        if 0 <= y < rows and 0 <= x < cols:
            glyph = int(glyphs[y, x])
            if nethack.glyph_is_monster(glyph) or nethack.glyph_is_pet(glyph):
                monsters.add((x, y))
    return monsters


def _find_corridor_frontiers(
    glyphs: np.ndarray, corridor_set: set[tuple[int, int]]
) -> list[tuple[int, int, str]]:
    """Find corridor tiles adjacent to unexplored areas.

    Returns list of (x, y, directions) where directions indicates unexplored neighbors.
    """
    rows, cols = glyphs.shape
    frontiers = []

    for x, y in corridor_set:
        unexplored_dirs = []
        for dir_name, (dy, dx) in DIRS.items():
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                if glyphs[ny, nx] == S_STONE:
                    unexplored_dirs.append(dir_name[0].upper())
        if unexplored_dirs:
            frontiers.append((x, y, "".join(sorted(set(unexplored_dirs)))))

    return frontiers


def _count_walkable_neighbors(
    glyphs: np.ndarray, tile: tuple[int, int], corridor_set: set[tuple[int, int]]
) -> int:
    """Count walkable neighbors (corridor, door, floor) for dead-end detection."""
    x, y = tile
    rows, cols = glyphs.shape
    count = 0
    for dy, dx in DIRS.values():
        nx, ny = x + dx, y + dy
        if 0 <= ny < rows and 0 <= nx < cols:
            # Count as neighbor if: corridor tile, or walkable cmap (door/floor/etc)
            if (nx, ny) in corridor_set:
                count += 1
            else:
                cmap_idx = int(glyphs[ny, nx]) - CMAP_OFF
                if cmap_idx in WALKABLE_CMAP:
                    count += 1
    return count


def plan_corridor_exploration(
    glyphs: np.ndarray, corridor: list[tuple[int, int]], pos: tuple[int, int]
) -> list[str]:
    """Plan corridor exploration prioritizing unexplored areas.

    Strategy:
    1. Find exploration frontiers (corridor tiles adjacent to unexplored)
    2. Find dead-ends (tiles with only 1 walkable neighbor)
    3. Visit frontiers first (prioritize discovery), then dead-ends
    4. Mark all corridor tiles + pet/monster positions as walkable for pathfinding

    Args:
        glyphs: 2D glyph array from observation
        corridor: Ordered list of corridor tiles
        pos: Current (x, y) position

    Returns:
        List of actions (directions + "search")
    """
    if not corridor:
        return []

    corridor_set = set(corridor)

    # All corridor tiles should be walkable for pathfinding
    # (some may have player/monster glyphs overlaying them)
    extra_walkable = set(corridor_set)
    extra_walkable.add(pos)

    # Find monsters/pets on corridor - also treat as walkable
    monsters = _find_monsters_on_corridor(glyphs, corridor_set)
    extra_walkable |= monsters

    # Find exploration frontiers (tiles adjacent to unexplored stone)
    frontiers = _find_corridor_frontiers(glyphs, corridor_set)

    # Find dead-ends using improved neighbor counting (includes doors/floors)
    dead_ends = [
        t for t in corridor
        if _count_walkable_neighbors(glyphs, t, corridor_set) == 1
    ]

    # Build target list: frontiers first (sorted by distance), then dead-ends
    targets = []

    # Add frontiers (prioritize tiles with more unexplored directions)
    frontier_tiles = [(x, y) for x, y, _ in frontiers]
    frontier_tiles.sort(key=lambda t: (
        -len([d for x, y, d in frontiers if (x, y) == t][0] if frontiers else ""),
        distance(pos[0], pos[1], t[0], t[1])
    ))
    for t in frontier_tiles:
        if t not in targets:
            targets.append(t)

    # Add dead-ends that aren't already in targets
    dead_ends.sort(key=lambda t: distance(pos[0], pos[1], t[0], t[1]))
    for t in dead_ends:
        if t not in targets:
            targets.append(t)

    # If no frontiers or dead-ends, walk to all corridor endpoints
    if not targets:
        # Find endpoints (tiles with only 1 corridor neighbor)
        endpoints = [
            t for t in corridor
            if sum(1 for dy, dx in DIRS.values() if (t[0] + dx, t[1] + dy) in corridor_set) == 1
        ]
        # Sort by distance, furthest first to explore more
        endpoints.sort(key=lambda t: -distance(pos[0], pos[1], t[0], t[1]))
        targets = endpoints

    # Still no targets? Just walk to furthest corridor tile
    if not targets and len(corridor) > 1:
        furthest = max(corridor, key=lambda t: distance(pos[0], pos[1], t[0], t[1]))
        targets = [furthest]

    if not targets:
        return []

    actions = []
    current = pos
    visited_targets: set[tuple[int, int]] = set()

    for target in targets:
        if target in visited_targets:
            continue

        path = pathfind(glyphs, current, target, extra_walkable)
        if path is not None:
            actions.extend(path)
            current = target
            visited_targets.add(target)

            # Search if this is a frontier or dead-end
            is_frontier = target in frontier_tiles
            is_dead_end = target in dead_ends

            if is_frontier or is_dead_end:
                # More searches at frontiers (might find secret doors)
                search_count = 3 if is_frontier else 2
                actions.extend(["search"] * search_count)

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
