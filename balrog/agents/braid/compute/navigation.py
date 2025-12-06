"""Navigation and scanning helpers for BRAID agent."""

from collections import deque

import numpy as np
from nle import nethack

CMAP_OFF = nethack.GLYPH_CMAP_OFF

# Hunger state names (blstats index 21)
HUNGER_STATES = ["Satiated", "OK", "Hungry", "Weak", "Fainting", "Fainted", "Starved"]

# Dungeon branch names (blstats index 23: dungeon_number)
# Order from NetHack dungeon.def - indices assigned in definition order
DUNGEON_BRANCHES = {
    0: "Dungeons of Doom",
    1: "Gehennom",  # Must be second per dungeon.def
    2: "Gnomish Mines",
    3: "The Quest",
    4: "Sokoban",
    5: "Fort Ludios",
    6: "Vlad's Tower",
    7: "Elemental Planes",
}


def get_branch_name(dungeon_num: int) -> str:
    """Get human-readable branch name from dungeon_number."""
    return DUNGEON_BRANCHES.get(dungeon_num, f"Unknown Branch ({dungeon_num})")

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


def find_monster_positions(glyphs: np.ndarray) -> set[tuple[int, int]]:
    """Find all positions containing monsters or pets.

    These tiles can be walked through (attack monster, swap with pet).
    """
    positions: set[tuple[int, int]] = set()
    rows, cols = glyphs.shape
    for row in range(rows):
        for col in range(cols):
            glyph = int(glyphs[row, col])
            if nethack.glyph_is_monster(glyph) or nethack.glyph_is_pet(glyph):
                positions.add((col, row))  # (x, y)
    return positions


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


def _direction_from(px: int, py: int, tx: int, ty: int) -> str:
    """Get cardinal/ordinal direction from (px,py) to (tx,ty)."""
    dx = tx - px
    dy = ty - py
    if dx == 0 and dy == 0:
        return "here"
    # Determine primary direction
    ns = "N" if dy < 0 else ("S" if dy > 0 else "")
    ew = "E" if dx > 0 else ("W" if dx < 0 else "")
    return ns + ew if ns or ew else "here"


def _is_corridor_tile(glyphs: np.ndarray, x: int, y: int) -> bool:
    """Check if tile is a corridor (not room floor)."""
    rows, cols = glyphs.shape
    if not (0 <= y < rows and 0 <= x < cols):
        return False
    cmap_idx = int(glyphs[y, x]) - CMAP_OFF
    return cmap_idx in CORRIDOR_CMAP  # 21, 22 = corridor, lit corridor


def _count_adjacent_floor(glyphs: np.ndarray, x: int, y: int) -> int:
    """Count adjacent floor/corridor tiles (indicates if in room vs corridor)."""
    rows, cols = glyphs.shape
    count = 0
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < rows and 0 <= nx < cols:
            cmap_idx = int(glyphs[ny, nx]) - CMAP_OFF
            if cmap_idx in WALKABLE_CMAP:
                count += 1
    return count


def find_unexplored(glyphs: np.ndarray, pos: tuple[int, int]) -> str:
    """Find exploration frontiers - places where unexplored areas can be reached.

    Only reports TRUE frontiers:
    - Corridor tiles (# characters) adjacent to unexplored stone (any direction)

    Does NOT report:
    - Room floor tiles (walls around rooms aren't explorable)

    Args:
        glyphs: 2D glyph array from observation
        pos: Current (x, y) position

    Returns:
        Actionable exploration guidance with nearest frontiers and directions.
    """
    px, py = pos
    walkable = build_walkable_mask(glyphs)
    rows, cols = glyphs.shape
    frontiers: list[tuple[int, int, str, int]] = []  # (x, y, unexplored_dirs, distance)

    # All 8 directions - corridors can connect diagonally
    all_dirs = {
        "N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1),
        "NE": (-1, 1), "NW": (-1, -1), "SE": (1, 1), "SW": (1, -1),
    }

    # Check each walkable tile for adjacent unexplored (stone) tiles
    for row in range(rows):
        for col in range(cols):
            if not walkable[row, col]:
                continue

            # Only corridor tiles are true frontiers
            # Room floor tiles adjacent to stone are just walls, not exploration opportunities
            if not _is_corridor_tile(glyphs, col, row):
                continue

            # Check all 8 neighbors for unexplored tiles
            unexplored_dirs = []
            for dir_name, (dy, dx) in all_dirs.items():
                ny, nx = row + dy, col + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    if glyphs[ny, nx] == S_STONE:
                        unexplored_dirs.append(dir_name)

            if not unexplored_dirs:
                continue

            d = distance(px, py, col, row)
            dirs_str = ",".join(sorted(unexplored_dirs))
            frontiers.append((col, row, dirs_str, d))

    if not frontiers:
        return "FULLY EXPLORED - no unexplored corridor endpoints. Search walls for secret doors or use stairs."

    # Sort by distance
    frontiers.sort(key=lambda f: f[3])

    # Build actionable output
    lines = []
    for i, (x, y, dirs, d) in enumerate(frontiers[:5]):
        direction = _direction_from(px, py, x, y)
        if d == 0:
            lines.append(f"- HERE @{x},{y}: unexplored {dirs}")
        else:
            lines.append(f"- {direction} @{x},{y} (dist {d}): unexplored {dirs}")

    total = len(frontiers)
    header = f"{total} frontier(s) found. Nearest:"
    suggestion = f"\nSuggestion: travel_to @{frontiers[0][0]},{frontiers[0][1]} to explore {_direction_from(px, py, frontiers[0][0], frontiers[0][1])}"

    return header + "\n" + "\n".join(lines) + suggestion


# Cmap indices for doors and corridors
S_VCDOOR = 12  # Doorway
S_HODOOR = 13  # Open door horizontal
S_VODOOR = 14  # Open door vertical
S_CORR = 21    # Corridor
S_LITCORR = 22  # Lit corridor
DOOR_CMAP = {12, 13, 14, 15, 16}  # All door types
CORRIDOR_CMAP = {21, 22}


# Floor cmap indices for room detection
# Includes floor tiles plus room features (altar, fountain, sink, throne)
FLOOR_CMAP = {19, 20}  # S_room, S_darkroom
ROOM_FEATURE_CMAP = {27, 29, 30, 31}  # altar, throne, sink, fountain
ROOM_TILE_CMAP = FLOOR_CMAP | ROOM_FEATURE_CMAP


def _is_room_tile(glyphs: np.ndarray, x: int, y: int, room_tiles: set[tuple[int, int]]) -> bool:
    """Check if tile is a room tile or likely on floor (item/monster on floor).

    Room tiles include floor, altar, fountain, sink, throne.
    Items and monsters overlay floor glyphs, so we check adjacency.
    Traps are also included as room tiles (they're on the floor).
    """
    rows, cols = glyphs.shape
    if not (0 <= y < rows and 0 <= x < cols):
        return False

    glyph = int(glyphs[y, x])
    cmap_idx = glyph - CMAP_OFF

    # Direct room tile (floor, altar, fountain, sink, throne)
    if cmap_idx in ROOM_TILE_CMAP:
        return True

    # Traps are on floor tiles - include them in room detection
    if nethack.glyph_is_trap(glyph):
        return True

    # Item or monster - check if adjacent to known room tile (probably on floor)
    if nethack.glyph_is_object(glyph) or nethack.glyph_is_monster(glyph) or nethack.glyph_is_pet(glyph):
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if (x + dx, y + dy) in room_tiles:
                return True
        # Also check if adjacent to room tile glyph directly
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= ny < rows and 0 <= nx < cols:
                adj_cmap = int(glyphs[ny, nx]) - CMAP_OFF
                if adj_cmap in ROOM_TILE_CMAP:
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

    # Player position may not show room glyph (player/monster/item overlays it)
    # Seed flood-fill from adjacent room tiles instead
    seed_tiles = []
    cmap_idx = int(glyphs[py, px]) - CMAP_OFF
    if cmap_idx in ROOM_TILE_CMAP:
        seed_tiles.append((px, py))
    else:
        # Check adjacent tiles for room tiles
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = px + dx, py + dy
            if 0 <= ny < rows and 0 <= nx < cols:
                adj_cmap = int(glyphs[ny, nx]) - CMAP_OFF
                if adj_cmap in ROOM_TILE_CMAP:
                    seed_tiles.append((nx, ny))

    if not seed_tiles:
        return None  # Not adjacent to any room tiles

    # Flood fill from seed tiles, including tiles with items/monsters
    room_tiles: set[tuple[int, int]] = set()
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

    # Include player position even if glyph differs (player/monster overlay)
    room_tiles.add((px, py))

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


def plan_visited_aware_exploration(
    glyphs: np.ndarray,
    walkable_tiles: set[tuple[int, int]],
    pos: tuple[int, int],
    visited: set[tuple[int, int]],
) -> list[str]:
    """Plan room exploration by walking unvisited perimeter tiles.

    Only walks wall-adjacent tiles (perimeter) since inner tiles don't need
    to be visited - items/monsters are visible from anywhere in a lit room.
    Searches at each perimeter tile for secret doors.

    Args:
        glyphs: 2D glyph array from observation
        walkable_tiles: Set of tiles known/believed to be walkable (room floor)
        pos: Current (x, y) position
        visited: Set of tiles player has stepped on this episode/level

    Returns:
        List of actions to walk perimeter with searches
    """
    # Find perimeter tiles (wall-adjacent) - only these need to be walked
    perimeter = set(_find_perimeter(walkable_tiles))
    unvisited_perimeter = perimeter - visited

    if not unvisited_perimeter:
        # All perimeter tiles visited - room fully explored
        return []

    # Only walk unvisited perimeter tiles
    targets = unvisited_perimeter

    # BFS tour through targets, using visited as known-walkable
    # Combine visited (confirmed) + walkable_tiles (visible) for pathfinding
    extra_walkable = visited | walkable_tiles | {pos}

    # Also add monster/pet positions as walkable (can swap places)
    # Check both within walkable_tiles AND adjacent to them (pet might be blocking path)
    rows, cols = glyphs.shape
    tiles_to_check = set(walkable_tiles)
    for x, y in walkable_tiles:
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= ny < rows and 0 <= nx < cols:
                tiles_to_check.add((nx, ny))

    for x, y in tiles_to_check:
        if 0 <= y < rows and 0 <= x < cols:
            glyph = int(glyphs[y, x])
            if nethack.glyph_is_monster(glyph) or nethack.glyph_is_pet(glyph):
                extra_walkable.add((x, y))

    # Use ordered perimeter walk for systematic coverage instead of greedy nearest
    ordered_targets = _order_perimeter(list(targets), pos)

    actions: list[str] = []
    current = pos

    for target in ordered_targets:
        path = pathfind(glyphs, current, target, extra_walkable)

        if path is None:
            # No path found - try from current position with all walkable as extra
            # This handles cases where the path requires going through visited tiles
            path = pathfind(glyphs, current, target, extra_walkable | walkable_tiles)

        if path is None:
            # Still no path - skip this target but continue with others
            continue

        actions.extend(path)
        current = target  # Only update position after successful path

        # Search if at perimeter
        if target in perimeter:
            actions.append("search")

        # Limit actions per call to avoid very long queues
        if len(actions) > 200:
            break

    return actions


def _find_room_frontiers(
    glyphs: np.ndarray, room_tiles: set[tuple[int, int]]
) -> list[tuple[int, int, str]]:
    """Find room tiles adjacent to unexplored stone (potential room continuation).

    Also considers tiles adjacent to darkness (space glyph) as frontiers,
    since dark rooms may extend beyond visible area.

    Returns list of (x, y, directions) where directions indicates unexplored neighbors.
    """
    rows, cols = glyphs.shape
    frontiers = []

    # S_STONE = 0 (unexplored), but also check for darkness (space character)
    # In dark rooms, unexplored areas show as space (glyph 32 or similar)
    for x, y in room_tiles:
        unexplored_dirs = []
        for dir_name, (dy, dx) in DIRS.items():
            nx, ny = x + dx, y + dy
            if 0 <= ny < rows and 0 <= nx < cols:
                glyph = glyphs[ny, nx]
                # Stone (unexplored) or space (dark/unexplored)
                if glyph == S_STONE or glyph == CMAP_OFF + 0:  # S_STONE
                    unexplored_dirs.append(dir_name[0])  # First letter
        if unexplored_dirs:
            frontiers.append((x, y, "".join(sorted(set(unexplored_dirs)))))

    return frontiers


def plan_room_exploration(
    glyphs: np.ndarray,
    room_tiles: set[tuple[int, int]],
    pos: tuple[int, int],
    visited: set[tuple[int, int]] | None = None,
) -> list[str]:
    """Plan room exploration using visited-aware algorithm.

    Handles lit, dark, and partially lit rooms uniformly by prioritizing
    unvisited tiles and using visited tiles as known-walkable for pathfinding.

    Args:
        glyphs: 2D glyph array from observation
        room_tiles: Set of room floor tiles
        pos: Current (x, y) position
        visited: Set of tiles player has stepped on (optional)

    Returns:
        List of actions (directions + "search")
    """
    # Find tiles with items/monsters that need to be walkable for pathfinding
    extra_walkable = {pos}
    rows, cols = glyphs.shape
    for x, y in room_tiles:
        if 0 <= y < rows and 0 <= x < cols:
            glyph = int(glyphs[y, x])
            if nethack.glyph_is_object(glyph) or nethack.glyph_is_monster(glyph) or nethack.glyph_is_pet(glyph):
                extra_walkable.add((x, y))

    # Use visited-aware exploration if visited data available
    if visited:
        actions = plan_visited_aware_exploration(glyphs, room_tiles, pos, visited)
        if actions:
            return actions
        # All visible tiles visited - check for frontiers to expand into

        # Find room frontiers (tiles adjacent to unexplored stone)
        frontiers = _find_room_frontiers(glyphs, room_tiles)
        if frontiers:
            # Pick frontier furthest from current position (explore outward)
            frontier_tiles = [(x, y) for x, y, _ in frontiers]
            target = max(frontier_tiles, key=lambda t: distance(pos[0], pos[1], t[0], t[1]))
            path = pathfind(glyphs, pos, target, extra_walkable | visited)
            if path:
                actions = list(path)
                # Get the direction(s) this frontier leads to unexplored area
                for fx, fy, dirs in frontiers:
                    if (fx, fy) == target and dirs:
                        # Continue in the first unexplored direction to reveal more
                        dir_map = {"n": "north", "s": "south", "e": "east", "w": "west"}
                        first_dir = dirs[0].lower()
                        if first_dir in dir_map:
                            actions.extend([dir_map[first_dir]] * 3)
                            actions.append("search")
                        break
                return actions

        # Dark room detection: small visible area might mean more room exists
        # If room is small and no frontiers, try walking to furthest unvisited perimeter tile
        if len(room_tiles) <= 15:  # Small visible area suggests dark room
            perimeter = set(_find_perimeter(room_tiles))
            unvisited_perimeter = perimeter - visited
            if unvisited_perimeter:
                # Walk to furthest unvisited perimeter tile
                target = max(unvisited_perimeter, key=lambda t: distance(pos[0], pos[1], t[0], t[1]))
                path = pathfind(glyphs, pos, target, extra_walkable | visited | room_tiles)
                if path:
                    actions = list(path)
                    actions.append("search")
                    return actions
            # All perimeter visited in small room - room is fully explored
            # Don't try walking into walls, that causes infinite loops

        # All tiles visited, no frontiers - room fully explored
        return []

    # Fallback: perimeter walk (only when no visited data available)
    perimeter_list = _find_perimeter(room_tiles)
    if not perimeter_list:
        return []

    ordered = _order_perimeter(perimeter_list, pos)

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
