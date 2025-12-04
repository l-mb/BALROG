"""Navigation helpers for BRAID agent - pathfinding on explored areas only."""

from collections import deque

import numpy as np
from nle import nethack

CMAP_OFF = nethack.GLYPH_CMAP_OFF

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
