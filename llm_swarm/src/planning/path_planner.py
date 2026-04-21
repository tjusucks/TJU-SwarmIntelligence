"""Grid A* planner for global route guidance."""

from __future__ import annotations

import heapq
from collections.abc import Iterable

import numpy as np

Cell = tuple[int, int]


def _heuristic(a: Cell, b: Cell) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _neighbors(cell: Cell, max_x: int, max_y: int) -> Iterable[Cell]:
    cx, cy = cell
    for dx, dy in (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ):
        nx, ny = cx + dx, cy + dy
        if 0 <= nx < max_x and 0 <= ny < max_y:
            yield nx, ny


def _is_corner_cut_blocked(curr: Cell, nxt: Cell, occ: np.ndarray) -> bool:
    """Return True when a diagonal move would cut through blocked corners."""
    dx = nxt[0] - curr[0]
    dy = nxt[1] - curr[1]
    if abs(dx) != 1 or abs(dy) != 1:
        return False

    # For diagonal transitions, both axis-adjacent support cells must be free.
    c1 = (curr[0] + dx, curr[1])
    c2 = (curr[0], curr[1] + dy)
    return bool(occ[c1[1], c1[0]] or occ[c2[1], c2[0]])


def _closest_free(cell: Cell, occ: np.ndarray) -> Cell:
    if not occ[cell[1], cell[0]]:
        return cell

    h, w = occ.shape
    visited = {cell}
    queue = [cell]
    while queue:
        new_queue: list[Cell] = []
        for c in queue:
            for nb in _neighbors(c, w, h):
                if nb in visited:
                    continue
                visited.add(nb)
                if not occ[nb[1], nb[0]]:
                    return nb
                new_queue.append(nb)
        queue = new_queue

    return cell


def _compress_path(cells: list[Cell]) -> list[Cell]:
    if len(cells) <= 2:
        return cells

    keep = [cells[0]]
    prev = cells[0]
    curr = cells[1]
    prev_dir = (curr[0] - prev[0], curr[1] - prev[1])

    for nxt in cells[2:]:
        new_dir = (nxt[0] - curr[0], nxt[1] - curr[1])
        if new_dir != prev_dir:
            keep.append(curr)
            prev_dir = new_dir
        prev, curr = curr, nxt

    keep.append(cells[-1])
    return keep


def _line_cells(a: Cell, b: Cell) -> list[Cell]:
    """Return grid cells traversed by a Bresenham-style line segment."""
    x0, y0 = a
    x1, y1 = b

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1 if x1 < x0 else 0
    sy = 1 if y1 > y0 else -1 if y1 < y0 else 0

    x, y = x0, y0
    out: list[Cell] = [(x, y)]

    if dx >= dy:
        err = dx / 2.0
        while x != x1:
            x += sx
            err -= dy
            if err < 0:
                y += sy
                err += dx
            out.append((x, y))
    else:
        err = dy / 2.0
        while y != y1:
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy
            out.append((x, y))

    return out


def _segment_is_free(a: Cell, b: Cell, occ: np.ndarray) -> bool:
    """Check whether a straight grid segment is obstacle-safe."""
    traversed = _line_cells(a, b)
    prev = traversed[0]
    if occ[prev[1], prev[0]]:
        return False

    for cur in traversed[1:]:
        if occ[cur[1], cur[0]]:
            return False
        if _is_corner_cut_blocked(prev, cur, occ):
            return False
        prev = cur
    return True


def _smooth_path(cells: list[Cell], occ: np.ndarray) -> list[Cell]:
    """Greedily skip intermediate waypoints when straight motion is safe."""
    if len(cells) <= 2:
        return cells

    smoothed: list[Cell] = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _segment_is_free(cells[i], cells[j], occ):
            j -= 1
        smoothed.append(cells[j])
        i = j

    return smoothed


def plan_path(
    width: int,
    height: int,
    obstacles: list[tuple[int, int, int, int]],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    cell_size: int = 40,
    inflate_margin: float = 70.0,
) -> list[tuple[float, float]]:
    """Plan a collision-aware route from start to goal.

    Returns:
        World-space waypoint list. If planning fails, returns [goal_xy].
    """
    nx = int(np.ceil(width / cell_size))
    ny = int(np.ceil(height / cell_size))
    occ = np.zeros((ny, nx), dtype=bool)

    for ox, oy, ow, oh in obstacles:
        left = ox - inflate_margin
        right = ox + ow + inflate_margin
        top = oy - inflate_margin
        bottom = oy + oh + inflate_margin

        gx0 = max(0, int(np.floor(left / cell_size)))
        gx1 = min(nx - 1, int(np.ceil(right / cell_size)))
        gy0 = max(0, int(np.floor(top / cell_size)))
        gy1 = min(ny - 1, int(np.ceil(bottom / cell_size)))
        occ[gy0 : gy1 + 1, gx0 : gx1 + 1] = True

    def to_cell(x: float, y: float) -> Cell:
        gx = int(np.clip(np.floor(x / cell_size), 0, nx - 1))
        gy = int(np.clip(np.floor(y / cell_size), 0, ny - 1))
        return gx, gy

    def to_world(c: Cell) -> tuple[float, float]:
        return ((c[0] + 0.5) * cell_size, (c[1] + 0.5) * cell_size)

    start = _closest_free(to_cell(*start_xy), occ)
    goal = _closest_free(to_cell(*goal_xy), occ)

    open_heap: list[tuple[float, Cell]] = []
    heapq.heappush(open_heap, (0.0, start))

    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, float] = {start: 0.0}

    found = False
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            found = True
            break

        for nb in _neighbors(current, nx, ny):
            if occ[nb[1], nb[0]]:
                continue
            if _is_corner_cut_blocked(current, nb, occ):
                continue
            step_cost = _heuristic(current, nb)
            tentative = g_score[current] + step_cost
            if tentative < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative
                f_score = tentative + _heuristic(nb, goal)
                heapq.heappush(open_heap, (f_score, nb))

    if not found:
        return [goal_xy]

    cells = [goal]
    cur = goal
    while cur in came_from:
        cur = came_from[cur]
        cells.append(cur)
    cells.reverse()

    cells = _compress_path(cells)
    cells = _smooth_path(cells, occ)
    waypoints = [to_world(c) for c in cells]

    if len(waypoints) == 0:
        waypoints = [goal_xy]
    if float(np.hypot(waypoints[-1][0] - goal_xy[0], waypoints[-1][1] - goal_xy[1])) > cell_size:
        waypoints.append(goal_xy)
    else:
        waypoints[-1] = goal_xy

    return waypoints
