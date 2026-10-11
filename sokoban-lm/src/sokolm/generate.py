"""Random level generator.

One sample:
  1. interior walls, at most K = floor(max_wall_frac * interior cells) of them; rejected if the remaining floor is
     not 4-connected
       per_set    (used for the split) every wall set of size <= K is equally likely: k is drawn with P(k)
                  proportional to C(interior, k), then k cells uniformly
       per_count  k ~ Uniform{0..K}, then k cells uniformly; this over-samples rooms with few walls (on 6x6,
                  wall-free rooms were 28% of the samples)
  2. n_boxes targets and n_boxes boxes on distinct floor cells, the player on any other non-box floor cell
     (the player may start on a target)
  3. BFS; rejected if unsolvable

`stream` yields the candidates of one seeded stream before step 3, so a caller can skip the BFS for candidates it
does not need (sokolm.splits does this); `generate` runs the BFS and keeps the solvable ones.
"""

from collections import Counter, deque
from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Iterator, Optional

import numpy as np

from sokolm.sokoban import Board
from sokolm.solver import Solution, solve


@dataclass
class Generated:
    board: Board
    solution: Solution
    seed: int
    index: int  # sample index within the seeded stream (accepted or not)

    @property
    def board_id(self) -> str:
        return f"g{self.board.h}x{self.board.w}-s{self.seed}-{self.index:06d}"


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    first = next(iter(cells))
    seen, todo = {first}, deque([first])
    while todo:
        r, c = todo.popleft()
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if q in cells and q not in seen:
                seen.add(q)
                todo.append(q)
    return len(seen) == len(cells)


WALL_SAMPLING = ("per_set", "per_count")


@lru_cache(maxsize=None)
def _wall_count_probs(n_inner: int, max_walls: int) -> np.ndarray:
    w = np.array([comb(n_inner, k) for k in range(max_walls + 1)], dtype=float)
    return w / w.sum()


def sample_text(rng: np.random.Generator, size: int, n_boxes: int = 1, max_wall_frac: float = 0.25,
                wall_sampling: str = "per_set") -> tuple[Optional[str], str]:
    """One candidate board as XSB text, or (None, reason) when the sample is rejected before solving."""
    inner = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    max_walls = int(max_wall_frac * len(inner))
    if wall_sampling == "per_set":
        k = int(rng.choice(max_walls + 1, p=_wall_count_probs(len(inner), max_walls)))
    elif wall_sampling == "per_count":
        k = int(rng.integers(0, max_walls + 1))
    else:
        raise ValueError(f"wall_sampling must be one of {WALL_SAMPLING}, got {wall_sampling!r}")
    order = rng.permutation(len(inner))
    walls = {inner[i] for i in order[:k]}
    floor = [p for p in inner if p not in walls]
    if not _connected(set(floor)):
        return None, "disconnected"
    if len(floor) < 2 * n_boxes + 1:
        return None, "too_small"
    pick = rng.permutation(len(floor))
    targets = [floor[i] for i in pick[:n_boxes]]
    boxes = [floor[i] for i in pick[n_boxes : 2 * n_boxes]]
    box_set = set(boxes)
    free = [p for p in floor if p not in box_set]
    player = free[int(rng.integers(len(free)))]
    grid = [["#"] * size for _ in range(size)]
    for r, c in floor:
        grid[r][c] = " "
    for r, c in targets:
        grid[r][c] = "."
    for r, c in boxes:
        grid[r][c] = "$"
    r, c = player
    grid[r][c] = "+" if grid[r][c] == "." else "@"
    return "\n".join("".join(row) for row in grid), "ok"


def stream(size: int, seed: int, n_boxes: int = 1, max_wall_frac: float = 0.25,
           wall_sampling: str = "per_set") -> Iterator[tuple[int, Optional[str], str]]:
    """Endless seeded stream of (sample index, XSB text or None, reason) candidates, before the BFS filter."""
    rng = np.random.default_rng(seed)
    index = 0
    while True:
        text, why = sample_text(rng, size, n_boxes, max_wall_frac, wall_sampling)
        yield index, text, why
        index += 1


def generate(n: int, size: int, seed: int, n_boxes: int = 1, max_wall_frac: float = 0.25,
             max_samples: Optional[int] = None, wall_sampling: str = "per_set") -> tuple[list[Generated], Counter]:
    """Up to `n` solvable boards from one seeded stream (at most `max_samples` candidates, default 50 * n), plus
    rejection counts by reason."""
    out, stats = [], Counter()
    max_samples = max_samples or 50 * n
    candidates = stream(size, seed, n_boxes, max_wall_frac, wall_sampling)
    while len(out) < n and stats["samples"] < max_samples:
        index, text, why = next(candidates)
        stats["samples"] += 1
        if text is None:
            stats[why] += 1
            continue
        board = Board(text)
        sol = solve(board)
        if not sol.solvable:
            stats["unsolvable"] += 1
            continue
        stats["accepted"] += 1
        out.append(Generated(board, sol, seed, index))
    return out, stats
