"""Sokoban state and transition core, kept free of rendering, parsing and reward so that a later multi-turn setup can
reuse it unchanged.

Board text uses the standard XSB characters:
  '#' wall, ' ' floor, '.' target, '$' box, '*' box on target, '@' player, '+' player on target.
Sizes are whole sizes including the outer wall ring (a 6x6 board has a 4x4 interior).

Cells are integer indices `r * w + c`. A `State` is the dynamic part (player cell, box cells); walls and targets
are static and live on the `Board`.
"""

import hashlib
from typing import NamedTuple

ACTIONS = "UDLR"  # action index -> letter
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # (row, col) offset of U, D, L, R
REVERSE = (1, 0, 3, 2)  # opposite action of U, D, L, R

# Events of one transition. "wall" (walking into a wall) and "blocked" (pushing a box into a wall or another box)
# leave the state unchanged: they are no-ops but still use one action of the budget.
MOVE, PUSH, WALL, BLOCKED = "move", "push", "wall", "blocked"
NOOP_EVENTS = (WALL, BLOCKED)


class State(NamedTuple):
    player: int
    boxes: frozenset  # of cell indices


class Board:
    """Static part of a level (size, walls, targets) plus its start state, built from XSB text."""

    def __init__(self, text: str):
        rows = text.strip("\n").split("\n")
        self.h, self.w = len(rows), max(len(r) for r in rows)
        rows = [r.ljust(self.w) for r in rows]
        walls, targets, boxes, player = set(), set(), set(), None
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                i = r * self.w + c
                if ch == "#":
                    walls.add(i)
                elif ch not in " .$*@+":
                    raise ValueError(f"unknown board character {ch!r} at ({r},{c})")
                if ch in ".*+":
                    targets.add(i)
                if ch in "$*":
                    boxes.add(i)
                if ch in "@+":
                    if player is not None:
                        raise ValueError("more than one player")
                    player = i
        ring = [r * self.w + c for r in range(self.h) for c in range(self.w) if r in (0, self.h - 1) or c in (0, self.w - 1)]
        if not all(i in walls for i in ring):
            raise ValueError("the outer ring must be walls")
        if player is None or not boxes or len(boxes) != len(targets):
            raise ValueError(f"need one player and as many boxes as targets (boxes={len(boxes)}, targets={len(targets)})")
        self.walls, self.targets = frozenset(walls), frozenset(targets)
        self.start = State(player, frozenset(boxes))

        def neighbour(i: int, dr: int, dc: int) -> int:
            r, c = divmod(i, self.w)
            j = (r + dr) * self.w + (c + dc)
            return -1 if j in walls else j

        # nbr[cell][action] -> neighbouring cell, or -1 when it is a wall (the outer ring keeps indices in range)
        self.nbr = tuple(
            (-1, -1, -1, -1) if i in walls else tuple(neighbour(i, dr, dc) for dr, dc in DELTAS)
            for i in range(self.h * self.w)
        )

    def rc(self, i: int) -> tuple[int, int]:
        return divmod(i, self.w)

    def is_solved(self, s: State) -> bool:
        return s.boxes == self.targets

    def render(self, s: State | None = None) -> str:
        """XSB text of the board in state `s` (default: the start state)."""
        s = self.start if s is None else s
        out = []
        for r in range(self.h):
            row = []
            for c in range(self.w):
                i = r * self.w + c
                if i in self.walls:
                    row.append("#")
                elif i in s.boxes:
                    row.append("*" if i in self.targets else "$")
                elif i == s.player:
                    row.append("+" if i in self.targets else "@")
                else:
                    row.append("." if i in self.targets else " ")
            out.append("".join(row))
        return "\n".join(out)

    def sha1(self) -> str:
        """Hash of the start-state text: the identity used for de-duplication and splits."""
        return hashlib.sha1(self.render().encode()).hexdigest()


def step(board: Board, s: State, a: int) -> tuple[State, str]:
    """Apply action `a` (0..3 = U, D, L, R) to state `s` -> (next state, event)."""
    p1 = board.nbr[s.player][a]
    if p1 < 0:
        return s, WALL
    if p1 not in s.boxes:
        return State(p1, s.boxes), MOVE
    b1 = board.nbr[p1][a]  # where the pushed box would go
    if b1 < 0 or b1 in s.boxes:
        return s, BLOCKED
    return State(p1, (s.boxes - {p1}) | {b1}), PUSH


def parse_actions(letters: str) -> list[int]:
    """'RUU' -> [3, 0, 0]. Strict: only the letters U, D, L, R (the LM output parser is separate)."""
    return [ACTIONS.index(ch) for ch in letters]


def action_letters(actions) -> str:
    return "".join(ACTIONS[a] for a in actions)
