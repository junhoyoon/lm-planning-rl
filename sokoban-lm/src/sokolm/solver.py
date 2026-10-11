"""Breadth-first search over Sokoban states.

Every action costs 1 (moves and pushes alike), so BFS gives the minimum number of actions: the optimal solution
length that defines difficulty and the solver-informed action budget. No-op actions (wall bumps, blocked pushes)
never enter the search because they do not change the state.

The search runs layer by layer (layer d = states first reached after d actions) and also counts, for every state,
how many distinct shortest action sequences reach it. That gives the number of optimal solutions; the returned one
is the first solved state discovered, with actions tried in U, D, L, R order (deterministic tie-breaking).
"""

from dataclasses import dataclass, field
from typing import Optional

from sokolm.sokoban import NOOP_EVENTS, Board, State, step


@dataclass
class Solution:
    actions: Optional[list[int]]  # one optimal action sequence; None when unsolvable
    n_states: int  # distinct states reached (search stops after the layer holding the first solved state)
    n_optimal: int  # number of distinct optimal action sequences (0 when unsolvable)
    layer_sizes: list[int] = field(default_factory=list)  # states first reached after d actions, d = 0, 1, ...

    @property
    def solvable(self) -> bool:
        return self.actions is not None

    @property
    def length(self) -> Optional[int]:
        return None if self.actions is None else len(self.actions)


def solve(board: Board, start: Optional[State] = None) -> Solution:
    start = board.start if start is None else start
    if board.is_solved(start):
        return Solution([], 1, 1, [1])
    depth = {start: 0}
    parent: dict[State, tuple[State, int]] = {}
    count = {start: 1}  # number of shortest action sequences from `start` to each state
    layer, layer_sizes, goals = [start], [1], []
    d = 0
    while layer and not goals:
        nxt = []
        for s in layer:
            for a in range(4):
                t, ev = step(board, s, a)
                if ev in NOOP_EVENTS:
                    continue
                if t not in depth:
                    depth[t], parent[t], count[t] = d + 1, (s, a), count[s]
                    nxt.append(t)
                    if board.is_solved(t):
                        goals.append(t)
                elif depth[t] == d + 1:
                    count[t] += count[s]  # another shortest way into a state of the next layer
        layer, d = nxt, d + 1
        if nxt:
            layer_sizes.append(len(nxt))
    if not goals:
        return Solution(None, len(depth), 0, layer_sizes)
    actions, s = [], goals[0]
    while s != start:
        s, a = parent[s]
        actions.append(a)
    return Solution(actions[::-1], len(depth), sum(count[g] for g in goals), layer_sizes)


def solvable(board: Board, s: State) -> bool:
    return solve(board, s).solvable
