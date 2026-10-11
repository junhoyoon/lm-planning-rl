"""Execute a parsed action sequence on a board and judge it (the reward function of single-turn Sokoban).

Rules:
  - wall bumps and blocked pushes are no-ops that still use one action of the budget
  - execution stops at the first solved state; the remaining actions are counted, not executed
  - at most `budget` actions are executed (budget = 2 x optimal length); the actions beyond it are counted
  - reward = 1 if the board was solved within the budget, else 0; no penalty for no-ops, detours or length
  - a parse failure gets reward 0; so does a truncated output (token limit reached without the end token), even
    if its actions solved the board
Text-level cases (parse failure, truncation by the token limit) are detected by the output parser, not here.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

from sokolm.sokoban import BLOCKED, PUSH, WALL, Board, State, step


@dataclass
class Outcome:
    budget: int
    n_actions: int  # actions in the parsed output
    solved: bool = False  # a solved state was reached within the budget
    solved_at: Optional[int] = None  # number of actions executed when the board became solved
    events: list[str] = field(default_factory=list)  # one per executed action: move / push / wall / blocked
    states: list[State] = field(default_factory=list)  # state before the first action and after each executed one

    @property
    def n_executed(self) -> int:
        return len(self.events)

    @property
    def n_wall(self) -> int:
        return self.events.count(WALL)

    @property
    def n_blocked(self) -> int:
        return self.events.count(BLOCKED)

    @property
    def n_noop(self) -> int:
        return self.n_wall + self.n_blocked

    @property
    def n_pushes(self) -> int:
        return self.events.count(PUSH)

    @property
    def n_after_solve(self) -> int:
        """Output actions after the solving action (not executed)."""
        return self.n_actions - self.solved_at if self.solved else 0

    @property
    def n_over_budget(self) -> int:
        """Output actions beyond the budget (never executed). Can overlap with n_after_solve."""
        return max(0, self.n_actions - self.budget)


def simulate(board: Board, actions: Sequence[int], budget: int) -> Outcome:
    s = board.start
    out = Outcome(budget=budget, n_actions=len(actions), states=[s])
    for i, a in enumerate(actions[:budget]):
        s, ev = step(board, s, a)
        out.events.append(ev)
        out.states.append(s)
        if board.is_solved(s):
            out.solved, out.solved_at = True, i + 1
            break
    return out


def reward(outcome: Optional[Outcome], parse_ok: bool = True, truncated: bool = False) -> float:
    """Binary reward (rules above). `outcome` is None when the output did not parse."""
    return float(parse_ok and not truncated and outcome is not None and outcome.solved)


def judge(o: Optional[Outcome], budget: int, opt_len: Optional[int] = None, parse_ok: bool = True,
          truncated: bool = False) -> dict:
    """Reward plus the separately logged fields as one flat record (`o` is None when parsing failed)."""
    rec = {
        "reward": reward(o, parse_ok, truncated),
        "parse_ok": parse_ok,
        "truncated": truncated,
        "budget": budget,
        "opt_len": opt_len,
    }
    if o is not None:
        rec |= {
            "solved": o.solved,
            "solved_at": o.solved_at,
            "n_actions": o.n_actions,
            "n_executed": o.n_executed,
            "n_noop": o.n_noop,
            "n_wall": o.n_wall,
            "n_blocked": o.n_blocked,
            "n_pushes": o.n_pushes,
            "n_after_solve": o.n_after_solve,
            "n_over_budget": o.n_over_budget,
            "len_ratio": round(o.solved_at / opt_len, 3) if o.solved and opt_len else None,
        }
    return rec
