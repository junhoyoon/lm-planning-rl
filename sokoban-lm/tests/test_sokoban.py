"""Light checks of the transition rules, the BFS solver and the reward function."""

import pytest

from sokolm.generate import generate
from sokolm.simulate import judge, simulate
from sokolm.sokoban import BLOCKED, MOVE, PUSH, WALL, Board, parse_actions, step
from sokolm.solver import solve

# the worked example of docs/guide/rl-quick-guide.ko.md §5: optimal R U U
GUIDE = """\
######
#  . #
#    #
#  $ #
# @  #
######"""

TWO_BOXES = """\
######
#    #
# $$@#
#..  #
#    #
######"""


def test_events():
    b = Board(GUIDE)
    s = b.start
    assert step(b, s, parse_actions("D")[0]) == (s, WALL)  # wall below the player
    t, ev = step(b, s, parse_actions("L")[0])
    assert ev == MOVE and b.rc(t.player) == (4, 1)
    t, _ = step(b, s, parse_actions("R")[0])
    u, ev = step(b, t, parse_actions("U")[0])
    assert ev == PUSH and {b.rc(i) for i in u.boxes} == {(2, 3)} and b.rc(u.player) == (3, 3)
    b2 = Board(TWO_BOXES)
    assert step(b2, b2.start, parse_actions("L")[0]) == (b2.start, BLOCKED)  # box behind box


def test_blocked_by_wall():
    b = Board("#####\n#@$ #\n#  .#\n#####")
    t, ev = step(b, b.start, parse_actions("R")[0])
    assert ev == PUSH
    assert step(b, t, parse_actions("R")[0]) == (t, BLOCKED)  # box against the right wall


def test_render_roundtrip():
    for text in (GUIDE, TWO_BOXES, "#####\n#+$ #\n#   #\n#.$ #\n#####"):
        assert Board(text).render() == text


def test_solve_known():
    sol = solve(Board(GUIDE))
    assert sol.actions == parse_actions("RUU") and sol.n_optimal == 1
    assert solve(Board("#####\n#$  #\n# @.#\n#####")).solvable is False  # box in a corner


def test_n_optimal_counts_paths():
    # the player reaches the cell left of the box with D D R in any order (3 routes), then pushes R
    b = Board("######\n#@   #\n#    #\n#  $.#\n######")
    sol = solve(b)
    assert sol.length == 4 and sol.n_optimal == 3
    # brute force over all action sequences of the optimal length
    from itertools import product

    n = sum(simulate(b, list(seq), sol.length).solved_at == sol.length for seq in product(range(4), repeat=sol.length))
    assert n == sol.n_optimal


def test_simulate_rules():
    b = Board(GUIDE)
    opt = parse_actions("RUU")
    r = judge(simulate(b, opt, 6), 6, 3)
    assert r["reward"] == 1 and r["solved_at"] == 3 and r["n_noop"] == 0 and r["len_ratio"] == 1
    r = judge(simulate(b, parse_actions("D") + opt, 6), 6, 3)  # wall bump counts in the budget
    assert r["reward"] == 1 and r["solved_at"] == 4 and r["n_wall"] == 1
    r = judge(simulate(b, opt + parse_actions("LL"), 6), 6, 3)  # stops at the solve
    assert r["reward"] == 1 and r["n_executed"] == 3 and r["n_after_solve"] == 2
    r = judge(simulate(b, parse_actions("LRLR") + opt, 6), 6, 3)  # solve would come at action 7 > budget 6
    assert r["reward"] == 0 and r["n_executed"] == 6 and r["n_over_budget"] == 1
    r = judge(None, 6, 3, parse_ok=False)
    assert r["reward"] == 0
    r = judge(simulate(b, opt, 6), 6, 3, truncated=True)
    assert r["reward"] == 0


@pytest.mark.parametrize("size", [6, 8])
def test_generated_solutions_replay(size):
    gens, stats = generate(50, size, seed=123)
    assert len(gens) == 50
    for g in gens:
        o = simulate(g.board, g.solution.actions, len(g.solution.actions))
        assert o.solved_at == g.solution.length and o.n_noop == 0
