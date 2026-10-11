"""The 8 board symmetries used to identify problems for the split."""

from sokolm.sokoban import Board
from sokolm.solver import solve
from sokolm.symmetry import canonical, cell_maps, layout, variants


def test_symmetry_maps_form_a_group():
    maps = cell_maps(6)
    assert len(set(maps)) == 8 and all(sorted(m) == list(range(36)) for m in maps)
    composed = {tuple(m2[m1[i]] for i in range(36)) for m1 in maps for m2 in maps}
    assert composed == set(maps)


def test_variants_keep_the_optimal_length():
    text = "######\n# @. #\n#  $ #\n#  ###\n#    #\n######"  # optimal length 14
    lengths = {solve(Board(v)).length for v in variants(text)}
    assert lengths == {14}
    assert canonical(text) in variants(text)
    assert layout(text) == "######\n#  . #\n#    #\n#  ###\n#    #\n######"
