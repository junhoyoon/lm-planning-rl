"""The 8 symmetries of a square board (4 rotations, each optionally mirrored) as permutations of cell indices.

Two boards related by one of them are the same puzzle seen in another orientation: the optimal solution maps along
with the board. De-duplication and splits can therefore treat the 8 variants as one board. The canonical form of a
board is the smallest of its 8 variant texts.
"""

from functools import lru_cache


@lru_cache(maxsize=None)
def cell_maps(size: int) -> tuple[tuple[int, ...], ...]:
    """maps[k][i] = the cell that cell i (= r * size + c) moves to under symmetry k; maps[0] is the identity."""
    n = size - 1
    moves = (
        lambda r, c: (r, c),          # identity
        lambda r, c: (c, n - r),      # rotate 90 degrees clockwise
        lambda r, c: (n - r, n - c),  # rotate 180 degrees
        lambda r, c: (n - c, r),      # rotate 270 degrees clockwise
        lambda r, c: (r, n - c),      # mirror left-right
        lambda r, c: (n - r, c),      # mirror top-bottom
        lambda r, c: (c, r),          # mirror on the main diagonal
        lambda r, c: (n - c, n - r),  # mirror on the anti-diagonal
    )
    maps = []
    for f in moves:
        m = []
        for i in range(size * size):
            r, c = f(*divmod(i, size))
            m.append(r * size + c)
        maps.append(tuple(m))
    return tuple(maps)


def variants(text: str) -> list[str]:
    """The 8 variants of a square XSB board text; the first one is the text itself."""
    rows = text.split("\n")
    size = len(rows)
    flat = "".join(rows)
    if len(flat) != size * size:
        raise ValueError("variants() needs a square board with full rows")
    out = []
    for m in cell_maps(size):
        cells = [""] * (size * size)
        for i, ch in enumerate(flat):
            cells[m[i]] = ch
        out.append("\n".join("".join(cells[r * size:(r + 1) * size]) for r in range(size)))
    return out


def canonical(text: str) -> str:
    return min(variants(text))


def layout(text: str) -> str:
    """The board without its box and player: walls, floor and targets only."""
    return text.translate(str.maketrans({"$": " ", "@": " ", "*": ".", "+": "."}))
