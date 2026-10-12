"""Animated figure of the per-board solve counts from the base model through SFT and GRPO (board-solves.svg).

    python board_solves.py                 writes board-solves.svg from board-solves.json and prints the counts
    python board_solves.py --extract R1 R6 first rebuilds board-solves.json from episodes.jsonl in the run
                                           directories R1 (run 1) and R6 (run 6), as `grpo` writes them, and from
                                           the base model's evaluation run in runs/

board-solves.json holds, for each of the 300 val in_dist boards, the number c of its 16 sampled answers that solve
it (reward 1): for the base model, from its evaluation run (runs/, BASE_RUN), and at every in-training evaluation of
run 1 (GRPO from the SFT model) and run 6 (GRPO from run 4), at steps 0, 100, ..., 1,500 and 1,562. All sample at
temperature 1; the in-training evaluations use the same seed every time. Except for the base model, these are not the
evaluation runs of report.md §5, which sampled the final checkpoints separately; runs/ keeps the training runs'
metrics but not these answers.

The figure has one square per board, in the same place in every frame and in both rows: columns by optimal length,
and within a length by the SFT model's c (run 1 at step 0), highest at the bottom. Boards with 1 <= c <= 15 are
orange, nearer the background color as c grows (the only boards whose sampled answers get mixed rewards); boards with
c = 0 are gray and boards with c = 16 blue. Both rows start at the base model and move to the SFT model. Then the
bottom row moves to run 4 (run 6 at step 0) while the top row does not train, and both rows run GRPO, one frame per
in-training evaluation. The SFT runs recorded no sampled answers during training, so their segments have a frame at
each end only. Colors change linearly between frames. The animation is CSS only, without script, so it plays where
an SVG is shown as an image, such as a README. Standard library only.
"""

import argparse
import gzip
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "board-solves.json"
FIGURE = HERE / "board-solves.svg"
BASE_RUN = "20261009-233904-p3-1-eval"
N_SAMPLES = 16
ROWS = ("run1", "run6")

# Colors. Orange (the hue of categorical slot 2) for c = 1..15, at OKLab lightness from LIGHTNESS[0] (c = 1) to
# LIGHTNESS[1] (c = 15); gray for c = 0 and a step of the sequential blue for c = 16.
ORANGE = "#eb6834"
LIGHTNESS = {"light": (0.50, 0.82), "dark": (0.82, 0.52)}
ENDS = {"light": ("#52514e", "#9ec5f4"), "dark": ("#898781", "#1c5cab")}  # c = 0, c = 16
CHROME = {"light": {"bg": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "rule": "#c3c2b7",
                    "track": "#e1e0d9"},
          "dark": {"bg": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "rule": "#6b6a65",
                   "track": "#383835"}}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"

# Timeline of one loop, in seconds.
LOOP = 20.0
BASE_END = 1.5              # both rows show the base model until here
SFT_AT = 3.5                # both rows reach the SFT model
SFT_END = 4.5               # the bottom row leaves the SFT model
RUN4_AT = 6.0               # the bottom row reaches run 4: both rows are at GRPO step 0
GRPO_START, GRPO_END = 7.2, 16.0  # GRPO leaves step 0 and reaches the last step, at a constant pace
FADE_OUT, HIDDEN, RESET, FADE_IN = 19.0, 19.3, 19.4, 19.5  # the loop fades out, resets to the base model, fades in

# Geometry (viewBox units).
WIDTH = 880
LEFT = 24
CELL, GAP = 11, 2
PITCH = CELL + GAP
PER_ROW = 4                 # squares per row within one optimal length
BLOCK_GAP = 10
TRACK = (110, 300, 410, 760)  # x of the base model, the SFT model, GRPO step 0 and the last GRPO step
LANES = {"run1": 88, "run6": 108}  # y of each row's lane in the timeline
COLS = (650, 742, 826)       # x centers of the class-count columns


# ------------------------------------------------------------------------------------------------------------ data


def _sample_counts(path: Path, opener=open) -> tuple[dict[int, dict[str, int]], dict[str, int]]:
    """{step: {board: c}} and {board: optimal length} from the sampled in_dist answers of one episodes file."""
    c: dict[int, dict[str, int]] = {}
    n: dict[tuple, int] = {}
    order: dict[str, int] = {}
    with opener(path, "rt") as f:
        for r in map(json.loads, f):
            if r["bucket"] != "in_dist" or r["mode"] != "sample":
                continue
            order.setdefault(r["board_id"], r["opt_len"])  # answers are recorded in val-file order
            at = r.get("step") or 0  # an evaluation run records no step
            step = c.setdefault(at, {})
            step[r["board_id"]] = step.get(r["board_id"], 0) + (r["reward"] == 1)
            n[at, r["board_id"]] = n.get((at, r["board_id"]), 0) + 1
    if len(order) != 300 or set(n.values()) != {N_SAMPLES} or len(n) != 300 * len(c):
        raise ValueError(f"{path}: expected 300 boards with {N_SAMPLES} sampled answers at every evaluation")
    return c, order


def extract(run1: Path, run6: Path) -> dict:
    """Solve counts per board: the base model's evaluation run and every in-training evaluation of runs 1 and 6."""
    base, order = _sample_counts(HERE / "runs" / BASE_RUN / "episodes.jsonl.gz", gzip.open)
    counts = {key: _sample_counts(d / "episodes.jsonl")[0] for key, d in zip(ROWS, (run1, run6))}
    steps = sorted(counts[ROWS[0]])
    if sorted(counts[ROWS[1]]) != steps:
        raise ValueError("the two runs were evaluated at different steps")
    (base,) = base.values()
    return {"runs": {"base": BASE_RUN, **{key: d.name for key, d in zip(ROWS, (run1, run6))}},
            "samples_per_board": N_SAMPLES, "steps": steps,
            "boards": [{"board_id": b, "opt_len": L, "base": base[b],
                        **{key: [counts[key][s][b] for s in steps] for key in ROWS}} for b, L in order.items()]}


def write_data(data: dict) -> None:
    """One board per line, so that the file stays readable and diffs stay small."""
    head = {k: v for k, v in data.items() if k != "boards"}
    lines = ",\n".join("  " + json.dumps(b) for b in data["boards"])
    DATA.write_text(json.dumps(head)[:-1] + ',\n "boards": [\n' + lines + "\n ]\n}\n")


def classes(cs: list[int]) -> tuple[int, int, int]:
    """Boards solved by 0, by 1-15 and by all 16 of the samples."""
    return sum(c == 0 for c in cs), sum(0 < c < N_SAMPLES for c in cs), sum(c == N_SAMPLES for c in cs)


def summary(data: dict) -> list[str]:
    """The counts the README caption quotes."""
    b = data["boards"]
    base = [x["base"] for x in b]
    sft = [x["run1"][0] for x in b]
    run4 = [x["run6"][0] for x in b]
    end1 = [x["run1"][-1] for x in b]
    end6 = [x["run6"][-1] for x in b]
    lines = []
    for name, cs in (("base model", base), ("SFT (run 1, step 0)", sft), ("run 4 (run 6, step 0)", run4),
                     ("run 1, last step", end1), ("run 6, last step", end6)):
        z, m, f = classes(cs)
        lines.append(f"{name}: solved by 0 / 1-15 / 16 of 16 samples: {z} / {m} / {f} boards; "
                     f"mean c/16 {sum(cs) / len(cs) / N_SAMPLES:.3f}")
    for name, start, end in (("SFT -> run 4", sft, run4), ("SFT -> run 1, last step", sft, end1),
                             ("run 4 -> run 6, last step", run4, end6)):
        z, m, f = classes(start)
        moved = {"0": [e for s, e in zip(start, end) if s == 0],
                 "1-15": [e for s, e in zip(start, end) if 0 < s < N_SAMPLES],
                 "16": [e for s, e in zip(start, end) if s == N_SAMPLES]}
        lines.append(f"{name}: of the {z} boards at 0, {sum(e > 0 for e in moved['0'])} above 0; "
                     f"of the {m} at 1-15, {sum(e == N_SAMPLES for e in moved['1-15'])} at 16 and "
                     f"{sum(e == 0 for e in moved['1-15'])} at 0; "
                     f"of the {f} at 16, {sum(e < N_SAMPLES for e in moved['16'])} below 16")
    both = sum(0 < s < N_SAMPLES and e == N_SAMPLES and r == N_SAMPLES for s, e, r in zip(sft, end1, run4))
    lines.append(f"boards at 1-15 for SFT that reached 16 both in run 4 and at the end of run 1: {both}")
    return lines


# ----------------------------------------------------------------------------------------------------------- color


def _to_oklab(hex_color: str) -> tuple[float, float, float]:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb]
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def _linear_rgb(L: float, a: float, b: float) -> tuple[float, float, float]:
    l_, m_, s_ = L + 0.3963377774 * a + 0.2158037573 * b, L - 0.1055613458 * a - 0.0638541728 * b, \
        L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


def _hex(rgb: tuple[float, float, float]) -> str:
    out = [12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055 for x in rgb]
    return "#" + "".join(f"{round(min(1.0, max(0.0, x)) * 255):02x}" for x in out)


def level_colors(mode: str) -> list[str]:
    """Color of c = 0..16: ORANGE's hue at the lightness of LIGHTNESS for 1..15, with the largest chroma up to 0.19
    that stays in sRGB; ENDS for 0 and 16."""
    _, a, b = _to_oklab(ORANGE)
    hue = math.atan2(b, a)
    lo, hi = LIGHTNESS[mode]
    out = [ENDS[mode][0]]
    for c in range(1, N_SAMPLES):
        L = lo + (hi - lo) * (c - 1) / (N_SAMPLES - 2)
        chroma = 0.19
        while not all(-1e-4 <= x <= 1 + 1e-4 for x in _linear_rgb(L, chroma * math.cos(hue), chroma * math.sin(hue))):
            chroma -= 0.002
        out.append(_hex(_linear_rgb(L, chroma * math.cos(hue), chroma * math.sin(hue))))
    return out + [ENDS[mode][1]]


# -------------------------------------------------------------------------------------------------------- timeline


def pct(t: float) -> str:
    return f"{100 * t / LOOP:.1f}".rstrip("0").rstrip(".") + "%"


def grpo_time(step: int, last: int) -> float:
    return GRPO_START + (GRPO_END - GRPO_START) * step / last


def frames(data: dict, row: str) -> list[tuple[float, float, list[int]]]:
    """(arrival time, departure time, counts) of every frame of one row of the figure."""
    steps, b = data["steps"], data["boards"]
    out = [(0.0, BASE_END, [x["base"] for x in b]),
           (SFT_AT, SFT_END if row == "run6" else GRPO_START, [x["run1"][0] for x in b])]
    if row == "run6":
        out.append((RUN4_AT, GRPO_START, [x["run6"][0] for x in b]))
    for k, s in enumerate(steps[1:], 1):
        t = grpo_time(s, steps[-1])
        out.append((t, HIDDEN if k == len(steps) - 1 else t, [x[row][k] for x in b]))
    return out


def windows(fr: list[tuple[float, float, list[int]]]) -> list[tuple[float, float]]:
    """When each frame's labels show: from halfway through the change into the frame to halfway through the change
    out of it. The last frame shows until the reset; the first also shows again after it."""
    cuts = [(dep + nxt[0]) / 2 for (_, dep, _), nxt in zip(fr, fr[1:])]
    return list(zip([0.0] + cuts, cuts + [RESET]))


# ---------------------------------------------------------------------------------------------------------- figure


def keyframes(name: str, stops: list[tuple[float, str]], prop: str) -> str:
    """@keyframes with the stops merged by value; the stops' times must be distinct after rounding."""
    by_value: dict[str, list[str]] = {}
    for t, v in stops:
        by_value.setdefault(v, []).append(pct(t))
    body = "".join(f"{','.join(ts)}{{{prop}:{v}}}" for v, ts in by_value.items())
    return f"@keyframes {name}{{{body}}}"


def thin(stops: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """Drop stops inside a run of equal values; the value between the kept stops is unchanged."""
    return [s for i, s in enumerate(stops)
            if i in (0, len(stops) - 1) or not (stops[i - 1][1] == s[1] == stops[i + 1][1])]


def cell_positions(data: dict) -> tuple[list[tuple[float, float]], list[tuple[int, float]], int]:
    """(x, y) of each board's square relative to the bottom-left of a grid, the x center of each optimal length's
    column, and the number of square rows of the tallest column."""
    b = data["boards"]
    lengths = sorted({x["opt_len"] for x in b})
    pos: list[tuple[float, float]] = [(0.0, 0.0)] * len(b)
    centers, tallest = [], 0
    block = PER_ROW * PITCH - GAP
    for j, L in enumerate(lengths):
        x0 = j * (block + BLOCK_GAP)
        ids = sorted((i for i, x in enumerate(b) if x["opt_len"] == L),
                     key=lambda i: (-b[i]["run1"][0], b[i]["board_id"]))
        for k, i in enumerate(ids):
            pos[i] = (x0 + (k % PER_ROW) * PITCH, -(k // PER_ROW) * PITCH - CELL)
        centers.append((L, x0 + block / 2))
        tallest = max(tallest, -(-len(ids) // PER_ROW))
    return pos, centers, tallest


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(data: dict) -> str:
    steps = data["steps"]
    pos, centers, tallest = cell_positions(data)
    grid_w = centers[-1][1] + (PER_ROW * PITCH - GAP) / 2
    grid_x = LEFT
    grid_h = tallest * PITCH - GAP
    css: list[str] = []
    body: list[str] = []
    dyn: list[str] = []

    def text(x, y, s, cls="t2", size=12, anchor="start", weight=None):
        w = f' font-weight="{weight}"' if weight else ""
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        return f'<text x="{x:g}" y="{y:g}" class="{cls}" font-size="{size}"{a}{w}>{esc(s)}</text>'

    def line(x1, y1, x2, y2, cls):
        return f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" class="{cls}"/>'

    # Visibility windows shared by all labels that change with the frames.
    win_names: dict[tuple[float, float], str] = {}

    def during(start: float, end: float, final: bool) -> str:
        """Class and animation of a label shown during [start, end) of every loop (and after the reset if start is
        0). Without animation, the labels of the last frame show."""
        key = (round(start, 3), round(end, 3))
        if key not in win_names:
            name = f"w{len(win_names)}"
            win_names[key] = name
            if start == 0.0:
                stops = [(0.0, "1"), (end, "0"), (RESET, "1"), (LOOP, "1")]
            else:
                stops = [(0.0, "0"), (start, "1"), (end, "0"), (LOOP, "0")]
            css.append(keyframes(name, stops, "opacity"))
        return f' class="sw{" on" if final else ""}" style="animation-name:{win_names[key]}"'

    # ---- header
    body.append(text(LEFT, 28, "300 validation boards from the base model through SFT and GRPO", "t1", 15, weight=600))
    body.append(text(LEFT, 47, "Each square is one in-distribution val board, in the same place in every frame. "
                     "Color: how many of 16 sampled answers solve it.", "t2", 12))

    # ---- timeline: one lane per row, a playhead across both, and the current frame at the right
    x_base, x_sft, x_run4, x_end = TRACK

    def x_at(step: int) -> float:
        return x_run4 + (x_end - x_run4) * step / steps[-1]

    for a, b, label in ((x_base, x_sft, "SFT, 3 epochs"), (x_sft, x_run4, "SFT, lr → 0, 1 epoch"),
                        (x_run4, x_end, f"GRPO, 1 epoch ({steps[-1]:,} steps)")):
        body.append(text((a + b) / 2, 74, label, "t2", 11, anchor="middle"))
    for row, y in LANES.items():
        body.append(text(LEFT, y + 4, "top row" if row == "run1" else "bottom row", "t3", 11))
        body.append(line(x_base, y, x_sft, y, "track"))
        if row == "run6":
            body.append(line(x_sft, y, x_run4, y, "track"))
        else:
            body.append(text((x_sft + x_run4) / 2, y + 4, "no training", "t3", 10.5, anchor="middle"))
        body.append(line(x_run4, y, x_end, y, "track"))
        body.extend(line(x_at(s), y - 3, x_at(s), y + 3, "tick") for s in steps)
        body.extend(f'<circle cx="{x:g}" cy="{y}" r="3" class="mark"/>' for x in (x_base, x_sft, x_run4))
    y_under = LANES["run6"] + 18
    for x, label in ((x_base, "base model"), (x_sft, "SFT model"), (x_run4, "step 0"), (x_end, f"{steps[-1]:,}")):
        body.append(text(x, y_under, label, "t3", 11, anchor="middle"))
    head = [(0.0, x_base), (BASE_END, x_base), (SFT_AT, x_sft), (SFT_END, x_sft), (RUN4_AT, x_run4),
            (GRPO_START, x_run4), (GRPO_END, x_end), (HIDDEN, x_end), (RESET, x_base), (LOOP, x_base)]
    css.append(keyframes("head", [(t, f"translateX({x - x_end:.1f}px)") for t, x in head], "transform"))
    top, bottom = LANES["run1"] - 9, LANES["run6"] + 9
    dyn.append(f'<rect x="{x_end - 1}" y="{top}" width="2" height="{bottom - top}" rx="1" class="head"/>')
    # the current frame: named by the bottom row's frames, which include run 4
    names = ["base model", "SFT model", "bottom row: SFT, lr → 0"] + [f"GRPO step {s:,}" for s in steps]
    fr = frames(data, "run6")
    win = windows(fr)
    spans = [win[0], (win[1][0], SFT_END), (SFT_END, RUN4_AT), (RUN4_AT, win[2][1])] + win[3:]
    for k, ((t0, t1), name) in enumerate(zip(spans, names)):
        label = text(WIDTH - LEFT, 74, name, "t1", 12.5, anchor="end")
        dyn.append(f'<g{during(t0, t1, k == len(names) - 1)}>{label}</g>')

    # ---- column heads for the class counts of both rows: a swatch and the class
    heads = ((0, "0 of 16"), (1, "1–15 of 16"), (2, "16 of 16"))
    body.append(text(COLS[0] - 40, 164, "boards solved by", "t2", 11, anchor="end"))
    for (j, label), cx in zip(heads, COLS):
        if j == 1:
            w = 3
            body.extend(f'<rect x="{cx - (N_SAMPLES - 1) * w / 2 + (c - 1) * w:g}" y="140" width="{w}" height="11" '
                        f'class="c{c}"/>' for c in range(1, N_SAMPLES))
        else:
            c = 0 if j == 0 else N_SAMPLES
            body.append(f'<rect x="{cx - 5.5:g}" y="140" width="11" height="11" rx="2" class="c{c}"/>')
        body.append(text(cx, 164, label, "t2", 11, anchor="middle"))

    # ---- the two grids
    titles = {"run1": "SFT → GRPO (run 1)",
              "run6": "SFT → SFT with the learning rate decayed to 0 (run 4) → GRPO (run 6)"}
    tops = {"run1": 196, "run6": 196 + 12 + grid_h + 51}
    traj_names: dict[tuple, str] = {}
    for row in ROWS:
        top = tops[row]
        bottom = top + 12 + grid_h
        fr = frames(data, row)
        body.append(text(LEFT, top, titles[row], "t1", 13, weight=600))
        for k, ((t0, t1), (_, _, cs)) in enumerate(zip(windows(fr), fr)):
            counts = classes(cs)
            labels = "".join(text(cx, top, str(counts[j]), "t1", 13, anchor="middle", weight=600)
                             for j, cx in enumerate(COLS))
            dyn.append(f'<g{during(t0, t1, k == len(fr) - 1)}>{labels}</g>')
        cells = []
        for i, (dx, dy) in enumerate(pos):
            values = [cs[i] for _, _, cs in fr]
            stops: list[tuple[float, str]] = []
            for arrival, departure, cs in fr:
                stops.append((arrival, f"var(--c{cs[i]})"))
                if departure != arrival:
                    stops.append((departure, f"var(--c{cs[i]})"))
            stops = thin(stops + [(RESET, f"var(--c{values[0]})"), (LOOP, f"var(--c{values[0]})")])
            attr = ""
            if len({v for _, v in stops}) > 1:
                key = tuple(stops)
                if key not in traj_names:
                    traj_names[key] = f"k{len(traj_names)}"
                    css.append(keyframes(traj_names[key], stops, "fill"))
                attr = f' style="animation-name:{traj_names[key]}"'
            cells.append(f'<rect x="{grid_x + dx:g}" y="{bottom + dy:g}" width="{CELL}" height="{CELL}" rx="2" '
                         f'class="c{values[-1]}{" sq" if attr else ""}"{attr}/>')
        dyn.append("".join(cells))
        for L, cx in centers:
            body.append(text(grid_x + cx, bottom + 15, str(L), "t3", 11, anchor="middle"))
    last_bottom = tops["run6"] + 12 + grid_h
    body.append(text(grid_x + grid_w / 2, last_bottom + 31, "optimal solution length", "t3", 11, anchor="middle"))
    body.append(text(LEFT, last_bottom + 52, "16 answers per board, sampled at temperature 1. Base model: its "
                     "evaluation run. The SFT segments were recorded at their start and end only.", "t3", 11))
    body.append(text(LEFT, last_bottom + 67, "All other frames: the in-training evaluations of runs 1 and 6, every "
                     f"{steps[1] - steps[0]} GRPO steps, with the same seed each time.", "t3", 11))
    height = last_bottom + 81

    # ---- style
    root = []
    for mode in ("light", "dark"):
        decl = "".join(f"--{k}:{x};" for k, x in CHROME[mode].items())
        root.append(decl + "".join(f"--c{c}:{x};" for c, x in enumerate(level_colors(mode))))
    style = (f"svg{{{root[0]}font-family:{FONT};font-variant-numeric:tabular-nums}}"
             f"@media (prefers-color-scheme:dark){{svg{{{root[1]}}}}}"
             ".bg{fill:var(--bg)}.t1{fill:var(--ink)}.t2{fill:var(--ink2)}.t3{fill:var(--muted)}"
             ".track{stroke:var(--track);stroke-width:3;stroke-linecap:round}.tick{stroke:var(--rule);stroke-width:1}"
             ".mark{fill:var(--rule)}.head{fill:var(--ink);animation-name:head}"
             + "".join(f".c{c}{{fill:var(--c{c})}}" for c in range(N_SAMPLES + 1))
             + f".sq,.head,.sw,.fade{{animation-duration:{LOOP:g}s;animation-iteration-count:infinite}}"
             ".sq,.head{animation-timing-function:linear}"
             ".sw{opacity:0;animation-timing-function:step-end}.sw.on{opacity:1}"
             ".fade{animation-name:fade;animation-timing-function:linear}"
             + keyframes("fade", [(0.0, "1"), (FADE_OUT, "1"), (HIDDEN, "0"), (FADE_IN, "0"), (LOOP, "1")],
                         "opacity")
             + "".join(css)
             + "@media (prefers-reduced-motion:reduce){.sq,.head,.sw,.fade{animation:none}}")
    desc = ("Animation of 300 squares in two rows, one square per in-distribution validation board, colored by how "
            "many of 16 sampled answers solve the board: orange for 1 to 15, gray for none, blue for all. "
            "Both rows start at the base model, which solves almost no board, and move to the SFT model, which "
            "solves most boards with every sample. The bottom row then trains one more SFT epoch with the learning "
            "rate decayed to 0 (run 4) while the top row waits, and both rows run GRPO (runs 1 and 6). The bottom "
            "row starts GRPO with about half as many orange boards; in both rows most orange boards turn blue during "
            "GRPO.")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:g}" width="{WIDTH}" '
            f'height="{height:g}" role="img" aria-labelledby="title desc">'
            f'<title id="title">Solve counts of 300 validation boards from the base model through SFT and GRPO</title>'
            f'<desc id="desc">{esc(desc)}</desc>'
            f"<style>{style}</style>"
            f'<rect class="bg" width="{WIDTH}" height="{height:g}" rx="8"/>'
            + "".join(body) + f'<g class="fade">{"".join(dyn)}</g></svg>\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--extract", nargs=2, type=Path, metavar=("RUN1_DIR", "RUN6_DIR"),
                    help="rebuild board-solves.json from these runs' episodes.jsonl first")
    args = ap.parse_args()
    if args.extract:
        write_data(extract(*args.extract))
    data = json.loads(DATA.read_text())
    FIGURE.write_text(svg(data))
    print("\n".join(summary(data)))
    print(f"wrote {FIGURE.name} ({FIGURE.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
