"""Evaluation on fixed boards.

Greedy solve rate: one greedy answer per board; the share of boards it solves (reward 1).
pass@k: n answers per board sampled at temperature T from the model's full distribution (T = 1, n = 16). With c of
them solving the board,
    pass@k = 1 - C(n - c, k) / C(n, k),
the probability that k answers drawn without replacement from the n include one that solves it (the unbiased
estimator of Chen et al., 2021), averaged over boards. Only k <= n is reported.
mixed: the share of boards whose n samples neither all solve nor all fail, i.e. whose group of n answers would have
rewards that differ, so that GRPO gets a nonzero advantage from it.

Board-blind baselines, the reward of answers that ignore the board, on the same boards:
  swap   the greedy answer of every other board of the same bucket, judged on this board with this board's budget,
         averaged over all ordered pairs of different boards. An answer that does not depend on its board solves
         other boards as often as its own; the greedy solve rate minus this value is the part that depends on it.
  fixed  per optimal length, the best answer of at most FIXED_MAX moves, given unchanged to every board of that
         length.

GRPO groups: a group is the g completions of one board in one optimizer step. When its rewards are all equal
(all solve or all fail, zero std) all its advantages are zero and it adds no gradient. `zero_std_prediction` predicts
the share of such groups from n sampled answers per board (`mixed_at`); `group_rows` and `group_summary` measure it in
the rollouts of a run.
"""

import itertools
from collections import Counter, defaultdict
from math import comb
from typing import Optional

import numpy as np

from sokolm.formats import EXAMPLE_ACTIONS, Format, answer_text, judge_completion, parse_answer, prompt_text
from sokolm.simulate import simulate
from sokolm.sokoban import Board

KS = (1, 2, 4, 8, 16)
FIXED_MAX = 4


def pass_at_k(n: int, c: int, k: int) -> float:
    """1 - C(n - c, k) / C(n, k): n answers, c of them solve, k drawn without replacement."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n, got n={n}, c={c}, k={k}")
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def mixed_at(n: int, c: int, g: int) -> float:
    """1 - [C(c, g) + C(n - c, g)] / C(n, g): n answers, c of them solve, g drawn without replacement; the chance that
    the g include both a solving and a failing answer. With g = the GRPO group size, the chance that a group's rewards
    differ, i.e. that its advantages are not all zero (C(c, g) = 0 when g > c)."""
    if not 0 <= c <= n or not 1 <= g <= n:
        raise ValueError(f"need 0 <= c <= n and 1 <= g <= n, got n={n}, c={c}, g={g}")
    return 1.0 - (comb(c, g) + comb(n - c, g)) / comb(n, g)


def zero_std_prediction(sample_recs: list[dict], shares: dict[int, float], g: int) -> dict:
    """The share of GRPO groups of g completions whose rewards are all equal, predicted from n sampled answers per
    board (records of `answer_records`, every board with the same n). For board b with c_b solving answers, drawing g
    of its n gives all solving answers with probability C(c_b, g) / C(n, g) and all failing ones with probability
    C(n - c_b, g) / C(n, g). Per optimal length L these are averaged over the boards of that length; `shares` = the
    share w_L of length L among the boards trained on, and
        Z = sum_L w_L (all_solved_L + all_failed_L),
    the predicted share of groups with zero reward std, i.e. no gradient."""
    per_board: dict[str, list[float]] = defaultdict(list)
    length = {}
    for r in sample_recs:
        per_board[r["board_id"]].append(r["reward"])
        length[r["board_id"]] = r["opt_len"]
    ns = {len(v) for v in per_board.values()}
    if len(ns) != 1:
        raise ValueError("every board needs the same number of samples")
    n = ns.pop()
    if not 1 <= g <= n:
        raise ValueError(f"groups of {g} need at least {g} samples per board, got {n}")
    by_len: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for bid, v in per_board.items():
        c = int(round(sum(v)))
        by_len[length[bid]].append((comb(c, g) / comb(n, g), comb(n - c, g) / comb(n, g)))
    if set(shares) - set(by_len):
        raise ValueError(f"no sampled boards of length {sorted(set(shares) - set(by_len))}")
    rows = [{"opt_len": L, "boards": len(by_len[L]), "share": round(shares[L], 4),
             "all_solved": round(float(np.mean([a for a, _ in by_len[L]])), 4),
             "all_failed": round(float(np.mean([b for _, b in by_len[L]])), 4)} for L in sorted(shares)]
    for r in rows:
        r["zero_std"] = round(r["all_solved"] + r["all_failed"], 4)
    total = {k: round(sum(shares[r["opt_len"]] * r[k] for r in rows), 4) for k in ("zero_std", "all_solved",
                                                                                    "all_failed")}
    return {"g": g, "n": n, "by_length": rows, **total}


def first_action(completion: str, fmt: Format) -> Optional[int]:
    """The first action of an answer, None when it does not parse or has no action."""
    actions, _ = parse_answer(completion, fmt)
    return actions[0] if actions else None


def group_rows(recs: list[dict], fmt: Format) -> list[dict]:
    """One row per GRPO group in rollout records (one record per completion, with step, board_id, opt_len, reward,
    completion): how many of its completions solve, whether its rewards are all equal (zero std: all solve or all
    fail), its number of distinct completion texts and of distinct first actions (an unparsed answer counts as one
    more first action, None)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        groups[(r["step"], r["board_id"])].append(r)
    rows = []
    for (step, bid), sel in groups.items():
        solved = int(round(sum(r["reward"] for r in sel)))
        rows.append({"step": step, "board_id": bid, "opt_len": sel[0]["opt_len"], "completions": len(sel),
                     "solved": solved, "zero_std": solved in (0, len(sel)), "all_solved": solved == len(sel),
                     "all_failed": solved == 0, "distinct_texts": len({r["completion"] for r in sel}),
                     "distinct_first": len({first_action(r["completion"], fmt) for r in sel})})
    return rows


def group_summary(rows: list[dict], bands=((1, 4), (5, 8), (9, 13))) -> dict:
    """Over GRPO group rows (`group_rows`): their number, the shares with all rewards equal (zero std), all solved and
    all failed, the mean numbers of distinct texts and first actions; and the zero-std share per band of optimal
    lengths (None for a band without groups)."""
    def share(sel, key):
        return round(float(np.mean([r[key] for r in sel])), 4) if sel else None

    out = {"groups": len(rows), "zero_std": share(rows, "zero_std"), "all_solved": share(rows, "all_solved"),
           "all_failed": share(rows, "all_failed"), "distinct_texts": share(rows, "distinct_texts"),
           "distinct_first": share(rows, "distinct_first")}
    for lo, hi in bands:
        sel = [r for r in rows if lo <= r["opt_len"] <= hi]
        out[f"zero_std_{lo}_{hi}"] = share(sel, "zero_std")
        out[f"groups_{lo}_{hi}"] = len(sel)
    return out


def answer_records(model, tok, boards: list[dict], fmt: Format, cfg, mode: str, n: int = 1, seed: int = 0,
                   batch_size: int = 512) -> list[dict]:
    """Generate n answers per board with generation config `cfg` and judge them; one record per answer, boards in
    the given order and the n answers of a board together. `ids` holds the generated token ids."""
    from sokolm.lm import generate

    prompts = [prompt_text(tok, b["text"], fmt) for b in boards]
    comps = generate(model, tok, prompts, cfg, n=n, batch_size=batch_size, seed=seed)
    recs = []
    for b, cs in zip(boards, comps):
        board = Board(b["text"])
        for k, c in enumerate(cs):
            judged = judge_completion(board, c.text, fmt, b["budget"], b["opt_len"], truncated=c.truncated)
            recs.append({"board_id": b["board_id"], "bucket": b["bucket"], "mode": mode, "sample": k,
                         "completion": c.text, "n_tokens": len(c.ids), "end_token": c.end_token, "ids": c.ids,
                         **judged})
    return recs


def _mean(xs) -> Optional[float]:
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else None


def summary_rows(recs: list[dict], fmt: Format, ks=KS, by_length: bool = False) -> list[dict]:
    """One row per (mode, bucket), or per (mode, bucket, optimal length) with by_length. Shares are over answers
    unless named per board. greedy: solve_rate. sample: pass@k for every k <= n, and mixed."""
    example = answer_text(EXAMPLE_ACTIONS, fmt)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        groups[(r["mode"], r["bucket"]) + ((r["opt_len"],) if by_length else ())].append(r)
    rows = []
    for key in sorted(groups, key=lambda k: (k[0] != "greedy",) + k[1:]):
        sel = groups[key]
        per_board: dict[str, list[float]] = defaultdict(list)
        for r in sel:
            per_board[r["board_id"]].append(r["reward"])
        texts = Counter(r["completion"] for r in sel)
        top, top_n = texts.most_common(1)[0]
        row = {"mode": key[0], "bucket": key[1]} | ({"opt_len": key[2]} if by_length else {})
        row |= {"boards": len(per_board), "answers": len(sel)}
        if key[0] == "greedy":
            row["solve_rate"] = _mean(r["reward"] for r in sel)
        else:
            n = min(len(v) for v in per_board.values())
            if any(len(v) != n for v in per_board.values()):
                raise ValueError("every board needs the same number of samples")
            cs = [int(sum(v)) for v in per_board.values()]
            for k in ks:
                if k <= n:
                    row[f"pass@{k}"] = _mean(pass_at_k(n, c, k) for c in cs)
            row["mixed"] = _mean(0 < c < n for c in cs)
        row |= {
            "parse_fail": _mean(not r["parse_ok"] for r in sel),
            "truncated": _mean(r["truncated"] for r in sel),
            "noop_mean": _mean(r["n_noop"] for r in sel if r["parse_ok"]),
            "len_ratio_mean": _mean(r["len_ratio"] for r in sel if r["reward"] == 1),
            "tokens_median": float(np.median([r["n_tokens"] for r in sel])),
            "example_share": _mean(r["completion"].strip() == example for r in sel),
            "distinct_texts": len(texts), "top_text": top, "top_count": top_n,
        }
        rows.append(row)
    return rows


def swap_baseline(boards: list[dict], greedy: list[dict], fmt: Format) -> list[dict]:
    """Per bucket: the greedy solve rate (own) and the swap baseline (module docstring)."""
    answer = {r["board_id"]: r for r in greedy}
    rows = []
    for bucket in dict.fromkeys(b["bucket"] for b in boards):
        sel = [b for b in boards if b["bucket"] == bucket]
        parsed = []
        for b in sel:
            r = answer[b["board_id"]]
            actions, _ = parse_answer(r["completion"], fmt)
            parsed.append(None if r["truncated"] else actions)  # a truncated answer gets reward 0 on any board
        solved = 0
        for i, b in enumerate(sel):
            board = Board(b["text"])
            solved += sum(simulate(board, a, b["budget"]).solved for j, a in enumerate(parsed) if j != i and a)
        n = len(sel)
        own = _mean(answer[b["board_id"]]["reward"] for b in sel)
        swap = round(solved / (n * (n - 1)), 4) if n > 1 else None
        rows.append({"bucket": bucket, "boards": n, "own": own, "swap": swap,
                     "own_minus_swap": None if swap is None else round(own - swap, 4)})
    return rows


def fixed_baseline(boards: list[dict], max_len: int = FIXED_MAX) -> list[dict]:
    """Per (bucket, optimal length): the answer of 1..max_len moves that solves the most boards of that group when
    given to all of them, and the share it solves (ties: the shorter, then the first in up/down/left/right order).
    An answer q of m moves solves a board iff the max_len-move sequence q + (up, ..., up) first solves it within
    min(budget, m) executed moves, so only the 4^max_len full sequences are executed."""
    full = list(itertools.product(range(4), repeat=max_len))
    index = {s: i for i, s in enumerate(full)}
    cands = [q for m in range(1, max_len + 1) for q in itertools.product(range(4), repeat=m)]
    first = np.full((len(boards), len(full)), max_len + 1)
    for i, b in enumerate(boards):
        board = Board(b["text"])
        for j, s in enumerate(full):
            o = simulate(board, s, min(b["budget"], max_len))
            if o.solved:
                first[i, j] = o.solved_at
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, b in enumerate(boards):
        groups[(b["bucket"], b["opt_len"])].append(i)
    rows = []
    for (bucket, L), idx in groups.items():
        f = first[idx]
        best, rate = None, -1.0
        for q in cands:
            v = float(np.mean(f[:, index[q + (0,) * (max_len - len(q))]] <= len(q)))
            if v > rate:
                best, rate = q, v
        rows.append({"bucket": bucket, "opt_len": L, "boards": len(idx), "rate": round(rate, 4),
                     "answer": " ".join(("up", "down", "left", "right")[a] for a in best)})
    return sorted(rows, key=lambda r: (r["bucket"], r["opt_len"]))
