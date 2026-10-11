"""Tables and figure of the SFT-vs-GRPO report (report.md) from the recorded runs in runs/.

Reads only runs/*/episodes.jsonl.gz, runs/*/metrics.json and runs/*/manifest.json (numpy and matplotlib; no GPU, no
model, no split files). Writes results.json and decomposition.png next to this file and prints the tables.

    python analysis.py          (any Python >= 3.10 with numpy and matplotlib, e.g. `uv run` from ../../sokoban-lm)

Per model and bucket, over the 300 val boards of the bucket (board i, one greedy answer g_i in {0, 1}, c_i of 16
sampled answers at temperature 1 solving it):
    greedy  = mean_i g_i
    pass@1  = mean_i c_i / 16
    pass@16 = mean_i 1[c_i >= 1]
A paired difference a - b is the mean over boards of the per-board difference; its interval is the 2.5 and 97.5
percentiles of that mean over 10,000 bootstrap resamples of the boards. Each interval's resampling seed is the CRC-32
of a key naming the pair, the bucket and the metric (`KEYS`), so one comparison's interval does not depend on which
others are computed.
"""

import gzip
import json
import zlib
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
N_SAMPLES = 16
B_RESAMPLES = 10_000
BUCKETS = ("in_dist", "long", "two_box")

# model -> (eval run, label). The keys also name the bootstrap seeds (see the module docstring).
MODELS = {
    "base": ("20261009-233904-p3-1-eval", "base model"),
    "sft": ("20261010-101128-p3-5-sft-eval", "SFT"),
    "sft_rl": ("20261010-165742-p4-2-eval-sft-rl", "run 1"),
    "sft_ext": ("20261010-165701-p4-2-eval-sft-extend", "run 2"),
    "sft_ext_decay": ("20261010-174229-p4-2-eval-sft-extend-decay", "run 3"),
    "sft_anneal": ("20261010-190020-p4-2-eval-sft-anneal", "run 4"),
    "sft_seen_const": ("20261010-205125-p4-3-eval-sft-seen-constant", "run 5"),
    "rl_anneal": ("20261010-223600-p4-3-eval-grpo-from-anneal", "run 6"),
    "sft_new_from_anneal": ("20261011-001557-p4-3-eval-sft-from-anneal", "run 7"),
}
PAIRS = [
    ("sft", "base"),                        # what SFT adds to the base model
    ("sft_rl", "sft"),                      # run 1 - SFT: GRPO for one more epoch
    ("sft_ext", "sft"),                     # run 2 - SFT: SFT for one more epoch, constant lr
    ("sft_ext_decay", "sft"),               # run 3 - SFT: the same with the lr decayed to 0
    ("sft_anneal", "sft"),                  # run 4 - SFT: lr decayed to 0, on the boards SFT had seen
    ("sft_seen_const", "sft"),              # run 5 - SFT: constant lr, on the boards SFT had seen
    ("sft_rl", "sft_ext"),                  # run 1 - run 2: GRPO against SFT with a constant lr, same boards
    ("sft_ext_decay", "sft_ext"),           # run 3 - run 2: the lr decay alone, on the new boards
    ("sft_anneal", "sft_seen_const"),       # run 4 - run 5: lr decay on the boards SFT had seen
    ("sft_ext_decay", "sft_anneal"),        # run 3 - run 4: new against seen boards, lr decayed to 0
    ("sft_ext", "sft_seen_const"),          # run 2 - run 5: new against seen boards, constant lr
    ("rl_anneal", "sft_anneal"),            # run 6 - run 4: GRPO on top of the annealed SFT
    ("sft_new_from_anneal", "sft_anneal"),  # run 7 - run 4: one more SFT epoch on the same new boards
    ("rl_anneal", "sft_new_from_anneal"),   # run 6 - run 7: GRPO against SFT, same start and boards
    ("sft_rl", "sft_ext_decay"),            # run 1 - run 3: the same comparison from the SFT checkpoint
]
# GRPO run -> (training run, MODELS key of its starting model's eval run).
GRPO_RUNS = {"run 1": ("20261010-140903-p4-2-grpo", "sft"),
             "run 6": ("20261010-214016-p4-3-grpo-from-anneal", "sft_anneal")}
G = 8          # answers per board in a GRPO step
WINDOW = 100   # steps per measured window
# The one-epoch training runs of the report.
EPOCH_RUNS = {"run 1": "20261010-140903-p4-2-grpo", "run 2": "20261010-165157-p4-2-sft-extend",
              "run 3": "20261010-173743-p4-2-sft-extend-decay", "run 4": "20261010-182721-p4-2-sft-anneal",
              "run 5": "20261010-204628-p4-3-sft-seen-constant", "run 6": "20261010-214016-p4-3-grpo-from-anneal",
              "run 7": "20261011-001103-p4-3-sft-from-anneal"}
DECOMPOSITION = [("rl_anneal", "sft_anneal"), ("sft_new_from_anneal", "sft_anneal"), ("rl_anneal", "sft_new_from_anneal")]


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar: n = b + c boards whose greedy outcome differs, each side with probability 1/2 under no
    difference."""
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def boot_ci(d, key: str) -> list[float]:
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    idx = rng.integers(0, len(d), size=(B_RESAMPLES, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def load_eval(run: str) -> dict:
    """Per bucket, in val-file order: board ids, greedy rewards g, solved-sample counts c and optimal lengths L."""
    greedy, c, n, bucket, length = {}, Counter(), Counter(), {}, {}
    with gzip.open(RUNS / run / "episodes.jsonl.gz", "rt") as f:
        for r in map(json.loads, f):
            bucket[r["board_id"]] = r["bucket"]
            length[r["board_id"]] = r["opt_len"]
            if r["mode"] == "greedy":
                greedy[r["board_id"]] = float(r["reward"] == 1)
            else:
                n[r["board_id"]] += 1
                c[r["board_id"]] += r["reward"] == 1
    if len(greedy) != 900 or set(n.values()) != {N_SAMPLES}:
        raise ValueError(f"{run}: expected 900 boards with one greedy and {N_SAMPLES} sampled answers each")
    out = {}
    for bk in BUCKETS:
        ids = [i for i in greedy if bucket[i] == bk]  # greedy answers are recorded in val-file order
        out[bk] = {"ids": ids, "g": np.array([greedy[i] for i in ids]), "c": np.array([c[i] for i in ids], float),
                   "L": np.array([length[i] for i in ids])}
    return out


def metrics(e: dict) -> dict:
    return {"greedy": e["g"], "pass@1": e["c"] / N_SAMPLES, "pass@16": (e["c"] > 0) * 1.0}


def table(ev: dict) -> list[dict]:
    return [{"model": label, "key": key, **{bk: {m: round(float(v.mean()), 4) for m, v in metrics(ev[key][bk]).items()}
                                           for bk in BUCKETS}}
            for key, (_, label) in MODELS.items()]


def paired(ev: dict) -> list[dict]:
    out = []
    for a, b in PAIRS:
        for bk in BUCKETS:
            if ev[a][bk]["ids"] != ev[b][bk]["ids"]:
                raise ValueError(f"{a} and {b} were evaluated on different {bk} boards")
            ma, mb = metrics(ev[a][bk]), metrics(ev[b][bk])
            ga, gb = ev[a][bk]["g"], ev[b][bk]["g"]
            only_a, only_b = int(((ga == 1) & (gb == 0)).sum()), int(((ga == 0) & (gb == 1)).sum())
            row = {"a": MODELS[a][1], "b": MODELS[b][1], "bucket": bk, "greedy_only_a": only_a,
                   "greedy_only_b": only_b, "greedy_mcnemar_p": float(f"{mcnemar_p(only_a, only_b):.3g}")}
            for m, seed_name in (("greedy", "greedy"), ("pass@1", "pass1"), ("pass@16", "pass16")):
                d = ma[m] - mb[m]
                row[m] = {"diff": round(float(d.mean()), 4), "ci": boot_ci(d, f"{a}-{b}/{bk}/{seed_name}")}
            out.append(row)
    return out


def grpo_signal(ev: dict) -> list[dict]:
    """Per GRPO run, the share of groups of G answers whose rewards are not all equal (the groups with a nonzero
    advantage), predicted before training and measured during it.

    Prediction, from the starting model's eval run: for a val in_dist board with c of 16 sampled answers solving it,
    G answers drawn from the 16 are all solving with probability C(c, G) / C(16, G) and all failing with probability
    C(16 - c, G) / C(16, G). Z_L is the sum of the two averaged over the val boards of optimal length L, w_L the share
    of length L among the training boards (stored in the run's metrics), and the prediction is 1 - sum_L w_L Z_L; it
    must equal the one the run logged. Measured: the share over the 16 groups per step of the first and of the last
    WINDOW steps."""
    out = []
    for label, (run, start) in GRPO_RUNS.items():
        m = json.loads((RUNS / run / "metrics.json").read_text())
        w = {r["opt_len"]: r["share"] for r in m["prediction"]["by_length"]}
        e = ev[start]["in_dist"]
        z = np.array([comb(int(c), G) + comb(N_SAMPLES - int(c), G) for c in e["c"]]) / comb(N_SAMPLES, G)
        predicted = 1 - sum(w[L] * z[e["L"] == L].mean() for L in w)
        if abs(predicted - (1 - m["prediction"]["zero_std"])) > 5e-4:
            raise ValueError(f"{run}: prediction {predicted:.4f} differs from the logged {1 - m['prediction']['zero_std']}")
        rows = m["steps"]
        first, last = rows[:WINDOW], rows[-WINDOW:]
        out.append({"run": label, "starts_from": MODELS[start][1], "predicted": round(float(predicted), 4),
                    "measured_first": {"steps": f"{first[0]['step']}-{first[-1]['step']}",
                                       "share": round(1 - float(np.mean([r["zero_std"] for r in first])), 4)},
                    "measured_last": {"steps": f"{last[0]['step']}-{last[-1]['step']}",
                                      "share": round(1 - float(np.mean([r["zero_std"] for r in last])), 4)}})
    return out


def sft_curves() -> list[dict]:
    """Val in_dist greedy solve rate at each in-training evaluation of the one-epoch SFT runs from the SFT model
    (runs 2-5; step 0 is the SFT model)."""
    out = []
    for label in ("run 2", "run 3", "run 4", "run 5"):
        run = EPOCH_RUNS[label]
        m = json.loads((RUNS / run / "metrics.json").read_text())
        cfg = json.loads((RUNS / run / "manifest.json").read_text())["config"]
        curve = [[e["step"], next(t["solve_rate"] for t in e["table"]
                                  if t["mode"] == "greedy" and t["bucket"] == "in_dist")] for e in m["evals"]]
        out.append({"run": label, "lr_schedule": m["sft_args"]["lr_scheduler_type"], "part": cfg["part"],
                    "in_dist_greedy": curve})
    return out


def timing() -> list[dict]:
    """Per one-epoch training run: the median step time (from step 3) and the training time of the epoch, both
    without the in-training evaluations."""
    out = []
    for label, run in EPOCH_RUNS.items():
        t = json.loads((RUNS / run / "metrics.json").read_text())["timing"]
        exp = json.loads((RUNS / run / "manifest.json").read_text())["config"]["experiment"]
        out.append({"run": label, "objective": "GRPO" if exp == "p4_grpo" else "SFT",
                    "median_step_s": t["steady_step_s"], "train_s": t["train_s"]})
    return out


def figure(rows: list[dict], path: Path) -> None:
    """Run 6 - run 4 split into run 7 - run 4 (one more SFT epoch on the new boards) and run 6 - run 7 (GRPO instead of
    SFT for that epoch), per metric, with the bootstrap intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {("rl_anneal", "sft_anneal"): "run 6 - run 4 (GRPO epoch after annealed SFT)",
              ("sft_new_from_anneal", "sft_anneal"): "run 7 - run 4 (SFT epoch on the same new boards)",
              ("rl_anneal", "sft_new_from_anneal"): "run 6 - run 7 (GRPO instead of SFT with the decay)"}
    colors = ("#0b0b0b", "#eb6834", "#2a78d6")
    panels = [("in_dist", "greedy"), ("in_dist", "pass@1"), ("long", "greedy"), ("long", "pass@1"), ("long", "pass@16")]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for j, (a, b) in enumerate(DECOMPOSITION):
        r = {x["bucket"]: x for x in rows if (x["a"], x["b"]) == (MODELS[a][1], MODELS[b][1])}
        ys = [len(panels) - 1 - k + (1 - j) * 0.22 for k in range(len(panels))]
        xs = [r[bk][m]["diff"] for bk, m in panels]
        lo = [r[bk][m]["diff"] - r[bk][m]["ci"][0] for bk, m in panels]
        hi = [r[bk][m]["ci"][1] - r[bk][m]["diff"] for bk, m in panels]
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", color=colors[j], ms=5, capsize=3, label=labels[(a, b)])
    ax.axvline(0, color="#898781", lw=1)
    ax.set_yticks(range(len(panels)), [f"{bk} {m}" for bk, m in reversed(panels)])
    ax.set_xlabel("paired difference over 300 val boards (95% bootstrap interval)")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ev = {key: load_eval(run) for key, (run, _) in MODELS.items()}
    out = {"table": table(ev), "paired": paired(ev), "grpo_signal": grpo_signal(ev),
           "sft_curves": sft_curves(), "timing": timing()}
    (HERE / "results.json").write_text(json.dumps(out, indent=1) + "\n")
    figure(out["paired"], HERE / "decomposition.png")
    print("model | " + " | ".join(f"{bk} {m}" for bk in BUCKETS for m in ("greedy", "pass@1", "pass@16")))
    for r in out["table"]:
        print(r["model"] + " | " + " | ".join(f"{r[bk][m]:.3f}" for bk in BUCKETS for m in ("greedy", "pass@1", "pass@16")))
    print()
    for r in out["paired"]:
        print(f"{r['a']} - {r['b']}, {r['bucket']}: " + ", ".join(
            f"{m} {r[m]['diff']:+.3f} [{r[m]['ci'][0]:.3f}, {r[m]['ci'][1]:.3f}]" for m in ("greedy", "pass@1", "pass@16"))
            + f"; greedy boards only a {r['greedy_only_a']}, only b {r['greedy_only_b']}")
    print()
    for r in out["grpo_signal"]:
        print(f"{r['run']} (from {r['starts_from']}): groups with a nonzero advantage predicted {r['predicted']:.3f}, "
              f"steps {r['measured_first']['steps']} {r['measured_first']['share']:.3f}, "
              f"steps {r['measured_last']['steps']} {r['measured_last']['share']:.3f}")
    print()
    for r in out["sft_curves"]:
        later = [v for step, v in r["in_dist_greedy"] if step >= 250]
        print(f"{r['run']} ({r['lr_schedule']}, {r['part']} part): in_dist greedy during training "
              + " ".join(f"{step}:{v:.3f}" for step, v in r["in_dist_greedy"])
              + f"; from step 250 {min(later):.3f}-{max(later):.3f}")
    print()
    for r in out["timing"]:
        print(f"{r['run']} ({r['objective']}): median step {r['median_step_s']:.3f} s, training {r['train_s']:.1f} s")


if __name__ == "__main__":
    main()
