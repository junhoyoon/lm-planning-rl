"""The runs behind the SFT and GRPO comparison: the fixed split, new boards beyond the split, SFT, evaluation, GRPO and
the validation loss of saved checkpoints. Each function writes one run directory (`sokolm.record`) and is one CLI
command (`sokolm.cli`)."""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from sokolm.assets import MODEL_REPO, MODEL_REVISION, load_tokenizer
from sokolm.formats import Format
from sokolm.record import RunWriter
from sokolm.splits import BUCKETS, SPLITS, build, check_writable, load, summarize, verify, write

# ---------------------------------------------------------------------------------------------------------- the split


def _plot_split(path, records: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, bucket in enumerate(BUCKETS):
        for split, rs in records.items():
            L = np.array([r["opt_len"] for r in rs if r["bucket"] == bucket])
            if len(L):
                share = np.bincount(L) / len(L)
                ax[i].step(np.arange(len(share)), share, where="mid", label=f"{split} ({len(L)})")
        ax[i].set(title=f"{bucket}: optimal length", xlabel="optimal solution length", ylabel="share of boards")
        ax[i].grid(alpha=0.3)
        ax[i].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build_split(name: str = "7x7-v1", force: bool = False, progress=None) -> dict:
    """Build the fixed split `name` (`sokolm.splits`): layout-class hash split into train / val / test, one board per
    layout class, buckets in_dist / long / two_box; writes the data files, the manifest splits/<name>.json and the
    split's statistics."""
    cfg = SPLITS[name]
    check_writable(name, force)  # fail before the build, not after it
    w = RunWriter(f"split-{name}", {"experiment": "build_split", "split": name, **cfg}, claim_level="own-experiment")
    records, streams = build(cfg, name, progress)
    checks = verify(records, cfg)
    summary = summarize(records)
    manifest = write(name, cfg, records, {"run_id": w.run_id, "code": w.manifest["code"], "streams": streams,
                                          "checks": checks, "summary": summary}, force=force)
    plots = w.dir / "plots"
    plots.mkdir()
    _plot_split(plots / "opt_len.png", records)
    table = [{"split": s, "bucket": b, "boards": v["boards"], **{k: v["opt_len"][k] for k in ("min", "median", "p90", "max")},
              "max_budget": v["max_budget"]}
             for s in ("train", "val", "test") for b, v in summary[s].items() if b in BUCKETS]
    metrics = {"table": table, "summary": summary, "streams": streams, "checks": checks, "files": manifest["files"],
               "manifest": f"splits/{name}.json", "plots": ["plots/opt_len.png"], "run_id": w.run_id}
    w.close(metrics)
    return metrics


# ------------------------------------------------------------------------------------------------- shared helpers

EVAL_SELECTION = "the first n_per_bucket val boards of each bucket, in file order (all 300 by default)"
EVAL_BATCH = 512  # sequences per generate call; a 128-token completion peaked at 5,570 MiB at this size
EVAL_MEM_CAP_GIB = 10

SFT_TRAINER_ARGS = ("learning_rate", "lr_scheduler_type", "warmup_steps", "weight_decay", "max_grad_norm", "optim",
                    "loss_type", "completion_only_loss", "max_length", "packing", "padding_free",
                    "per_device_train_batch_size", "gradient_accumulation_steps", "bf16", "gradient_checkpointing")
GRPO_TRAINER_ARGS = ("learning_rate", "lr_scheduler_type", "warmup_steps", "weight_decay", "max_grad_norm", "optim",
                     "loss_type", "beta", "scale_rewards", "epsilon", "epsilon_high", "delta", "num_iterations",
                     "importance_sampling_level", "mask_truncated_completions", "top_entropy_quantile",
                     "num_generations", "max_completion_length", "temperature", "top_p", "top_k", "min_p",
                     "repetition_penalty", "per_device_train_batch_size", "gradient_accumulation_steps",
                     "steps_per_generation", "generation_batch_size", "bf16", "gradient_checkpointing")
STEADY_FROM = 3  # step 1 sets up and creates the AdamW states, step 2 still grows the reserved memory

# plot colors
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BAND_COLORS = ("#6da7ec", "#2a78d6", "#104281")  # light -> dark


def _trainer_args(trainer, keys) -> dict:
    return {k: getattr(getattr(trainer.args, k, None), "value", getattr(trainer.args, k, None)) for k in keys}


def _median(xs) -> Optional[float]:
    return round(float(np.median(xs)), 3) if len(xs) else None


def _mean_of(xs) -> float:
    return round(float(np.mean(list(xs))), 4)


def _checkpoint_dir(spec: str):
    """'<run id>/<best|final>' of an sft or grpo run -> that run's saved checkpoint directory, and the step it holds."""
    from sokolm.paths import RUNS

    run_id, _, which = spec.partition("/")
    if which not in ("best", "final"):
        raise ValueError(f"checkpoint must be '<run id>/best' or '<run id>/final', got {spec!r}")
    d = RUNS / run_id / "checkpoints"
    if not (d / which / "config.json").exists():
        raise FileNotFoundError(f"no saved checkpoint {d / which}")
    return d / which, json.loads((d / "checkpoints.json").read_text())[which]["step"]


def _cap_gpu_memory(mem_cap_gib: float) -> None:
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, mem_cap_gib * 2**30 / total))


def _val_boards(split: str, n_per_bucket: int = 300) -> list[dict]:
    """The first n_per_bucket val boards of each bucket, in file order."""
    val = load(split, "val")
    return [r for b in BUCKETS for r in [x for x in val if x["bucket"] == b][:n_per_bucket]]


def _board_numbers(boards: list[dict]) -> dict:
    """board id -> its position among the boards of its bucket."""
    return {b["board_id"]: j for bucket in dict.fromkeys(x["bucket"] for x in boards)
            for j, b in enumerate(x for x in boards if x["bucket"] == bucket)}


def _write_answers(w: RunWriter, answers: list[dict], boards: list[dict], step: Optional[int] = None,
                   mode_in_id: bool = True) -> None:
    """One episode record per answer; the generated ids are dropped (the completion text stays). Episode ids:
    '<mode>-<bucket>-<j>[-s<sample>]' for an eval run; at a training step 's<step>-<bucket>-<j>' (sft, greedy only)
    or 's<step>-<greedy|sample<k>>-<bucket>-<j>' (grpo, `mode_in_id`); j = the board's position in its bucket, with
    at least three digits (four for the 2,000 new boards)."""
    by_id = {b["board_id"]: b for b in boards}
    number = _board_numbers(boards)
    width = max(3, len(str(max(number.values(), default=0))))
    for r in answers:
        b = by_id[r["board_id"]]
        j = number[b["board_id"]]
        r.pop("ids")
        if step is None:
            eid = f"{r['mode']}-{b['bucket']}-{j:0{width}d}" + (
                f"-s{r['sample']:02d}" if r["mode"] == "sample" else "")
            w.add({"episode_id": eid, "case": r["mode"], "title": f"{r['mode']}, {b['bucket']}, optimal {b['opt_len']}",
                   **r})
            continue
        title = f"step {step}, {r['mode']}, {b['bucket']}, optimal {b['opt_len']}"
        if mode_in_id:
            tag = "greedy" if r["mode"] == "greedy" else f"sample{r['sample']:02d}"
            w.add({"episode_id": f"s{step:04d}-{tag}-{b['bucket']}-{j:0{width}d}", "step": step,
                   "case": f"step {step}, {r['mode']}", "title": title, **r})
        else:
            w.add({"episode_id": f"s{step:04d}-{b['bucket']}-{j:0{width}d}", "step": step, "case": f"step {step}",
                   "title": title, **r})


# ------------------------------------------------------------------------------------------------------- evaluation


def run_eval(split: str = "7x7-v1", name: str = "eval", n_per_bucket: int = 300, n_samples: int = 16,
             temperature: float = 1.0, seed: int = 0, batch_size: int = EVAL_BATCH,
             mem_cap_gib: float = EVAL_MEM_CAP_GIB, checkpoint: Optional[str] = None,
             boards_file: Optional[str] = None) -> dict:
    """Evaluate the base model, or with `checkpoint` ('<run id>/<best|final>' of an sft or grpo run) a saved
    checkpoint, on the val boards (EVAL_SELECTION), or with `boards_file` on the boards of that file in file order
    (bucket NEW_BUCKET; a file of `run_new_boards`).

    Per board one greedy answer and n_samples sampled answers at `temperature`, with the weights in bf16.
    Metrics (`sokolm.evaluate`): greedy solve rate and pass@k per bucket and per optimal length, the share of boards
    with mixed sampled outcomes, parse failures, truncations, no-ops, answer length relative to optimal, the share of
    answers equal to the prompt's example, distinct texts; on the val boards also the swap and fixed board-blind
    baselines (the swap baseline grows with the square of the boards per bucket)."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    from sokolm.evaluate import KS, answer_records, fixed_baseline, summary_rows, swap_baseline
    from sokolm.lm import END_TOKENS, generation_config, load_model

    t_start = time.perf_counter()
    ckpt_dir, ckpt_step = _checkpoint_dir(checkpoint) if checkpoint else (None, None)
    fmt = Format()
    tok = load_tokenizer()
    boards = file_boards(boards_file) if boards_file else _val_boards(split, n_per_bucket)
    greedy_cfg = generation_config(tok, "greedy")
    sample_cfg = generation_config(tok, "sample", temperature=temperature)
    ks = tuple(k for k in KS if k <= n_samples)
    keys = ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens", "eos_token_id",
            "pad_token_id")
    _cap_gpu_memory(mem_cap_gib)
    config = {"experiment": "eval", "split": split, "model": MODEL_REPO, "revision": MODEL_REVISION,
              "checkpoint": checkpoint, "checkpoint_step": ckpt_step, "dtype": "bfloat16", "torch": torch.__version__,
              "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0), "format": fmt.name,
              "boards": EVAL_SELECTION if boards_file is None else f"the boards of {Path(boards_file).name} in file "
                                                                   f"order ({len(boards)})",
              **({} if boards_file is None else {"boards_file": Path(boards_file).name,
                                                 "boards_sha256": _sha256(Path(boards_file))}),
              "n_per_bucket": n_per_bucket if boards_file is None else None, "n_samples": n_samples, "ks": list(ks),
              "seed": seed, "batch_size": batch_size, "mem_cap_gib": mem_cap_gib,
              "greedy": {k: getattr(greedy_cfg, k) for k in keys},
              "sample": {k: getattr(sample_cfg, k) for k in keys}}
    w = RunWriter(name, config, claim_level="own-experiment")

    torch.cuda.reset_peak_memory_stats()
    model = load_model() if ckpt_dir is None else \
        AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=torch.bfloat16).to("cuda").eval()
    timing = {}
    t0 = time.perf_counter()
    greedy = answer_records(model, tok, boards, fmt, greedy_cfg, "greedy", seed=seed, batch_size=batch_size)
    timing["greedy_s"] = round(time.perf_counter() - t0, 1)
    print(f"greedy: {len(greedy)} answers in {timing['greedy_s']} s", flush=True)
    t0 = time.perf_counter()
    sampled = answer_records(model, tok, boards, fmt, sample_cfg, "sample", n=n_samples, seed=seed,
                             batch_size=batch_size)
    timing["sample_s"] = round(time.perf_counter() - t0, 1)
    print(f"sample: {len(sampled)} answers in {timing['sample_s']} s", flush=True)
    gpu = {"peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
           "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)}
    del model
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    _write_answers(w, greedy + sampled, boards)
    recs = greedy + sampled
    errors = Counter(r["parse_error"] for r in recs if not r["parse_ok"])
    metrics = {
        "table": summary_rows(recs, fmt, ks),
        "by_length": summary_rows(recs, fmt, ks, by_length=True),
        **({} if boards_file else {"swap": swap_baseline(boards, greedy, fmt), "fixed": fixed_baseline(boards)}),
        "parse_errors": dict(errors),
        "parse_error_examples": {e: [r["completion"] for r in recs if r["parse_error"] == e][:3] for e in errors},
        "end_tokens": dict(Counter(str(r["end_token"]) for r in recs)),
        "end_token_ids": list(END_TOKENS),
        "timing": timing | {"analysis_s": round(time.perf_counter() - t0, 1),
                            "total_s": round(time.perf_counter() - t_start, 1)},
        "gpu": gpu,
        "run_id": w.run_id,
    }
    w.close(metrics)
    return metrics


# ------------------------------------------------------------------------------------------------------- new boards

NEW_BOARDS_N = 2000
NEW_BUCKET = "new"  # bucket label of boards read from a file
BOARD_FIELDS = ("board_id", "text", "opt_len", "budget")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_boards(n: int = NEW_BOARDS_N, split: str = "7x7-v1", progress=None) -> tuple[list[dict], dict]:
    """`n` one-box boards that no split holds: the split's one-box stream read on past its last consumed candidate,
    keeping a board when its layout class hashes to train but holds no board of the split (nor of an earlier kept
    board) and the board is solvable in fewer than long_from moves (the in_dist range). They come from the training
    distribution, but no model trains on them and no checkpoint is chosen on them. Returns the records and per-reason
    counts of the candidates."""
    from sokolm.generate import stream
    from sokolm.simulate import simulate
    from sokolm.sokoban import PUSH, Board, action_letters
    from sokolm.solver import solve
    from sokolm.splits import SPLIT_NAMES, class_id, layout_class, split_of

    cfg = SPLITS[split]
    taken, start = set(), 0
    for s in SPLIT_NAMES:
        for r in load(split, s):
            taken.add(r["layout_class"])
            if r["n_boxes"] == 1:
                start = max(start, r["stream_index"] + 1)
    st = Counter()
    recs = []
    for index, text, why in stream(cfg["size"], cfg["seeds"]["one_box"], 1, cfg["max_wall_frac"],
                                   cfg["wall_sampling"]):
        if index < start:
            continue
        st["candidates"] += 1
        if text is None:
            st[why] += 1
            continue
        cls = layout_class(text)
        cid = class_id(cls)
        if cid in taken:
            st["class_taken"] += 1
            continue
        where = split_of(cls, cfg)
        if where != "train":
            st[f"class_of_{where}"] += 1
            continue
        board = Board(text)
        sol = solve(board)
        st["bfs"] += 1
        if not sol.solvable:
            st["unsolvable"] += 1
            continue
        if sol.length >= cfg["long_from"]:
            st["long"] += 1
            continue
        taken.add(cid)
        o = simulate(board, sol.actions, sol.length)
        recs.append({"board_id": f"{split}-new-1b-{index:07d}", "text": text, "opt_len": sol.length,
                     "budget": 2 * sol.length, "solution": action_letters(sol.actions),
                     "n_pushes": o.events.count(PUSH), "n_optimal": sol.n_optimal, "layout_class": cid,
                     "stream_index": index})
        st["accepted"] += 1
        if progress and st["accepted"] % 500 == 0:
            progress(f"{st['accepted']} boards, {st['candidates']} candidates")
        if len(recs) == n:
            break
    return recs, {"first_index": start, **dict(sorted(st.items()))}


def check_new_boards(recs: list[dict], split: str = "7x7-v1") -> dict:
    """Recompute what the records of `new_boards` claim; raise on the first violation."""
    from sokolm.simulate import simulate
    from sokolm.sokoban import Board, parse_actions
    from sokolm.solver import solve
    from sokolm.splits import SPLIT_NAMES, class_id, layout_class, split_of

    cfg = SPLITS[split]
    split_classes = {r["layout_class"] for s in SPLIT_NAMES for r in load(split, s)}
    seen = set()
    for r in recs:
        cls = layout_class(r["text"])
        if class_id(cls) != r["layout_class"] or split_of(cls, cfg) != "train":
            raise ValueError(f"{r['board_id']}: layout class does not hash to train")
        if r["layout_class"] in split_classes or r["layout_class"] in seen:
            raise ValueError(f"{r['board_id']}: layout class already used")
        seen.add(r["layout_class"])
        board = Board(r["text"])
        if len(board.targets) != 1 or not 1 <= r["opt_len"] < cfg["long_from"]:
            raise ValueError(f"{r['board_id']}: not a one-box board of the in_dist range")
        o = simulate(board, parse_actions(r["solution"]), r["budget"])
        if not (o.solved_at == r["opt_len"] == solve(board).length and o.n_noop == 0):
            raise ValueError(f"{r['board_id']}: the stored solution is not optimal")
    return {"boards": len(recs), "distinct_layout_classes": len(seen), "class_hashes_to_train": True,
            "no_class_shared_with_the_split": True, "solutions_replay_to_solved_at_opt_len": True}


def run_new_boards(n: int = NEW_BOARDS_N, split: str = "7x7-v1", name: str = "new-boards", progress=None) -> dict:
    """Run: `new_boards`, checked, written to boards.jsonl in the run directory (the file `run_eval` takes as
    `boards_file`), with the share of each optimal length against the split's train boards."""
    t0 = time.perf_counter()
    w = RunWriter(name, {"experiment": "new_boards", "split": split, "n": n}, claim_level="own-experiment")
    recs, stats = new_boards(n, split, progress)
    checks = check_new_boards(recs, split)
    path = w.dir / "boards.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    train = Counter(r["opt_len"] for r in load(split, "train"))
    new = Counter(r["opt_len"] for r in recs)
    lengths = [{"opt_len": L, "new": new[L], "new_share": round(new[L] / max(1, len(recs)), 4),
                "train_share": round(train[L] / sum(train.values()), 4)} for L in sorted(set(train) | set(new))]
    metrics = {"stream": stats, "checks": checks, "lengths": lengths, "file": path.name, "sha256": _sha256(path),
               "seconds": round(time.perf_counter() - t0, 1), "run_id": w.run_id}
    w.close(metrics)
    return metrics


def file_boards(path: str) -> list[dict]:
    """The boards of a JSONL file in file order, labelled with bucket NEW_BUCKET."""
    recs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for r in recs:
        missing = [k for k in BOARD_FIELDS if k not in r]
        if missing:
            raise ValueError(f"{path}: board {r.get('board_id')} lacks {missing}")
        r["bucket"] = NEW_BUCKET
    if len({r["board_id"] for r in recs}) != len(recs):
        raise ValueError(f"{path}: repeated board ids")
    return recs


# -------------------------------------------------------------------------------------------------------------- SFT

SFT_LR = 2e-5
SFT_STEPS = 1563  # 1,563 x 16 = 25,008 boards: one pass over a 25,000-board part
SFT_WARMUP = 50
SFT_BATCH = 16  # boards per optimizer step; 9,132 MiB reserved for one step with gradient checkpointing
SFT_EVAL_EVERY = 250
SFT_TRAIN_EVAL = 300
SFT_GEN_BATCH = 256  # sequences per generate call at an evaluation; the training state stays on the GPU
SFT_LOSS_BATCH = 16
SFT_MEM_CAP_GIB = 16
SFT_SELECTION = ("training: the boards of the 'sft' part (25,000), drawn without replacement in the order the "
                 "Trainer's sampler fixes with the seed; evaluation: " + EVAL_SELECTION + " (900 val boards); "
                 "val loss on the 300 val in_dist boards")


def _sft_boards(split: str, part: str = "sft") -> list[dict]:
    """The boards SFT trains on: those of `part` ('sft', or 'rl', the boards GRPO trains on), in the part's order."""
    from sokolm.splits import load_part

    return list(load_part(split, part))


def _sft_role(j: int, n: int) -> str:
    """Role of target token j of n: prefix (`Answer`, `:`), action, end (<|im_end|>)."""
    return "prefix" if j < 2 else "end" if j == n - 1 else "action"


def _sft_val_loss(model, tok, examples: list[dict], batch_size: int = SFT_LOSS_BATCH, autocast: bool = True) -> dict:
    """The SFT loss over all of `examples` at once: -(1/M) sum log p over their target tokens (answer line +
    <|im_end|>), M = the number of those tokens (`train.sft_target_logprobs`), under bf16 autocast like the training
    steps unless `autocast` is False. Also per token role (`_sft_role`): the mean -log p, the share of the summed
    -log p, and for the end token the mean probability."""
    import torch

    from sokolm.train import sft_target_logprobs

    sums, counts, p_end = Counter(), Counter(), []
    device = next(model.parameters()).device.type
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        for s in range(0, len(examples), batch_size):
            ref = sft_target_logprobs(model, tok, examples[s:s + batch_size])
            for b, t in enumerate(ref["target_ids"]):
                for j in range(len(t)):
                    role = _sft_role(j, len(t))
                    sums[role] += -float(ref["logp"][b, j])
                    counts[role] += 1
                p_end.append(float(np.exp(float(ref["logp"][b, len(t) - 1]))))
    total, M = sum(sums.values()), sum(counts.values())
    return {"loss": round(total / M, 5), "M": M, "examples": len(examples),
            "nll_mean": {k: round(sums[k] / counts[k], 5) for k in ("prefix", "action", "end")},
            "share": {k: round(sums[k] / total, 4) for k in ("prefix", "action", "end")},
            "p_end_mean": round(float(np.mean(p_end)), 4)}


def _copy_weights(src, dst) -> None:
    """dst's parameters := src's, cast to dst's dtype (fp32 training weights -> the bf16 evaluation copy)."""
    import torch

    s = dict(src.named_parameters())
    d = dict(dst.named_parameters())
    if s.keys() != d.keys():
        raise ValueError("the two models have different parameters")
    with torch.no_grad():
        for k, p in d.items():
            p.copy_(s[k])


def _same_greedy(run_id: str, greedy: list[dict]) -> dict:
    """The greedy answers of `greedy` against those of an earlier run on the same boards."""
    from sokolm.paths import RUNS

    with open(RUNS / run_id / "episodes.jsonl") as f:
        old = {r["board_id"]: r for r in map(json.loads, f) if r.get("mode") == "greedy"}
    same = [old[r["board_id"]]["completion"] == r["completion"] for r in greedy]
    return {"run": run_id, "boards": len(greedy), "same_completion": sum(same),
            "solve_rate_then": _mean_of(old[r["board_id"]]["reward"] for r in greedy),
            "solve_rate_now": _mean_of(r["reward"] for r in greedy)}


def _plot_sft(path, steps: list[dict], evals: list[dict], title: str) -> None:
    """Four panels: training loss per step and val loss; val loss per token role; greedy solve rate per bucket with
    the swap baseline of in_dist; gradient norm (before clipping) per step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = [r["step"] for r in steps]
    e = [r["step"] for r in evals]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    a = ax[0, 0]
    a.plot(s, [r["loss"] for r in steps], ".", alpha=0.4, label="training batch (16 boards)")
    a.plot(e, [r["val"]["loss"] for r in evals], "o-", label="val in_dist (300 boards)")
    a.set_yscale("log")
    a.set_title("SFT loss: mean -log p per target token")
    a.set_xlabel("step")
    a.legend(fontsize=8)
    a = ax[0, 1]
    for role in ("action", "end", "prefix"):
        a.plot(e, [r["val"]["nll_mean"][role] for r in evals], "o-", label=role)
    a.set_yscale("log")
    a.set_title("val in_dist: mean -log p per token role")
    a.set_xlabel("step")
    a.legend(fontsize=8)
    a = ax[1, 0]
    for bucket in [x["bucket"] for x in evals[0]["table"]]:
        a.plot(e, [next(x["solve_rate"] for x in r["table"] if x["bucket"] == bucket) for r in evals], "o-",
               label=f"greedy, {bucket}")
    a.plot(e, [next(x["swap"] for x in r["swap"] if x["bucket"] == "in_dist") for r in evals], "s--",
           label="swap baseline, in_dist")
    a.set_title("greedy solve rate (300 boards per bucket; train = SFT boards)")
    a.set_xlabel("step")
    a.legend(fontsize=8)
    a = ax[1, 1]
    a.plot(s, [r["grad_norm"] for r in steps], ".", alpha=0.5)
    a.axhline(1.0, color="0.5", lw=1, ls="--")
    a.set_yscale("log")
    a.set_title("gradient norm before clipping (clipped to 1, dashed)")
    a.set_xlabel("step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


class _BestWeights:
    """The weights of the evaluation with the highest val in_dist greedy solve rate so far (ties: the earlier one),
    kept as a copy on the model's device so that the run writes them to disk once, at its end."""

    def __init__(self, model):
        self.step: Optional[int] = None
        self.rate = -1.0
        self.params = {k: p.detach().clone() for k, p in model.named_parameters()}

    def offer(self, step: int, rate: float, model) -> bool:
        """Keep `model`'s weights if `rate` is higher than every earlier one; True if kept."""
        if rate <= self.rate:
            return False
        _copy_weights(model, _ParamHolder(self.params))
        self.step, self.rate = step, rate
        return True

    def save(self, model, path) -> None:
        """Put the kept weights into `model` (overwriting its own) and save it with save_pretrained."""
        _copy_weights(_ParamHolder(self.params), model)
        model.save_pretrained(path)


class _ParamHolder:
    """A dict of tensors that `_copy_weights` can read from or write to like a model."""

    def __init__(self, params: dict):
        self.params = params

    def named_parameters(self):
        return self.params.items()


def _save_checkpoints(w: RunWriter, eval_model, best: "_BestWeights", evals: list[dict], steps: int) -> dict:
    """checkpoints/final (the bf16 copy of the last evaluation, which must be at the last step) and checkpoints/best,
    with checkpoints/checkpoints.json naming their steps."""
    import os

    if evals[-1]["step"] != steps:
        raise RuntimeError("the last evaluation is not at the last step; eval_model does not hold its weights")
    t0 = time.perf_counter()
    ckpt = w.dir / "checkpoints"
    eval_model.save_pretrained(ckpt / "final")
    best.save(eval_model, ckpt / "best")
    os.sync()
    checkpoints = {"best": {"step": best.step, "in_dist_greedy": best.rate, "path": "checkpoints/best"},
                   "final": {"step": steps, "path": "checkpoints/final"}, "dtype": "bfloat16",
                   "rule": "highest val in_dist greedy solve rate over the evaluations, ties: the earlier",
                   "save_s": round(time.perf_counter() - t0, 1)}
    (ckpt / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2) + "\n")
    return checkpoints


def run_sft(split: str = "7x7-v1", name: str = "sft", lr: float = SFT_LR, steps: int = SFT_STEPS,
            warmup: int = SFT_WARMUP, batch_size: int = SFT_BATCH, eval_every: int = SFT_EVAL_EVERY, seed: int = 0,
            gen_batch: int = SFT_GEN_BATCH, mem_cap_gib: float = SFT_MEM_CAP_GIB, train_eval: int = SFT_TRAIN_EVAL,
            save: bool = False, init: Optional[str] = None, part: str = "sft",
            step0_reference: Optional[str] = None, lr_schedule: str = "constant") -> dict:
    """One SFT run on the BFS answers of the boards of `part` ('sft', or 'rl': the boards GRPO trains on), evaluated
    during training. Starts from the base model, or with `init` = '<run id>/<best|final>' from a saved checkpoint
    (both copies loaded straight to the GPU), with a fresh AdamW and the same warmup either way.

    Training (TRL SFTTrainer, `train.sft_run_trainer`): fp32 weights, gradients and AdamW states, bf16 autocast,
    gradient checkpointing; `steps` optimizer steps of `batch_size` boards (SFT_SELECTION); the lr rises linearly over
    `warmup` steps, then stays constant (`lr_schedule` "constant") or falls linearly to 0 at the last step ("linear").
    Evaluation at step 0 and every `eval_every` steps (after the step's update):
      val loss  `_sft_val_loss` on the BFS answers of the 300 val in_dist boards, with the training weights
      greedy    one greedy answer per val board (900), from a bf16 copy of the training weights; solve rate per
                bucket and per optimal length, the swap baseline
      train     with train_eval > 0: the same greedy evaluation on the first train_eval training boards (bucket
                "train"), generated after the val boards so that the val batches stay the same
    With `step0_reference` (an eval run of the starting model) the step-0 greedy answers are compared with that run's.
    With `save`: the bf16 weights of the evaluation with the highest val in_dist greedy solve rate (ties: the earlier)
    are kept on the GPU during the run; at the end they and the last weights are written to checkpoints/best and
    checkpoints/final (save_pretrained)."""
    import torch
    import transformers
    import trl
    from transformers import TrainerCallback

    from sokolm.evaluate import answer_records, summary_rows, swap_baseline
    from sokolm.lm import generation_config, load_model
    from sokolm.train import sft_examples, sft_run_trainer, step_timer

    t_start = time.perf_counter()
    fmt = Format()
    tok = load_tokenizer()
    gen_tok = load_tokenizer()  # generate() sets left padding on the tokenizer it gets; TRL's collator keeps its own
    sft_boards = _sft_boards(split, part)
    train_ex = sft_examples(tok, sft_boards, fmt)
    boards = _val_boards(split)
    loss_ex = sft_examples(tok, [b for b in boards if b["bucket"] == "in_dist"], fmt)
    train_boards = [{**r, "bucket": "train"} for r in sft_boards[:train_eval]]
    greedy_cfg = generation_config(tok, "greedy")
    init_dir, init_step = _checkpoint_dir(init) if init else (None, None)
    _cap_gpu_memory(mem_cap_gib)
    config = {"experiment": "sft", "split": split, "model": MODEL_REPO,
              "revision": MODEL_REVISION, "precision": "fp32 weights, gradients and AdamW states; bf16 autocast; "
              "greedy evaluation with a bf16 copy of the weights", "torch": torch.__version__,
              "transformers": transformers.__version__, "trl": trl.__version__, "gpu": torch.cuda.get_device_name(0),
              "format": fmt.name, "boards": SFT_SELECTION.replace("'sft' part", f"'{part}' part"),
              "part": part, "init": init, "init_step": init_step, "train_boards": len(train_ex),
              "epochs": round(steps * batch_size / len(train_ex), 3), "lr": lr, "lr_schedule": lr_schedule,
              "steps": steps, "warmup": warmup, "batch_size": batch_size, "eval_every": eval_every, "seed": seed,
              "gen_batch": gen_batch, "loss_batch": SFT_LOSS_BATCH, "mem_cap_gib": mem_cap_gib,
              "step0_reference": step0_reference, "train_eval": train_eval, "save": save}
    w = RunWriter(name, config, claim_level="own-experiment")

    torch.cuda.reset_peak_memory_stats()
    if init is None:
        model = load_model(dtype=torch.float32)
        eval_model = load_model()
    else:  # bf16 weights read as fp32 are exact; device_map puts them on the GPU without a host copy
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(init_dir, dtype=torch.float32, device_map="cuda")
        eval_model = AutoModelForCausalLM.from_pretrained(init_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    best = _BestWeights(eval_model) if save else None
    all_boards = boards + train_boards
    evals: list[dict] = []
    first_greedy: list[dict] = []

    def evaluate_now(step: int) -> None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        was_training = model.training
        model.eval()
        val_loss = _sft_val_loss(model, tok, loss_ex)
        _copy_weights(model, eval_model)
        greedy = answer_records(eval_model, gen_tok, boards, fmt, greedy_cfg, "greedy", seed=None,
                                batch_size=gen_batch)  # greedy needs no seed; the global RNG is left alone
        if step == 0:
            first_greedy.extend(greedy)
        if train_boards:
            greedy += answer_records(eval_model, gen_tok, train_boards, fmt, greedy_cfg, "greedy", seed=None,
                                     batch_size=gen_batch)
        model.train(was_training)
        table = summary_rows(greedy, fmt)
        rate = next(x for x in table if x["bucket"] == "in_dist")["solve_rate"]
        kept = best is not None and best.offer(step, rate, eval_model)
        torch.cuda.synchronize()
        row = {"step": step, "val": val_loss, "table": table,
               "by_length": summary_rows(greedy, fmt, by_length=True),
               "swap": swap_baseline(all_boards, greedy, fmt), "kept_as_best": kept,
               "seconds": round(time.perf_counter() - t0, 1),
               "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
               "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)}
        evals.append(row)
        _write_answers(w, greedy, all_boards, step=step, mode_in_id=False)
        ind = next(x for x in row["table"] if x["bucket"] == "in_dist")
        print(f"step {step}: val loss {val_loss['loss']:.4f} (action {val_loss['nll_mean']['action']:.3f}, end "
              f"{val_loss['nll_mean']['end']:.3f}), greedy " + ", ".join(
                  f"{x['bucket']} {x['solve_rate']:.3f}" for x in row["table"]) + f", in_dist parse fail "
              f"{ind['parse_fail']:.3f}; {row['seconds']} s, peak reserved {row['peak_reserved_mib']} MiB", flush=True)

    class Evaluate(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % eval_every == 0 or state.global_step == args.max_steps:
                evaluate_now(state.global_step)

    evaluate_now(0)
    step0 = _same_greedy(step0_reference, first_greedy) if step0_reference else None
    if step0:
        print(f"step 0 against {step0_reference}: {step0}", flush=True)
    timer = step_timer()
    tr = sft_run_trainer(model, tok, train_ex, w.dir / "trainer", lr=lr, steps=steps, warmup=warmup,
                         batch_size=batch_size, seed=seed, callbacks=[timer, Evaluate()], schedule=lr_schedule)
    sft_args = _trainer_args(tr, SFT_TRAINER_ARGS)
    t0 = time.perf_counter()
    tr.train()
    train_s = round(time.perf_counter() - t0 - sum(r["seconds"] for r in evals[1:]), 1)
    checkpoints = _save_checkpoints(w, eval_model, best, evals, steps) if save else None

    logs = {h["step"]: h for h in tr.state.log_history if "loss" in h}
    keys = ("loss", "grad_norm", "learning_rate", "mean_token_accuracy", "entropy", "num_tokens")
    rows = [{"step": t["step"], **{k: logs[t["step"]].get(k) for k in keys}, "seconds": t["seconds"],
             "peak_allocated_mib": t["peak_allocated_mib"], "peak_reserved_mib": t["peak_reserved_mib"]}
            for t in timer.steps]
    plots = w.dir / "plots"
    plots.mkdir(exist_ok=True)
    _plot_sft(plots / "sft.png", rows, evals, f"{w.run_id}: SFT, lr {lr:g}, warmup {warmup}, {batch_size} boards/step"
              + (f", from {init}" if init else "") + (f", '{part}' boards" if part != "sft" else "")
              + (f", lr schedule {lr_schedule}" if lr_schedule != "constant" else ""))
    last = evals[-1]
    eval_s = round(sum(r["seconds"] for r in evals), 1)
    total_s = round(time.perf_counter() - t_start, 1)
    metrics = {
        "selection": {"step": last["step"], "in_dist_greedy": next(x["solve_rate"] for x in last["table"]
                                                                   if x["bucket"] == "in_dist"),
                      "val_loss": last["val"]["loss"]},
        "evals": evals, "steps": rows, "step0_vs_reference": step0, "sft_args": sft_args, "checkpoints": checkpoints,
        "timing": {"train_s": train_s, "eval_s": eval_s, "other_s": round(total_s - train_s - eval_s, 1),
                   "steady_step_s": _median([r["seconds"] for r in rows if r["step"] >= STEADY_FROM]),
                   "total_s": total_s},
        "gpu": {"train_peak_allocated_mib": max(r["peak_allocated_mib"] for r in rows),
                "train_peak_reserved_mib": max(r["peak_reserved_mib"] for r in rows),
                "eval_peak_allocated_mib": max(r["peak_allocated_mib"] for r in evals),
                "eval_peak_reserved_mib": max(r["peak_reserved_mib"] for r in evals)},
        "plots": ["plots/sft.png"], "run_id": w.run_id,
    }
    w.close(metrics)
    return metrics


# ------------------------------------------------------------------------------------------------------------- GRPO

GRPO_PROMPTS = 16  # boards per optimizer step
GRPO_G = 8  # completions per board: TRL's default num_generations
GRPO_MICRO = 8  # completions per loss micro-batch; 8 x 8 reserved 10,344 MiB with gradient checkpointing
GRPO_LR = 1e-6  # TRL's GRPO default
GRPO_STEPS = 1562  # 1,562 x 16 = 24,992 of the 25,000 'rl' boards
GRPO_EVAL_EVERY = 100
GRPO_EVAL_SAMPLES = 16  # sampled answers per val in_dist board at an evaluation
GRPO_EVAL_SEED = 0
GRPO_GEN_BATCH = 256  # sequences per generate call at an evaluation
GRPO_MEM_CAP_GIB = 18
LENGTH_BANDS = ((1, 4), (5, 8), (9, 13))
ROLLOUT_KEYS = ("board_id", "opt_len", "completion", "reward", "parse_ok", "truncated", "solved_at", "n_actions",
                "n_noop", "len_ratio")  # and n_tokens, the completion's token count (end token included)
TRL_LOG_KEYS = ("loss", "grad_norm", "learning_rate", "reward", "reward_std", "frac_reward_zero_std", "entropy",
                "completions/mean_length", "completions/max_length", "completions/clipped_ratio",
                "clip_ratio/region_mean", "num_tokens", "step_time")
GRPO_SELECTION = ("training: the 25,000 boards of the 'rl' part (optimal length 1-13), drawn without replacement in "
                  "the order of TRL's RepeatSampler (seed); evaluation: the first 300 val boards of each bucket in file "
                  "order (one greedy answer each), and the in_dist ones also with eval_samples sampled answers each")


def _prediction(run_id: str, shares: dict[int, float], g: int, checkpoint: str) -> dict:
    """The zero-std prediction (`evaluate.zero_std_prediction`) from the sampled val in_dist answers of an eval run,
    which must have evaluated `checkpoint`, the model the GRPO run starts from."""
    from sokolm.evaluate import zero_std_prediction
    from sokolm.paths import RUNS

    evaluated = json.loads((RUNS / run_id / "manifest.json").read_text())["config"].get("checkpoint")
    if evaluated != checkpoint:
        raise ValueError(f"{run_id} evaluated {evaluated}, not the starting checkpoint {checkpoint}")
    with open(RUNS / run_id / "episodes.jsonl") as f:
        recs = [r for r in map(json.loads, f) if r["mode"] == "sample" and r["bucket"] == "in_dist"]
    return {"run": run_id, **zero_std_prediction(recs, shares, g)}


def _band_value(pred: dict, lo: int, hi: int, key: str = "zero_std") -> float:
    """A per-length prediction over optimal lengths lo..hi, weighted by the lengths' shares of the training boards."""
    rows = [r for r in pred["by_length"] if lo <= r["opt_len"] <= hi]
    return round(sum(r["share"] * r[key] for r in rows) / sum(r["share"] for r in rows), 4)


def _rolling(xs: list, k: int) -> list:
    """Trailing mean over the last k values (fewer at the start), skipping None."""
    out = []
    for i in range(len(xs)):
        win = [x for x in xs[max(0, i - k + 1):i + 1] if x is not None]
        out.append(float(np.mean(win)) if win else None)
    return out


def _plot_grpo(path, rows: list[dict], evals: list[dict], prediction: dict, title: str, k: int = 20) -> None:
    """2 x 3: the training reward per step; the share of groups with all-equal rewards (and all solved) against the
    prediction from the starting checkpoint's samples; that share per band of optimal lengths; entropy and completion
    length; val greedy solve rate per bucket with the swap baseline; the val in_dist samples (pass@1, pass@n, distinct
    answers per board)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sokolm.evaluate import KS

    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9})
    s = [r["step"] for r in rows]
    e = [r["step"] for r in evals]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))

    def finish(a, t, handles=None):
        a.set_title(t, loc="left", color=INK, fontsize=10)
        a.set_xlabel("optimizer step")
        a.legend(handles=handles, fontsize=8, frameon=False)

    a = axes[0, 0]
    a.plot(s, [r["reward"] for r in rows], ".", color=MUTED, ms=3, label="one step")
    a.plot(s, _rolling([r["reward"] for r in rows], k), color=BLUE, lw=2, label=f"mean of the last {k} steps")
    finish(a, "training: mean reward of the step's completions (RL boards)")
    a = axes[0, 1]
    for key, color, marker, label in (("zero_std", BLUE, "o", "all equal"), ("all_solved", ORANGE, "s", "all solved")):
        a.plot(s, _rolling([r[key] for r in rows], k), color=color, lw=2, label=f"{label}, last {k} steps")
        a.axhline(prediction[key], color=color, lw=1, ls="--")
        pe = [ev for ev in evals if ev["prediction"]]  # none when an evaluation samples fewer than g per board
        a.plot([ev["step"] for ev in pe], [ev["prediction"][key] for ev in pe], marker, color=color, ms=5,
               label=f"{label}, predicted from the evaluation's samples")
    finish(a, "groups with all-equal rewards (dashed: predicted from the start's samples)")
    a = axes[0, 2]
    for (lo, hi), color in zip(LENGTH_BANDS, BAND_COLORS):
        a.plot(s, _rolling([r[f"zero_std_{lo}_{hi}"] for r in rows], k), color=color, lw=2,
               label=f"optimal length {lo}-{hi}")
        a.axhline(_band_value(prediction, lo, hi), color=color, lw=1, ls="--")
    finish(a, f"all-equal groups per length band, last {k} steps (dashed: predicted)")
    a = axes[1, 0]
    h1 = a.plot(s, _rolling([r["entropy"] for r in rows], k), color=BLUE, lw=2,
                label="entropy per completion token (nats)")
    b = a.twinx()
    h2 = b.plot(s, _rolling([r["completions/mean_length"] for r in rows], k), color=ORANGE, lw=2,
                label="completion length (tokens, right)")
    b.grid(False)
    finish(a, f"entropy and completion length, last {k} steps", handles=h1 + h2)
    a = axes[1, 1]
    for bucket, color, marker in (("in_dist", BLUE, "o"), ("long", ORANGE, "s"), ("two_box", MUTED, "^")):
        a.plot(e, [next(x["solve_rate"] for x in ev["table"] if x["mode"] == "greedy" and x["bucket"] == bucket)
                   for ev in evals], marker=marker, color=color, lw=1.5, ms=4, label=f"greedy, {bucket}")
    a.plot(e, [next(x["swap"] for x in ev["swap"] if x["bucket"] == "in_dist") for ev in evals], ls="--",
           color=INK, lw=1, label="swap baseline, in_dist")
    finish(a, "val greedy solve rate (300 boards per bucket)")
    a = axes[1, 2]
    samp = [next(x for x in ev["table"] if x["mode"] == "sample") for ev in evals]
    n = evals[0]["samples_per_board"]
    kmax = max(kk for kk in KS if kk <= n)
    h1 = a.plot(e, [x["pass@1"] for x in samp], "o-", color=BLUE, ms=4, label="pass@1")
    h1 += a.plot(e, [x[f"pass@{kmax}"] for x in samp], "s-", color=ORANGE, ms=4, label=f"pass@{kmax}")
    b = a.twinx()
    h2 = b.plot(e, [ev["distinct_per_board"] for ev in evals], "^--", color=MUTED, ms=4,
                label=f"distinct answers per board, of {n} (right)")
    b.grid(False)
    finish(a, f"val in_dist samples ({n} per board, T = 1)", handles=h1 + h2)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_grpo(checkpoint: str, prediction_run: str, split: str = "7x7-v1", name: str = "grpo",
             steps: int = GRPO_STEPS, prompts_per_step: int = GRPO_PROMPTS, g: int = GRPO_G,
             micro_batch: int = GRPO_MICRO, lr: float = GRPO_LR, lr_schedule: str = "constant",
             eval_every: int = GRPO_EVAL_EVERY, eval_samples: int = GRPO_EVAL_SAMPLES, seed: int = 0,
             gen_batch: int = GRPO_GEN_BATCH, mem_cap_gib: float = GRPO_MEM_CAP_GIB, save: bool = True) -> dict:
    """GRPO with TRL's GRPOTrainer from a saved checkpoint ('<run id>/<best|final>' of an sft run).

    Training (`train.grpo_run_trainer`): the checkpoint's bf16 weights loaded in fp32 (exact), fp32 gradients and AdamW
    states, bf16 autocast, gradient checkpointing; every optimizer step takes `prompts_per_step` RL boards and `g`
    completions per board (T = 1, the full distribution, at most COMPLETION_CAP tokens), judged by the binary reward
    (`train.reward_function`); lr `lr` (`lr_schedule`), TRL's other defaults.
    Per step: TRL's logs (TRL_LOG_KEYS), wall time and peak GPU memory, and from the step's rollouts the groups
    (`evaluate.group_rows`): the share with all-equal rewards (zero std) overall and per band of optimal lengths, all
    solved, all failed, distinct texts and first actions per group. TRL's own reward mean and zero-std share are
    checked against ours. Every completion goes to rollouts.jsonl (ROLLOUT_KEYS, with the step).
    Prediction: the zero-std share that the sampled answers of `prediction_run` (an eval run of the starting
    checkpoint) predict for groups of `g` (`evaluate.zero_std_prediction`, weighted by the RL boards' length shares).
    Evaluation at step 0 and every `eval_every` steps, with a bf16 copy of the training weights: one greedy answer per
    val board (900; solve rate per bucket and length, swap baseline) and `eval_samples` sampled answers per val in_dist
    board (pass@k, mixed, distinct answers per board, the zero-std prediction at this step), under a fixed seed with
    the training RNG state restored afterwards.
    With `save`: checkpoints/best (highest val in_dist greedy solve rate, ties: the earlier) and checkpoints/final as
    in `run_sft`, so that `eval --checkpoint <run id>/final` evaluates them."""
    import torch
    import transformers
    import trl
    from transformers import AutoModelForCausalLM, TrainerCallback

    from sokolm.evaluate import (answer_records, group_rows, group_summary, summary_rows, swap_baseline,
                                 zero_std_prediction)
    from sokolm.lm import generation_config
    from sokolm.splits import load_part
    from sokolm.train import grpo_examples, grpo_run_trainer, reward_function, step_timer

    t_start = time.perf_counter()
    fmt = Format()
    tok = load_tokenizer()
    gen_tok = load_tokenizer()  # generate() sets left padding on the tokenizer it gets; the trainer keeps its own
    ckpt_dir, ckpt_step = _checkpoint_dir(checkpoint)
    rl = load_part(split, "rl")
    examples = grpo_examples(tok, rl, fmt)
    counts = Counter(r["opt_len"] for r in rl)
    shares = {L: counts[L] / len(rl) for L in sorted(counts)}
    boards = _val_boards(split)
    in_dist = [b for b in boards if b["bucket"] == "in_dist"]
    greedy_cfg = generation_config(tok, "greedy")
    sample_cfg = generation_config(tok, "sample", temperature=1.0)
    prediction = _prediction(prediction_run, shares, g, checkpoint)
    _cap_gpu_memory(mem_cap_gib)
    config = {"experiment": "grpo", "split": split, "model": MODEL_REPO,
              "revision": MODEL_REVISION, "checkpoint": checkpoint, "checkpoint_step": ckpt_step,
              "precision": "fp32 weights, gradients and AdamW states; bf16 autocast; evaluation with a bf16 copy",
              "torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__,
              "gpu": torch.cuda.get_device_name(0), "format": fmt.name, "boards": GRPO_SELECTION,
              "train_boards": len(examples), "steps": steps, "prompts_per_step": prompts_per_step,
              "num_generations": g, "micro_batch": micro_batch, "lr": lr, "lr_schedule": lr_schedule,
              "boards_seen": steps * prompts_per_step, "epochs": round(steps * prompts_per_step / len(examples), 4),
              "eval_every": eval_every, "eval_samples": eval_samples, "eval_seed": GRPO_EVAL_SEED, "seed": seed,
              "gen_batch": gen_batch, "mem_cap_gib": mem_cap_gib, "save": save,
              "prediction_run": prediction_run}
    w = RunWriter(name, config, claim_level="own-experiment")

    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=torch.float32, device_map="cuda")
    eval_model = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    best = _BestWeights(eval_model) if save else None
    evals: list[dict] = []
    rollout_log: list[dict] = []
    step_stats: dict[int, dict] = {}
    groups_all: list[dict] = []

    def evaluate_now(step: int) -> None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        _copy_weights(model, eval_model)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):  # the training rollouts' RNG goes on
            greedy = answer_records(eval_model, gen_tok, boards, fmt, greedy_cfg, "greedy", seed=None,
                                    batch_size=gen_batch)
            sampled = answer_records(eval_model, gen_tok, in_dist, fmt, sample_cfg, "sample", n=eval_samples,
                                     seed=GRPO_EVAL_SEED, batch_size=gen_batch)
        table = summary_rows(greedy + sampled, fmt)
        rate = next(x for x in table if x["mode"] == "greedy" and x["bucket"] == "in_dist")["solve_rate"]
        kept = best is not None and best.offer(step, rate, eval_model)
        texts: dict[str, set] = {}
        for r in sampled:
            texts.setdefault(r["board_id"], set()).add(r["completion"])
        torch.cuda.synchronize()
        row = {"step": step, "table": table, "by_length": summary_rows(greedy + sampled, fmt, by_length=True),
               "swap": swap_baseline(boards, greedy, fmt), "samples_per_board": eval_samples,
               "distinct_per_board": round(float(np.mean([len(v) for v in texts.values()])), 3),
               "prediction": zero_std_prediction(sampled, shares, g) if eval_samples >= g else None,
               "kept_as_best": kept,
               "seconds": round(time.perf_counter() - t0, 1),
               "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
               "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)}
        evals.append(row)
        _write_answers(w, greedy + sampled, boards, step=step)
        ind = next(x for x in table if x["mode"] == "greedy" and x["bucket"] == "in_dist")
        smp = next(x for x in table if x["mode"] == "sample")
        pred = row["prediction"]
        print(f"eval step {step}: greedy " + ", ".join(f"{x['bucket']} {x['solve_rate']:.3f}" for x in table
                                                      if x["mode"] == "greedy")
              + f"; in_dist samples pass@1 {smp['pass@1']:.3f}, distinct/board {row['distinct_per_board']}, "
              + (f"predicted zero-std {pred['zero_std']:.3f}; " if pred else "")
              + f"parse fail {ind['parse_fail']:.3f}; "
              f"{row['seconds']} s, peak reserved {row['peak_reserved_mib']} MiB", flush=True)

    class Track(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            s = state.global_step
            mine = [r for r in rollout_log if r["step"] == s]
            rollout_log[:] = [r for r in rollout_log if r["step"] != s]
            rows = group_rows(mine, fmt)
            groups_all.extend(rows)
            step_stats[s] = {"boards": len({r["board_id"] for r in mine}), "completions": len(mine),
                             "reward_ours": float(np.mean([r["reward"] for r in mine])) if mine else None,
                             "truncated": _mean_of(r["truncated"] for r in mine) if mine else None,
                             "parse_fail": _mean_of(not r["parse_ok"] for r in mine) if mine else None,
                             **group_summary(rows, LENGTH_BANDS)}
            with open(w.dir / "rollouts.jsonl", "a") as f:
                for r in mine:
                    f.write(json.dumps({"step": s, **{k: r.get(k) for k in ROLLOUT_KEYS},
                                        "n_tokens": len(r["completion_ids"])}) + "\n")
            st = step_stats[s]
            print(f"step {s}: reward {st['reward_ours']}, zero-std {st['zero_std']} (all solved {st['all_solved']}, "
                  f"all failed {st['all_failed']}), boards {st['boards']}, completions {st['completions']}",
                  flush=True)
            if s % eval_every == 0 or s == args.max_steps:
                evaluate_now(s)

    evaluate_now(0)
    timer = step_timer()
    tr = grpo_run_trainer(model, tok, examples, reward_function(fmt, log=rollout_log), w.dir / "trainer",
                          prompts_per_step=prompts_per_step, num_generations=g, micro_batch=micro_batch, steps=steps,
                          lr=lr, lr_schedule=lr_schedule, seed=seed, callbacks=[timer, Track()])
    grpo_args = _trainer_args(tr, GRPO_TRAINER_ARGS)
    t0 = time.perf_counter()
    tr.train()
    train_s = round(time.perf_counter() - t0 - sum(r["seconds"] for r in evals[1:]), 1)
    checkpoints = _save_checkpoints(w, eval_model, best, evals, steps) if save else None

    logs = {h["step"]: h for h in tr.state.log_history if "loss" in h}
    rows = [{"step": t["step"], **{k: logs[t["step"]].get(k) for k in TRL_LOG_KEYS}, **step_stats[t["step"]],
             "seconds": t["seconds"], "peak_allocated_mib": t["peak_allocated_mib"],
             "peak_reserved_mib": t["peak_reserved_mib"]} for t in timer.steps]
    agree = {"steps": len(rows),
             "reward": all(abs(r["reward"] - r["reward_ours"]) < 1e-6 for r in rows),  # reward_ours is not rounded
             "zero_std": all(abs(r["frac_reward_zero_std"] - r["zero_std"]) < 5e-5 + 1e-9 for r in rows),  # 4 decimals
             "boards_per_step": all(r["boards"] == prompts_per_step for r in rows),
             "completions_per_step": all(r["completions"] == prompts_per_step * g for r in rows),
             "boards_repeated": len({x["board_id"] for x in groups_all}) != len(groups_all)}
    first = [x for x in groups_all if x["step"] <= eval_every]
    measured_first = {"steps": f"1-{min(eval_every, steps)}", **group_summary(first, LENGTH_BANDS),
                      "by_length": [{"opt_len": L, **{k: v for k, v in group_summary(
                          [x for x in first if x["opt_len"] == L], ()).items()
                          if k in ("groups", "zero_std", "all_solved", "all_failed")}} for L in sorted(shares)]}
    predicted_bands = {f"zero_std_{lo}_{hi}": _band_value(prediction, lo, hi) for lo, hi in LENGTH_BANDS}
    plots = w.dir / "plots"
    plots.mkdir(exist_ok=True)
    _plot_grpo(plots / "grpo.png", rows, evals, prediction,
               f"{w.run_id}: GRPO from {checkpoint}, {prompts_per_step} boards x {g} completions per step, "
               f"lr {lr:g} ({lr_schedule})")
    eval_s = round(sum(r["seconds"] for r in evals), 1)
    total_s = round(time.perf_counter() - t_start, 1)
    metrics = {
        "prediction": {**prediction, "bands": predicted_bands}, "measured_first_window": measured_first,
        "agreement_with_trl": agree, "evals": evals, "steps": rows, "grpo_args": grpo_args,
        "checkpoints": checkpoints,
        "timing": {"train_s": train_s, "eval_s": eval_s, "other_s": round(total_s - train_s - eval_s, 1),
                   "steady_step_s": _median([r["seconds"] for r in rows if r["step"] >= STEADY_FROM]),
                   "total_s": total_s},
        "gpu": {"train_peak_allocated_mib": max(r["peak_allocated_mib"] for r in rows),
                "train_peak_reserved_mib": max(r["peak_reserved_mib"] for r in rows),
                "eval_peak_allocated_mib": max(r["peak_allocated_mib"] for r in evals),
                "eval_peak_reserved_mib": max(r["peak_reserved_mib"] for r in evals)},
        "plots": ["plots/grpo.png"], "run_id": w.run_id,
    }
    w.close(metrics)
    return metrics


# ---------------------------------------------------------------------------------------- validation loss of checkpoints


def _in_training_val_loss(run_id: str, step: int) -> Optional[float]:
    """The val loss an sft run recorded at `step`; None where it recorded none (grpo runs record none)."""
    from sokolm.paths import RUNS

    evals = json.loads((RUNS / run_id / "metrics.json").read_text()).get("evals", [])
    return next((e["val"]["loss"] for e in evals if e.get("step") == step and "val" in e), None)


def run_val_loss(checkpoints: list[str], split: str = "7x7-v1", name: str = "val-loss",
                 mem_cap_gib: float = EVAL_MEM_CAP_GIB) -> dict:
    """The SFT loss (`_sft_val_loss`) of saved checkpoints ('<run id>/<best|final>' of sft or grpo runs) on the BFS
    answers of the first 300 val boards of each bucket, one checkpoint at a time. A GRPO run never trains on these
    answers and records no such loss; this puts its checkpoints on the same scale as the SFT runs.

    Each checkpoint is loaded with fp32 weights straight to the GPU (bf16 weights read as fp32 are exact) and scored
    under bf16 autocast, as `run_sft` scores its training weights, so the checkpoint an sft run continued from gets
    that run's step-0 val loss on in_dist. An sft run scored its fp32 training weights; its saved checkpoints hold
    them rounded to bf16, so `in_training` (the run's own value at that step, where it has one) can differ a little."""
    import gc

    import torch
    from transformers import AutoModelForCausalLM

    from sokolm.paths import RUNS
    from sokolm.train import sft_examples

    t_start = time.perf_counter()
    fmt = Format()
    tok = load_tokenizer()
    val = load(split, "val")
    examples = {b: sft_examples(tok, [x for x in val if x["bucket"] == b][:300], fmt) for b in BUCKETS}
    _cap_gpu_memory(mem_cap_gib)
    config = {"experiment": "val_loss", "split": split, "model": MODEL_REPO,
              "checkpoints": checkpoints, "format": fmt.name,
              "boards": "the first 300 val boards of each bucket, in file order (900); the target is each board's "
                        "stored BFS answer line and <|im_end|>",
              "precision": "fp32 weights read from the bf16 checkpoint (exact), bf16 autocast",
              "loss_batch": SFT_LOSS_BATCH, "mem_cap_gib": mem_cap_gib}
    w = RunWriter(name, config, claim_level="own-experiment")
    results = []
    for spec in checkpoints:
        ckpt_dir, step = _checkpoint_dir(spec)
        run_id = spec.partition("/")[0]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=torch.float32, device_map="cuda").eval()
        loss = {b: _sft_val_loss(model, tok, ex) for b, ex in examples.items()}
        del model
        gc.collect()
        torch.cuda.synchronize()
        results.append({"checkpoint": spec, "step": step,
                        "experiment": json.loads((RUNS / run_id / "manifest.json").read_text())["config"]["experiment"],
                        "loss": loss, "in_training": _in_training_val_loss(run_id, step),
                        "seconds": round(time.perf_counter() - t0, 1),
                        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)})
        torch.cuda.empty_cache()
    metrics = {"results": results, "seconds": round(time.perf_counter() - t_start, 1), "run_id": w.run_id}
    w.close(metrics)
    return metrics
