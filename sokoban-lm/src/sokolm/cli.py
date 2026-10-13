"""Command line entry point: `uv run python -m sokolm <command> ...` (from this project's directory).

Thread pools are capped before numpy/torch are imported (SOKOLM_THREADS, default 4), the CUDA caching allocator uses
expandable segments (below), and only one experiment runs at a time (RUNS/.lock, held until the process exits). Every
experiment also runs under sokolm.guard: it refuses to start when Windows or WSL is short of RAM or the GPU is busy,
logs RAM and GPU every 2 s to RUNS/resources/, and stops the run when a limit is crossed. The guard needs WSL2 (it
reads Windows RAM through PowerShell); elsewhere set SOKOLM_GUARD=0 to run without it (no limits and no resource log).
"""

import os

_THREADS = os.environ.get("SOKOLM_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)
# The KV cache of HF generate grows by one token per step (torch.cat). With the default allocator every slightly
# larger tensor needs a new block, and the freed smaller ones stay reserved: one 4096-token generation filled the
# whole 32 GB GPU, and a 1024-token measurement reserved 8.2 GB (its cap) for 3.3 GB of live tensors. Expandable
# segments let freed memory be reused (3.6 GB reserved in the same measurement, same speed and outputs).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import fcntl  # noqa: E402
import resource  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

from sokolm.paths import RUNS  # noqa: E402


_LOCK = None


def hold_experiment_lock():
    """Refuse to start while another sokolm experiment holds RUNS/.lock. The lock is never released here: the
    kernel releases it when this process exits, so the next experiment cannot start while this one is still tearing
    down (the guard's GPU check covers the moment after that)."""
    global _LOCK
    if _LOCK is not None:
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    f = open(RUNS / ".lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise SystemExit(f"another sokolm experiment is running ({RUNS / '.lock'}); run experiments one at a time")
    _LOCK = f


@contextlib.contextmanager
def single_experiment_lock():
    """For scripts outside the CLI: the same lock, held until the process exits."""
    hold_experiment_lock()
    yield


def start_guard(command: str) -> subprocess.Popen:
    """Start sokolm.guard for this process and wait for its start check; exit if it refuses the run."""
    from sokolm.guard import INTERVAL_S, log_path

    log = log_path(command)
    g = subprocess.Popen([sys.executable, "-m", "sokolm.guard", "--pid", str(os.getpid()), "--log", str(log)],
                         stdout=subprocess.PIPE, text=True)
    answer = g.stdout.readline().strip()
    if answer != "ready":
        g.wait()
        raise SystemExit(f"resource guard: not starting {command}: {answer.removeprefix('refuse ') or 'no answer'} "
                         f"(log {log})")
    print(f"resource guard: sampling every {INTERVAL_S:g} s, log {log}", flush=True)
    return g


def stop_guard(g: subprocess.Popen):
    g.send_signal(signal.SIGTERM)
    try:
        g.wait(timeout=15)
    except subprocess.TimeoutExpired:
        g.kill()


def _print_table(rows: list[dict]) -> None:
    cols = list(dict.fromkeys(c for r in rows for c in r))
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")) for c in cols))
    print()


def cmd_build_split(a):
    from sokolm.pipeline import build_split

    m = build_split(name=a.name, force=a.force, progress=print)
    _print_table(m["table"])
    print("held-out boards whose room (walls up to symmetry) is also in train:", m["summary"]["held_out_room_in_train"])
    for name, st in m["streams"].items():
        print(f"{name} stream: {st}")
    print("checks:", m["checks"])
    for split, f in m["files"].items():
        print(f"{split}: {f['records']} records  {f['path']}  sha256 {f['sha256'][:16]}")
    print(f"manifest: {m['manifest']}")
    print(f"run: {RUNS / m['run_id']}")


def cmd_new_boards(a):
    from sokolm.pipeline import run_new_boards

    m = run_new_boards(n=a.n, split=a.split, name=a.name, progress=print)
    print("stream:", m["stream"])
    print("checks:", m["checks"])
    print("optimal length: share of new boards / of train boards: "
          + ", ".join(f"{r['opt_len']} {r['new_share']}/{r['train_share']}" for r in m["lengths"]))
    print(f"boards: {RUNS / m['run_id'] / m['file']}  sha256 {m['sha256'][:16]}")
    print(f"run: {RUNS / m['run_id']}")


def cmd_fetch(a):
    from sokolm.assets import MODELS, fetch

    for key in (MODELS if a.model == "all" else [a.model]):
        m = fetch(key)
        print(f"{key}: {m['repo']}@{m['revision']}: " + ", ".join(f"{f} {h[:12]}" for f, h in m["files"].items()))


def cmd_eval(a):
    from sokolm.pipeline import run_eval

    m = run_eval(split=a.split, name=a.name, n_per_bucket=a.n, n_samples=a.samples, temperature=a.temperature,
                 seed=a.seed, batch_size=a.batch_size, checkpoint=a.checkpoint, boards_file=a.boards)
    _print_table(m["table"])
    if "swap" in m:
        _print_table(m["swap"])
    print("parse errors:", m["parse_errors"])
    print("timing:", m["timing"], "gpu:", m["gpu"])
    print(f"run: {RUNS / m['run_id']}")


def cmd_sft(a):
    from sokolm.pipeline import run_sft

    m = run_sft(split=a.split, name=a.name, lr=a.lr, steps=a.steps, warmup=a.warmup, batch_size=a.batch_size,
                eval_every=a.eval_every, seed=a.seed, gen_batch=a.gen_batch, train_eval=a.train_eval, save=a.save,
                init=a.init, part=a.part, step0_reference=a.step0_reference, lr_schedule=a.lr_schedule)
    for r in m["evals"]:
        print(f"step {r['step']}: val loss {r['val']['loss']}, nll {r['val']['nll_mean']}, greedy "
              + ", ".join(f"{x['bucket']} {x['solve_rate']}" for x in r["table"])
              + f", swap in_dist {next(x['swap'] for x in r['swap'] if x['bucket'] == 'in_dist')}")
    if m["step0_vs_reference"]:
        print("step 0 against the reference run:", m["step0_vs_reference"])
    print("selection:", m["selection"])
    if m["checkpoints"]:
        print("checkpoints:", m["checkpoints"])
    print("timing:", m["timing"], "gpu:", m["gpu"])
    print(f"run: {RUNS / m['run_id']}")


def cmd_grpo(a):
    from sokolm.pipeline import run_grpo

    m = run_grpo(a.checkpoint, a.prediction_run, split=a.split, name=a.name, steps=a.steps,
                 prompts_per_step=a.prompts, g=a.g, micro_batch=a.micro_batch, lr=a.lr, lr_schedule=a.lr_schedule,
                 eval_every=a.eval_every, eval_samples=a.eval_samples, seed=a.seed, gen_batch=a.gen_batch,
                 save=not a.no_save)
    p = m["prediction"]
    f = m["measured_first_window"]
    print(f"zero-std groups, predicted from {p['run']}: {p['zero_std']} (all solved {p['all_solved']}, all failed "
          f"{p['all_failed']}), bands {p['bands']}")
    print(f"zero-std groups, measured in steps {f['steps']}: {f['zero_std']} (all solved {f['all_solved']}, all "
          f"failed {f['all_failed']}), bands " + str({k: f[k] for k in p["bands"]}))
    for r in m["evals"]:
        smp = next(x for x in r["table"] if x["mode"] == "sample")
        print(f"eval step {r['step']}: greedy " + ", ".join(f"{x['bucket']} {x['solve_rate']}" for x in r["table"]
                                                           if x["mode"] == "greedy")
              + f", swap in_dist {next(x['swap'] for x in r['swap'] if x['bucket'] == 'in_dist')}; in_dist samples "
              f"pass@1 {smp['pass@1']}, distinct/board {r['distinct_per_board']}")
    print("agreement with TRL's logs:", m["agreement_with_trl"])
    if m["checkpoints"]:
        print("checkpoints:", m["checkpoints"])
    print("timing:", m["timing"], "gpu:", m["gpu"])
    print(f"run: {RUNS / m['run_id']}")


def cmd_val_loss(a):
    from sokolm.pipeline import run_val_loss

    m = run_val_loss(a.checkpoints, split=a.split, name=a.name)
    for r in m["results"]:
        print(f"{r['checkpoint']} (step {r['step']}, {r['experiment']}): "
              + ", ".join(f"{b} {x['loss']}" for b, x in r["loss"].items())
              + f"; in_dist nll per role {r['loss']['in_dist']['nll_mean']}; in-training val loss {r['in_training']}; "
              f"{r['seconds']} s, peak allocated {r['peak_allocated_mib']} MiB, reserved {r['peak_reserved_mib']} MiB")
    print(f"run: {RUNS / m['run_id']}")


def cmd_solve(a):
    from sokolm.sokoban import Board, action_letters
    from sokolm.solver import solve

    board = Board(open(a.file).read())
    sol = solve(board)
    print(board.render())
    if sol.solvable:
        print(f"optimal length {sol.length}: {action_letters(sol.actions)}  ({sol.n_optimal} optimal solutions, "
              f"{sol.n_states} states, layer sizes {sol.layer_sizes})")
    else:
        print(f"unsolvable ({sol.n_states} states explored)")


def cmd_resources(a):
    from sokolm.guard import check

    raise SystemExit(check())


def main():
    from sokolm import pipeline as pl

    p = argparse.ArgumentParser(prog="sokolm")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("build-split", help="build the fixed train/val/test split files and their statistics")
    s.add_argument("--name", default="7x7-v1", help="a key of sokolm.splits.SPLITS")
    s.add_argument("--force", action="store_true", help="overwrite an existing split (data files and manifest)")
    s.set_defaults(fn=cmd_build_split, lock=True)

    s = sub.add_parser("new-boards", help="one-box boards beyond the split: layout classes that hash to train but hold "
                                          "no board of the split, in the in_dist length range")
    s.add_argument("--split", default="7x7-v1")
    s.add_argument("--name", default="new-boards")
    s.add_argument("--n", type=int, default=pl.NEW_BOARDS_N)
    s.set_defaults(fn=cmd_new_boards, lock=True)

    s = sub.add_parser("fetch", help="download the pinned files of the model and record their sha256")
    s.add_argument("--model", default="qwen2.5-0.5b-instruct", help="a key of sokolm.assets.MODELS, or all")
    s.set_defaults(fn=cmd_fetch, lock=False)

    s = sub.add_parser("eval", help="greedy solve rate, pass@k and board-blind baselines on the val boards (or the "
                                    "boards of a file), for the base model or a saved checkpoint")
    s.add_argument("--split", default="7x7-v1")
    s.add_argument("--name", default="eval")
    s.add_argument("--n", type=int, default=300, help="val boards per bucket (first in file order)")
    s.add_argument("--samples", type=int, default=16, help="sampled answers per board for pass@k")
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--batch-size", type=int, default=pl.EVAL_BATCH, help="sequences per generate call")
    s.add_argument("--checkpoint", default=None,
                   help="evaluate a saved checkpoint instead of the base model: <sft or grpo run id>/<best|final>")
    s.add_argument("--boards", default=None,
                   help="evaluate on the boards of this JSONL file (from new-boards) instead of the val boards; "
                        "--n is then unused and the board-blind baselines are skipped")
    s.set_defaults(fn=cmd_eval, lock=True)

    s = sub.add_parser("sft", help="one SFT run (TRL SFTTrainer), with val loss and val greedy solve rates during "
                                   "training")
    s.add_argument("--split", default="7x7-v1")
    s.add_argument("--name", default="sft")
    s.add_argument("--lr", type=float, default=pl.SFT_LR)
    s.add_argument("--steps", type=int, default=pl.SFT_STEPS)
    s.add_argument("--warmup", type=int, default=pl.SFT_WARMUP, help="steps of linear warmup")
    s.add_argument("--batch-size", type=int, default=pl.SFT_BATCH, help="boards per optimizer step")
    s.add_argument("--eval-every", type=int, default=pl.SFT_EVAL_EVERY,
                   help="steps between evaluations (also at step 0 and the end)")
    s.add_argument("--gen-batch", type=int, default=pl.SFT_GEN_BATCH, help="sequences per generate call at an evaluation")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--train-eval", type=int, default=pl.SFT_TRAIN_EVAL,
                   help="also evaluate greedily on this many training boards, the first of the part (0: none)")
    s.add_argument("--save", action="store_true",
                   help="save the bf16 weights of the best evaluation (val in_dist greedy) and of the last step")
    s.add_argument("--init", default=None, help="continue from a saved checkpoint '<sft run id>/<best|final>'")
    s.add_argument("--part", default="sft", choices=("sft", "rl"),
                   help="train part: 'sft', or 'rl' (the boards GRPO trains on)")
    s.add_argument("--lr-schedule", default="constant", choices=("constant", "linear"),
                   help="after the warmup: keep the lr constant, or lower it linearly to 0 at the last step")
    s.add_argument("--step0-reference", default=None,
                   help="an eval run of the starting model whose greedy val answers the step-0 answers are compared with")
    s.set_defaults(fn=cmd_sft, lock=True)

    s = sub.add_parser("grpo", help="GRPO from a saved checkpoint with TRL's GRPOTrainer, logging the groups with "
                                    "all-equal rewards per length band, entropy and val solve rates during training")
    s.add_argument("--checkpoint", required=True, help="the checkpoint to start from: <sft run id>/<best|final>")
    s.add_argument("--prediction-run", required=True,
                   help="eval run of the starting checkpoint whose sampled val in_dist answers predict the share of "
                        "groups with all-equal rewards")
    s.add_argument("--split", default="7x7-v1")
    s.add_argument("--name", default="grpo")
    s.add_argument("--steps", type=int, default=pl.GRPO_STEPS)
    s.add_argument("--prompts", type=int, default=pl.GRPO_PROMPTS, help="boards per optimizer step")
    s.add_argument("--g", type=int, default=pl.GRPO_G, help="completions per board (GRPO group size)")
    s.add_argument("--micro-batch", type=int, default=pl.GRPO_MICRO, help="completions per loss micro-batch")
    s.add_argument("--lr", type=float, default=pl.GRPO_LR)
    s.add_argument("--lr-schedule", default="constant", choices=("constant", "linear"))
    s.add_argument("--eval-every", type=int, default=pl.GRPO_EVAL_EVERY,
                   help="steps between evaluations (also at step 0 and the end)")
    s.add_argument("--eval-samples", type=int, default=pl.GRPO_EVAL_SAMPLES,
                   help="sampled answers per val in_dist board at evaluations")
    s.add_argument("--gen-batch", type=int, default=pl.GRPO_GEN_BATCH, help="sequences per generate call at an evaluation")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-save", action="store_true", help="do not save the best and final checkpoints")
    s.set_defaults(fn=cmd_grpo, lock=True)

    s = sub.add_parser("val-loss", help="the SFT loss on the BFS answers of the val boards (300 per bucket) of saved "
                                        "sft or grpo checkpoints, one at a time")
    s.add_argument("checkpoints", nargs="+", help="<sft or grpo run id>/<best|final>")
    s.add_argument("--split", default="7x7-v1")
    s.add_argument("--name", default="val-loss")
    s.set_defaults(fn=cmd_val_loss, lock=True)

    s = sub.add_parser("solve", help="BFS on an XSB board file")
    s.add_argument("file")
    s.set_defaults(fn=cmd_solve, lock=False)

    s = sub.add_parser("resources", help="current Windows/WSL RAM and GPU use, and whether an experiment could start")
    s.set_defaults(fn=cmd_resources, lock=False)

    a = p.parse_args()
    t0 = time.time()
    if not a.lock:
        a.fn(a)
        return
    hold_experiment_lock()
    g = start_guard(a.cmd) if os.environ.get("SOKOLM_GUARD", "1") != "0" else None
    if g is None:
        print("resource guard: off (SOKOLM_GUARD=0)", flush=True)
    try:
        a.fn(a)
    finally:
        if g is not None:
            stop_guard(g)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[{a.cmd}] done in {time.time() - t0:.1f}s, peak RSS {rss:.0f} MB, threads {_THREADS}")
