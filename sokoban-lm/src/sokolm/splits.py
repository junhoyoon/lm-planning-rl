"""Fixed train/validation/test splits.

What counts as one problem: boards with the same walls and targets are the same
problem whatever the box and player positions, because solving one can pass through the other's states; boards
related by a rotation or reflection are the same problem; a different target makes a different problem. The split
unit is therefore the layout class, canonical(layout(text)). A salted hash of the class puts it in exactly one split,
so no problem appears in two splits, and every split takes at most one board per class.

Buckets:
  train      in_dist  1 box, optimal length < long_from: the training board pool
  val, test  in_dist  same range
             long     1 box, optimal length >= long_from; never trained on
             two_box  2 boxes, same generator; never trained on

Boards come from two seeded generator streams (one box, two boxes), read in order. A candidate's class and split
are known before its BFS, so the BFS runs only for candidates that can still fill a quota. The files are a
deterministic function of the config.

Files: DATA/splits/<name>/{train,val,test}.jsonl (`sokolm.paths`; not tracked by git) and splits/<name>.json in this
project (tracked: config, counts, length statistics, sha256 of every file). `load` checks the sha256, so every later
run reads exactly the files the manifest pins.

Train parts: the train pool is split once more into the boards SFT trains on ("sft") and the boards RL trains on
("rl"), half each, with the share of every optimal length kept, so that continued SFT and RL can be compared on the
"rl" boards, which the first SFT never saw. `train_parts` sorts the pool by
optimal length and then by a salted hash of the board id and deals the boards alternately to the two parts, so each
length goes half and half (one board more on one side when its count is odd) and the parts are a deterministic
function of the train file.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from sokolm.generate import stream
from sokolm.paths import DATA, PROJECT
from sokolm.simulate import simulate
from sokolm.sokoban import PUSH, Board, action_letters, parse_actions
from sokolm.solver import solve
from sokolm.symmetry import canonical, layout

SPLITS = {
    "7x7-v1": {
        "size": 7,
        "max_wall_frac": 0.25,
        "wall_sampling": "per_set",
        "salt": "sokolm-split-v1",
        "val_share": 0.1,
        "test_share": 0.1,
        "long_from": 14,
        "quota": {"val": 300, "test": 500},  # boards per bucket
        "train_pool": 50000,
        "seeds": {"one_box": 1, "two_box": 2},
        "max_candidates": 10_000_000,  # per stream; a guard against a config whose quotas cannot be filled
    },
}
TRAIN_PARTS = {"7x7-v1": "sokolm-train-parts-v1"}  # salt of the hash that orders boards of one optimal length
PART_NAMES = ("sft", "rl")
SPLIT_NAMES = ("train", "val", "test")
BUCKETS = ("in_dist", "long", "two_box")
MANIFESTS = PROJECT / "splits"


def layout_class(text: str) -> str:
    """Canonical layout text: walls and targets only, the smallest of the 8 rotations/reflections."""
    return canonical(layout(text))


def class_id(cls: str) -> str:
    return hashlib.sha256(cls.encode()).hexdigest()[:16]


def room(text: str) -> str:
    """Canonical walls-only text (targets removed too): the room a layout is drawn in."""
    return canonical(layout(text).replace(".", " "))


def split_of(cls: str, cfg: dict) -> str:
    u = int(hashlib.sha256(f"{cfg['salt']}|{cls}".encode()).hexdigest()[:15], 16) / 16**15
    if u < cfg["val_share"]:
        return "val"
    if u < cfg["val_share"] + cfg["test_share"]:
        return "test"
    return "train"


def quotas(cfg: dict) -> dict[tuple[str, str], int]:
    q = {("train", "in_dist"): cfg["train_pool"]}
    for split in ("val", "test"):
        for bucket in BUCKETS:
            q[(split, bucket)] = cfg["quota"][split]
    return q


def _interior_walls(text: str) -> int:
    return sum(ch == "#" for row in text.split("\n")[1:-1] for ch in row[1:-1])


def build(cfg: dict, name: str, progress: Optional[Callable[[str], None]] = None) -> tuple[dict, dict]:
    """Records per split ({split: [record, ...]}) and per-stream counts of what happened to the candidates."""
    need = quotas(cfg)
    taken: set[str] = set()  # layout classes that already hold a board
    records: dict[str, list[dict]] = {s: [] for s in SPLIT_NAMES}
    streams = {}
    for n_boxes, stream_name in ((1, "one_box"), (2, "two_box")):
        mine = [k for k in need if (k[1] == "two_box") == (n_boxes == 2)]
        st = Counter()
        candidates = stream(cfg["size"], cfg["seeds"][stream_name], n_boxes, cfg["max_wall_frac"], cfg["wall_sampling"])
        while any(need[k] for k in mine):
            if st["candidates"] >= cfg["max_candidates"]:
                raise RuntimeError(f"{stream_name}: quotas not filled after {st['candidates']} candidates: {need}")
            index, text, why = next(candidates)
            st["candidates"] += 1
            if text is None:
                st[why] += 1
                continue
            cls = layout_class(text)
            if cls in taken:
                st["class_taken"] += 1
                continue
            split = split_of(cls, cfg)
            if not any(need[k] for k in mine if k[0] == split):
                st[f"split_full {split}"] += 1
                continue
            board = Board(text)
            sol = solve(board)
            st["bfs"] += 1
            if not sol.solvable:
                st["unsolvable"] += 1
                continue
            bucket = "two_box" if n_boxes == 2 else ("long" if sol.length >= cfg["long_from"] else "in_dist")
            if not need.get((split, bucket)):
                st[f"no_room {split}/{bucket}"] += 1
                continue
            need[(split, bucket)] -= 1
            taken.add(cls)
            events = simulate(board, sol.actions, sol.length).events
            records[split].append({
                "board_id": f"{name}-{n_boxes}b-{index:07d}", "split": split, "bucket": bucket, "n_boxes": n_boxes,
                "text": text, "opt_len": sol.length, "budget": 2 * sol.length,
                "solution": action_letters(sol.actions), "n_pushes": events.count(PUSH), "n_optimal": sol.n_optimal,
                "bfs_states": sol.n_states, "n_walls": _interior_walls(text), "layout_class": class_id(cls),
                "stream_index": index,
            })
            st["accepted"] += 1
            if progress and st["accepted"] % 5000 == 0:
                progress(f"{stream_name}: {st['accepted']} accepted, {st['candidates']} candidates")
        streams[stream_name] = dict(sorted(st.items()))
    for split in SPLIT_NAMES:
        records[split].sort(key=lambda r: (BUCKETS.index(r["bucket"]), r["stream_index"]))
    return records, streams


def verify(records: dict, cfg: dict) -> dict:
    """Recompute everything a record claims; raise on the first violation, return the counts checked."""
    need = quotas(cfg)
    counts = Counter((r["split"], r["bucket"]) for rs in records.values() for r in rs)
    if dict(counts) != {k: v for k, v in need.items() if v}:
        raise AssertionError(f"bucket counts {dict(counts)} differ from the quotas {need}")
    classes, texts, n = set(), set(), 0
    for split, rs in records.items():
        for r in rs:
            cls = layout_class(r["text"])
            assert r["split"] == split == split_of(cls, cfg), r["board_id"]
            assert class_id(cls) == r["layout_class"] and r["layout_class"] not in classes, r["board_id"]
            assert r["text"] not in texts, r["board_id"]
            classes.add(r["layout_class"])
            texts.add(r["text"])
            board = Board(r["text"])
            assert len(board.targets) == r["n_boxes"], r["board_id"]
            if r["bucket"] == "in_dist":
                assert r["n_boxes"] == 1 and 1 <= r["opt_len"] < cfg["long_from"], r["board_id"]
            elif r["bucket"] == "long":
                assert r["n_boxes"] == 1 and r["opt_len"] >= cfg["long_from"], r["board_id"]
            else:
                assert r["n_boxes"] == 2, r["board_id"]
            actions = parse_actions(r["solution"])
            o = simulate(board, actions, r["budget"])
            assert len(actions) == r["opt_len"] and o.solved_at == r["opt_len"] and o.n_noop == 0, r["board_id"]
            assert r["budget"] == 2 * r["opt_len"], r["board_id"]
            n += 1
    return {"records": n, "distinct_layout_classes": len(classes), "distinct_texts": len(texts),
            "split_matches_class_hash": True, "one_board_per_class": True, "bucket_ranges": True,
            "solutions_replay_to_solved_at_opt_len": True}


def summarize(records: dict) -> dict:
    """Per split and bucket: optimal length, pushes, interior walls, plus how many held-out rooms train also uses."""
    out: dict = {}
    for split, rs in records.items():
        for bucket in BUCKETS:
            sel = [r for r in rs if r["bucket"] == bucket]
            if not sel:
                continue
            L = np.array([r["opt_len"] for r in sel])
            out.setdefault(split, {})[bucket] = {
                "boards": len(sel),
                "opt_len": {"min": int(L.min()), "median": float(np.median(L)), "mean": round(float(L.mean()), 3),
                            "p90": int(np.percentile(L, 90, method="lower")), "max": int(L.max()),
                            "hist": {str(k): int(c) for k, c in enumerate(np.bincount(L)) if c}},
                "n_pushes_median": float(np.median([r["n_pushes"] for r in sel])),
                "n_walls_hist": dict(sorted(Counter(str(r["n_walls"]) for r in sel).items(), key=lambda kv: int(kv[0]))),
                "max_budget": int(2 * L.max()),
            }
    train_rooms = {room(r["text"]) for r in records["train"]}
    out["held_out_room_in_train"] = {
        f"{split}/{bucket}": round(float(np.mean([room(r["text"]) in train_rooms for r in records[split]
                                                  if r["bucket"] == bucket])), 4)
        for split in ("val", "test") for bucket in BUCKETS if any(r["bucket"] == bucket for r in records[split])}
    return out


def _paths(name: str) -> tuple[Path, Path]:
    return DATA / "splits" / name, MANIFESTS / f"{name}.json"


def check_writable(name: str, force: bool = False):
    data_dir, manifest = _paths(name)
    existing = [p for p in [manifest, *(data_dir / f"{s}.jsonl" for s in SPLIT_NAMES)] if p.exists()]
    if existing and not force:
        raise SystemExit(f"split {name} already exists ({', '.join(str(p) for p in existing)}); "
                         f"a fixed split is not rebuilt unless --force is given")


def write(name: str, cfg: dict, records: dict, extra: dict, force: bool = False) -> dict:
    """Write the data files and the tracked manifest; return the manifest."""
    check_writable(name, force)
    data_dir, manifest_path = _paths(name)
    data_dir.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in SPLIT_NAMES:
        body = "".join(json.dumps(r) + "\n" for r in records[split]).encode()
        path = data_dir / f"{split}.jsonl"
        path.write_bytes(body)
        files[split] = {"path": f"data/splits/{name}/{split}.jsonl", "records": len(records[split]),
                        "sha256": hashlib.sha256(body).hexdigest()}
    counts = {s: dict(Counter(r["bucket"] for r in records[s])) for s in SPLIT_NAMES}
    manifest = {"name": name, "config": cfg, "files": files, "counts": counts, **extra}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def load(name: str, split: str, bucket: Optional[str] = None) -> list[dict]:
    """Records of one split (optionally one bucket) after checking the file against the tracked manifest."""
    data_dir, manifest_path = _paths(name)
    entry = json.loads(manifest_path.read_text())["files"][split]
    body = (data_dir / f"{split}.jsonl").read_bytes()
    if hashlib.sha256(body).hexdigest() != entry["sha256"]:
        raise ValueError(f"{entry['path']} does not match the sha256 in {manifest_path}; rebuild with "
                         f"`python -m sokolm build-split --name {name} --force` at the manifest's commit")
    recs = [json.loads(line) for line in body.decode().splitlines()]
    return [r for r in recs if bucket is None or r["bucket"] == bucket]


def train_parts(records: list[dict], salt: str) -> dict[str, list[dict]]:
    """{"sft": [...], "rl": [...]}: the train records dealt alternately in (optimal length, salted hash of the board
    id) order; each part keeps the train file order."""
    def key(r):
        return r["opt_len"], hashlib.sha256(f"{salt}|{r['board_id']}".encode()).hexdigest()

    order = sorted(range(len(records)), key=lambda i: key(records[i]))
    picked = {part: sorted(order[k::2]) for k, part in enumerate(PART_NAMES)}
    return {part: [records[i] for i in idx] for part, idx in picked.items()}


def load_part(name: str, part: str) -> list[dict]:
    """The train boards of one part ("sft" or "rl"), from the manifest-checked train file."""
    if part not in PART_NAMES:
        raise ValueError(f"part must be one of {PART_NAMES}, got {part!r}")
    return train_parts(load(name, "train"), TRAIN_PARTS[name])[part]
