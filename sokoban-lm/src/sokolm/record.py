"""Run directories.

Layout (RUNS: `sokolm.paths`):
  RUNS/<run_id>/manifest.json     provenance (git commit, dirty flag, environment) + config + claim level
  RUNS/<run_id>/code.diff         uncommitted changes the run was made with (only when the tree was dirty)
  RUNS/<run_id>/episodes.jsonl    one line per episode (board, answer, judged outcome)
  RUNS/<run_id>/metrics.json      aggregates
"""

import datetime as _dt
import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Optional

from sokolm.paths import PROJECT, REPO, RUNS

MAX_UNTRACKED_DIFF_BYTES = 1_000_000


def _git(*args) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True)
        return r.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def code_diff() -> str:
    """Patch of uncommitted changes: tracked files vs HEAD plus untracked, non-ignored files (large ones listed only)."""
    parts = [_git("diff", "HEAD", "--binary") or ""]
    for f in (_git("ls-files", "--others", "--exclude-standard") or "").splitlines():
        p = REPO / f
        if p.is_file() and p.stat().st_size > MAX_UNTRACKED_DIFF_BYTES:
            parts.append(f"# untracked file not included (size {p.stat().st_size} bytes): {f}\n")
            continue
        r = subprocess.run(["git", "-C", str(REPO), "diff", "--no-index", "--binary", "--", "/dev/null", f],
                           capture_output=True, text=True)  # exit status 1 means "files differ"
        parts.append(r.stdout)
    return "".join(parts)


def _sha256(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _version(module: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(module)
    except Exception:
        return None


def provenance() -> dict:
    status = _git("status", "--porcelain")
    return {
        "code": {
            "commit": (_git("rev-parse", "HEAD") or "").strip() or None,
            "dirty": bool(status),
            "dirty_paths": status.splitlines() if status else [],
        },
        "env": {
            "python": platform.python_version(),
            **{m: _version(m) for m in ("numpy", "torch", "transformers", "trl", "accelerate", "datasets")},
            "uv_lock_sha256": _sha256(PROJECT / "uv.lock"),
            "threads": os.environ.get("OMP_NUM_THREADS"),
            "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
    }


class RunWriter:
    def __init__(self, name: str, config: dict, claim_level: str):
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}-{re.sub(r'[^A-Za-z0-9._+-]', '-', name)}"
        self.dir = RUNS / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "run_id": self.run_id,
            "name": name,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "claim_level": claim_level,
            "config": config,
            **provenance(),
        }
        if self.manifest["code"]["dirty"]:
            (self.dir / "code.diff").write_text(code_diff())
            self.manifest["code"]["diff"] = "code.diff"
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")

    def add(self, rec: dict) -> str:
        """Log one episode record (must have `episode_id`)."""
        with open(self.dir / "episodes.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec["episode_id"]

    def close(self, metrics: Optional[dict] = None):
        if metrics is not None:
            (self.dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
