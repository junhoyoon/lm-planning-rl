"""Pinned Hugging Face files of the model and their hashes.

The model is Qwen2.5-0.5B-Instruct at a pinned commit. Files land in DATA/hf/<repo name>/ (`sokolm.paths`; not
tracked by git). assets.lock.json (tracked) records, per model, the repo, the commit and the sha256 of every fetched
file, and loading checks those hashes first, so every run reads exactly the pinned files.
"""

import hashlib
import json
from pathlib import Path

from sokolm.paths import DATA, PROJECT

MODELS = {  # key -> (hub repo, pinned commit); the comment is the commit date
    "qwen2.5-0.5b-instruct": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),  # 2024-09-25
}
MAIN = "qwen2.5-0.5b-instruct"
MODEL_REPO, MODEL_REVISION = MODELS[MAIN]
FETCH_SUFFIXES = (".json", ".jinja", ".txt", ".model", ".safetensors")  # configs, tokenizer, weights; no docs
LOCK = PROJECT / "assets.lock.json"


def model_dir(key: str = MAIN) -> Path:
    return DATA / "hf" / MODELS[key][0].split("/")[-1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry(key: str) -> dict:
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    entry = lock.get("models", {}).get(key)
    if entry is None or (entry["repo"], entry["revision"]) != MODELS[key]:
        raise SystemExit(f"{key} is not pinned in {LOCK} at {MODELS[key]}; run `python -m sokolm fetch --model {key}`")
    return entry


def fetch(key: str = MAIN) -> dict:
    """Download the top-level config, tokenizer and weight files of `key` at the pinned commit and record their
    sha256 in the lock (other models' entries are kept)."""
    from huggingface_hub import HfApi, hf_hub_download

    repo, revision = MODELS[key]
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    entry = lock.setdefault("models", {}).setdefault(key, {"repo": repo, "revision": revision, "files": {}})
    if (entry["repo"], entry["revision"]) != (repo, revision):
        raise SystemExit(f"{LOCK} pins {key} at {entry['repo']}@{entry['revision']}, the code at {repo}@{revision}")
    names = sorted(s.rfilename for s in HfApi().model_info(repo, revision=revision).siblings)
    for f in [n for n in names if "/" not in n and n.endswith(FETCH_SUFFIXES)]:
        local = hf_hub_download(repo, f, revision=revision, local_dir=model_dir(key))
        entry["files"][f] = _sha256(Path(local))
    entry["files"] = dict(sorted(entry["files"].items()))
    lock["models"] = dict(sorted(lock["models"].items()))
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    return entry


def verify(key: str = MAIN, weights: bool = True) -> Path:
    """Raise unless every pinned file of `key` (without the weights if `weights` is false) exists locally with the
    sha256 recorded in the lock; return the model directory."""
    for f, digest in _entry(key)["files"].items():
        if not weights and f.endswith(".safetensors"):
            continue
        path = model_dir(key) / f
        if not path.exists():
            raise SystemExit(f"{path} is missing; run `python -m sokolm fetch --model {key}`")
        if _sha256(path) != digest:
            raise SystemExit(f"{path} does not match the sha256 in {LOCK}")
    return model_dir(key)


def load_tokenizer(key: str = MAIN):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(verify(key, weights=False))
