"""Tests marked `split` or `tokenizer` read the 7x7-v1 split files or the pinned tokenizer files; they are skipped when
those files are absent (`python -m sokolm build-split`, `python -m sokolm fetch`)."""

import pytest

from sokolm.assets import model_dir
from sokolm.splits import SPLIT_NAMES, _paths

WHY = {"split": "split files not built (python -m sokolm build-split)",
       "tokenizer": "pinned tokenizer files not fetched (python -m sokolm fetch)"}


def pytest_configure(config):
    for name, why in WHY.items():
        config.addinivalue_line("markers", f"{name}: skipped when the {why.split(' (')[0]}")


def pytest_collection_modifyitems(config, items):
    data_dir, _ = _paths("7x7-v1")
    have = {"split": all((data_dir / f"{s}.jsonl").exists() for s in SPLIT_NAMES),
            "tokenizer": (model_dir() / "tokenizer.json").exists()}
    for item in items:
        for name in WHY:
            if item.get_closest_marker(name) and not have[name]:
                item.add_marker(pytest.mark.skip(reason=WHY[name]))
