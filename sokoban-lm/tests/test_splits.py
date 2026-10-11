"""The generator, the problem identity used for splits, and the split builder."""

import json

import pytest

from sokolm import splits
from sokolm.generate import generate
from sokolm.splits import SPLITS, build, layout_class, quotas, split_of, verify
from sokolm.symmetry import variants

BOARD = """\
######
# @. #
#  $ #
#  ###
#    #
######"""


def test_generate_stops_at_n_or_max_samples():
    gens, stats = generate(5, 7, 0)
    assert len(gens) == 5 and stats["samples"] == gens[-1].index + 1
    gens, stats = generate(5, 7, 0, max_samples=3)
    assert stats["samples"] == 3 and len(gens) <= 3
    with pytest.raises(ValueError):
        generate(1, 7, 0, wall_sampling="uniform")


def test_layout_class_is_the_problem_identity():
    cls = layout_class(BOARD)
    assert all(layout_class(v) == cls for v in variants(BOARD))  # rotations and reflections
    moved = BOARD.replace("@", " ").replace("$", " ")
    moved = moved[:moved.index("#    #")] + "#@ $ #" + moved[moved.index("#    #") + 6:]  # box and player elsewhere
    assert moved != BOARD and layout_class(moved) == cls
    other_target = BOARD.replace(".", " ").replace("#    #", "#.   #")
    assert layout_class(other_target) != cls


def test_split_of_is_a_deterministic_hash_with_the_configured_shares():
    cfg = SPLITS["7x7-v1"]
    names = [split_of(f"class-{i}", cfg) for i in range(20000)]
    assert names == [split_of(f"class-{i}", cfg) for i in range(20000)]
    for split, share in (("val", 0.1), ("test", 0.1), ("train", 0.8)):
        assert abs(names.count(split) / len(names) - share) < 0.01


@pytest.fixture(scope="module")
def tiny():
    cfg = {**SPLITS["7x7-v1"], "quota": {"val": 4, "test": 6}, "train_pool": 40}
    return cfg, build(cfg, "tiny")


def test_tiny_build_fills_quotas_without_shared_classes(tiny):
    cfg, (records, streams) = tiny
    checks = verify(records, cfg)  # raises on a violated rule
    assert checks["records"] == sum(quotas(cfg).values()) == checks["distinct_layout_classes"]
    by_split = {s: {r["layout_class"] for r in rs} for s, rs in records.items()}
    assert not (by_split["train"] & by_split["val"] or by_split["train"] & by_split["test"]
                or by_split["val"] & by_split["test"])
    assert streams["one_box"]["accepted"] == 40 + 2 * (4 + 6)
    assert streams["two_box"]["accepted"] == 4 + 6


def test_tiny_build_is_deterministic(tiny):
    cfg, (records, _) = tiny
    assert build(cfg, "tiny")[0] == records


def test_write_and_load_check_the_manifest(tiny, tmp_path, monkeypatch):
    cfg, (records, _) = tiny
    monkeypatch.setattr(splits, "DATA", tmp_path / "data")
    monkeypatch.setattr(splits, "MANIFESTS", tmp_path / "splits")
    splits.write("tiny", cfg, records, {})
    assert splits.load("tiny", "val") == records["val"]
    assert splits.load("tiny", "test", "long") == [r for r in records["test"] if r["bucket"] == "long"]
    with pytest.raises(SystemExit):
        splits.write("tiny", cfg, records, {})  # a fixed split is not silently rebuilt
    path = tmp_path / "data" / "splits" / "tiny" / "val.jsonl"
    rows = path.read_text().splitlines()
    rows[0] = json.dumps({**json.loads(rows[0]), "opt_len": 1})
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError):
        splits.load("tiny", "val")
