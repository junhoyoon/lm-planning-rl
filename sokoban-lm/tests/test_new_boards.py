"""Boards beyond the split and evaluation on the boards of a file: the file reader, episode ids of a large bucket, and
(with the split files) generated boards that pass the checks and share no layout class with the split; CPU only."""

import json

import pytest

from sokolm import pipeline as pl
from sokolm.splits import load


def _write(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_file_boards(tmp_path):
    board = {"board_id": "b0", "text": "#####", "opt_len": 3, "budget": 6}
    _write(tmp_path / "ok.jsonl", [board, {**board, "board_id": "b1"}])
    recs = pl.file_boards(tmp_path / "ok.jsonl")
    assert [r["board_id"] for r in recs] == ["b0", "b1"] and {r["bucket"] for r in recs} == {pl.NEW_BUCKET}
    _write(tmp_path / "dup.jsonl", [board, board])
    with pytest.raises(ValueError, match="repeated"):
        pl.file_boards(tmp_path / "dup.jsonl")
    _write(tmp_path / "bad.jsonl", [{"board_id": "b0", "text": "#"}])
    with pytest.raises(ValueError, match="lacks"):
        pl.file_boards(tmp_path / "bad.jsonl")


class _Episodes:
    def __init__(self):
        self.recs = []

    def add(self, rec):
        self.recs.append(rec)


def test_episode_ids_fit_the_bucket():
    """Three digits up to 1,000 boards in a bucket (the val evaluations), four beyond (the 2,000 new boards)."""
    boards = [{"board_id": f"b{j}", "bucket": pl.NEW_BUCKET, "opt_len": 3} for j in range(1001)]
    w = _Episodes()
    pl._write_answers(w, [{"board_id": "b0", "mode": "greedy", "ids": []},
                          {"board_id": "b1000", "mode": "sample", "sample": 2, "ids": []}], boards)
    assert [r["episode_id"] for r in w.recs] == ["greedy-new-0000", "sample-new-1000-s02"]
    w = _Episodes()
    pl._write_answers(w, [{"board_id": "b299", "mode": "greedy", "ids": []}], boards[:300])
    assert w.recs[0]["episode_id"] == "greedy-new-299"


@pytest.mark.split
def test_new_boards_pass_the_checks():
    recs, stats = pl.new_boards(20)
    assert len(recs) == stats["accepted"] == 20
    assert all(r["board_id"] == f"7x7-v1-new-1b-{r['stream_index']:07d}" for r in recs)
    assert min(r["stream_index"] for r in recs) >= stats["first_index"]
    assert pl.check_new_boards(recs)["distinct_layout_classes"] == 20
    with pytest.raises(ValueError, match="already used"):
        pl.check_new_boards([recs[0], recs[0]])
    train = load("7x7-v1", "train")[0]
    with pytest.raises(ValueError, match="already used"):
        pl.check_new_boards([{**recs[0], "text": train["text"], "layout_class": train["layout_class"]}])
