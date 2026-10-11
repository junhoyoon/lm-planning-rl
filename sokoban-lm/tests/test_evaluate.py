"""sokolm.evaluate (pass@k, summaries, board-blind baselines) and the SFT/RL parts of the train pool; CPU only."""

import hashlib
import itertools
from collections import Counter

import pytest
import torch

from sokolm import evaluate
from sokolm.formats import Format, answer_text, judge_completion
from sokolm.simulate import simulate
from sokolm.sokoban import Board
from sokolm.splits import PART_NAMES, TRAIN_PARTS, load, load_part, train_parts

FMT = Format()
WORDS = ("up", "down", "left", "right")


@pytest.fixture(scope="module")
def one_move():
    """1-move train boards with the move that solves each (its BFS solution)."""
    recs = [r for r in load("7x7-v1", "train") if r["opt_len"] == 1][:40]
    return [(r, "UDLR".index(r["solution"])) for r in recs]


def _rec(b: dict, completion: str, mode: str = "greedy", sample: int = 0) -> dict:
    j = judge_completion(Board(b["text"]), completion, FMT, b["budget"], b["opt_len"])
    return {"board_id": b["board_id"], "bucket": b["bucket"], "mode": mode, "sample": sample,
            "completion": completion, "n_tokens": 4, "end_token": 151645, **j}


def test_pass_at_k():
    assert evaluate.pass_at_k(16, 0, 4) == 0.0 and evaluate.pass_at_k(16, 16, 1) == 1.0
    assert evaluate.pass_at_k(16, 3, 1) == pytest.approx(3 / 16)
    assert evaluate.pass_at_k(4, 1, 2) == pytest.approx(0.5)  # 1 - C(3,2)/C(4,2)
    assert evaluate.pass_at_k(4, 3, 2) == 1.0  # fewer than k failures: any k answers include a solving one
    with pytest.raises(ValueError):
        evaluate.pass_at_k(4, 1, 8)


@pytest.mark.split
def test_summary_rows(one_move):
    (b0, m0), (b1, m1) = one_move[:2]
    recs = [_rec(b0, answer_text([m0], FMT)), _rec(b1, "Answer: left up up right")]
    recs += [_rec(b0, answer_text([m0], FMT), "sample", k) for k in range(2)]
    recs += [_rec(b1, answer_text([m1], FMT), "sample", 0), _rec(b1, "no answer", "sample", 1)]
    g, s = evaluate.summary_rows(recs, FMT, ks=(1, 2, 4))
    solved_example = float(judge_completion(Board(b1["text"]), "Answer: left up up right", FMT, 2, 1)["reward"])
    assert g["mode"] == "greedy" and g["solve_rate"] == pytest.approx((1 + solved_example) / 2)
    assert g["example_share"] == 0.5 and g["distinct_texts"] == 2
    assert s["pass@1"] == pytest.approx((1 + 0.5) / 2) and s["pass@2"] == 1.0 and "pass@4" not in s
    assert s["mixed"] == 0.5 and s["parse_fail"] == 0.25  # board 1's samples: one solves, one fails
    rows = evaluate.summary_rows(recs, FMT, ks=(1,), by_length=True)
    assert [r["opt_len"] for r in rows] == [1, 1]


@pytest.mark.split
def test_swap_baseline(one_move):
    a = next((b, m) for b, m in one_move if m == 0)  # solved by up only
    c = next((b, m) for b, m in one_move  # solved by down, and up does not solve it
             if m == 1 and not simulate(Board(b["text"]), [0], 2).solved
             and not simulate(Board(a[0]["text"]), [1], 2).solved)
    boards = [a[0], c[0]]
    own = [_rec(b, answer_text([m], FMT)) for b, m in (a, c)]
    row, = evaluate.swap_baseline(boards, own, FMT)
    assert row["own"] == 1.0 and row["swap"] == 0.0  # each answer fits its own board only
    same = [_rec(b, "Answer: up down") for b in boards]
    row, = evaluate.swap_baseline(boards, same, FMT)
    assert row["swap"] == row["own"] == 1.0  # one answer for both boards: swapping changes nothing


@pytest.mark.split
def test_fixed_baseline_matches_brute_force(one_move):
    boards = [b for b, _ in one_move]
    row, = evaluate.fixed_baseline(boards, max_len=2)
    best = max((sum(simulate(Board(b["text"]), q, b["budget"]).solved for b in boards) / len(boards), -len(q))
               for m in (1, 2) for q in itertools.product(range(4), repeat=m))
    assert row["rate"] == pytest.approx(best[0], abs=1e-4) and len(row["answer"].split()) == -best[1]


@pytest.mark.split
@pytest.mark.tokenizer
def test_answer_records_on_a_tiny_model(one_move):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from sokolm.assets import load_tokenizer
    from sokolm.lm import generation_config

    cfg = Qwen2Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).eval()
    tok = load_tokenizer()
    boards = [b for b, _ in one_move[:2]]
    gen = generation_config(tok, "sample", max_new_tokens=6)
    recs = evaluate.answer_records(model, tok, boards, FMT, gen, "sample", n=3, seed=0, batch_size=4)
    assert [(r["board_id"], r["sample"]) for r in recs] == [(b["board_id"], k) for b in boards for k in range(3)]
    assert all(len(r["ids"]) == r["n_tokens"] <= 6 and r["reward"] in (0.0, 1.0) for r in recs)
    assert all(r["truncated"] == (r["end_token"] is None) for r in recs)


def test_train_parts_are_halves_by_length():
    recs = [{"board_id": f"b{i}", "opt_len": 1 + i % 3} for i in range(31)]
    parts = train_parts(recs, "salt")
    ids = [{r["board_id"] for r in parts[p]} for p in PART_NAMES]
    assert not ids[0] & ids[1] and len(ids[0] | ids[1]) == 31
    for L in (1, 2, 3):
        n = [sum(r["opt_len"] == L for r in parts[p]) for p in PART_NAMES]
        assert abs(n[0] - n[1]) <= 1
    again = train_parts(list(reversed(recs)), "salt")  # the order of the input does not change the parts
    assert [{r["board_id"] for r in again[p]} for p in PART_NAMES] == ids
    assert [r["board_id"] for r in parts["sft"]] == sorted((r["board_id"] for r in parts["sft"]),
                                                            key=lambda x: int(x[1:]))  # train file order kept


@pytest.mark.split
def test_train_parts_of_7x7_v1():
    sft, rl = load_part("7x7-v1", "sft"), load_part("7x7-v1", "rl")
    assert len(sft) == len(rl) == 25000 and not {r["board_id"] for r in sft} & {r["board_id"] for r in rl}
    cs, cr = Counter(r["opt_len"] for r in sft), Counter(r["opt_len"] for r in rl)
    assert set(cs) == set(cr) == set(range(1, 14)) and all(abs(cs[L] - cr[L]) <= 1 for L in cs)
    digest = hashlib.sha256("\n".join(r["board_id"] for r in sft).encode()).hexdigest()
    assert TRAIN_PARTS["7x7-v1"] == "sokolm-train-parts-v1"
    assert digest == "cdda5f4bfc3eecb616ea09cbd2ef0c28c3339bae3da4d5ac6beab85499b0c1a0"  # pinned on 2026-10-09
