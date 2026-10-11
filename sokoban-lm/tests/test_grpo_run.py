"""The GRPO run's pieces on the CPU: the zero-std prediction against enumeration, the group statistics of rollout
records, and TRL's GRPOTrainer as `train.grpo_run_trainer` sets it up (boards per step, groups, constant lr, TRL's own
zero-std share against ours). A tiny random Qwen2 with the real vocabulary; no GPU needed."""

import itertools
from math import comb

import pytest
import torch

from sokolm.assets import load_tokenizer
from sokolm.evaluate import group_rows, group_summary, mixed_at, zero_std_prediction
from sokolm.formats import Format
from sokolm.lm import END_TOKENS
from sokolm.splits import load_part
from sokolm.train import grpo_examples, grpo_run_trainer, reward_function


def _model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg)
    m.generation_config.eos_token_id = list(END_TOKENS)
    return m


@pytest.mark.parametrize("n", [4, 6])
def test_mixed_at_counts_the_subsets(n):
    """Every g-subset of n answers (c solving) counted: the share that holds both a solving and a failing answer."""
    for c in range(n + 1):
        solved = [1] * c + [0] * (n - c)
        for g in range(1, n + 1):
            subsets = list(itertools.combinations(range(n), g))
            both = sum(0 < sum(solved[i] for i in s) < g for s in subsets)
            assert mixed_at(n, c, g) == pytest.approx(both / len(subsets)), (c, g)


def test_zero_std_prediction_by_hand():
    """Length 1: one board solved 16/16, one 0/16 -> every group of 8 is all-solved or all-failed. Length 2: one board
    8/16 -> a group of 8 is all-solved with probability 1 / C(16, 8), all-failed likewise."""
    recs = [{"board_id": b, "opt_len": L, "reward": float(k < c)}
            for b, L, c in (("a", 1, 16), ("b", 1, 0), ("c", 2, 8)) for k in range(16)]
    p = zero_std_prediction(recs, {1: 0.25, 2: 0.75}, g=8)
    one, two = p["by_length"]
    assert (one["all_solved"], one["all_failed"], one["zero_std"]) == (0.5, 0.5, 1.0)
    assert two["all_solved"] == two["all_failed"] == round(1 / comb(16, 8), 4)
    assert p["zero_std"] == pytest.approx(0.25 * 1.0 + 0.75 * 2 / comb(16, 8), abs=1e-4)
    assert p["n"] == 16 and p["g"] == 8
    with pytest.raises(ValueError):
        zero_std_prediction(recs, {1: 0.25, 2: 0.75}, g=17)  # 16 samples cannot form a group of 17


def test_group_rows_and_summary():
    fmt = Format()
    recs = []
    for step, bid, L, texts, rewards in ((1, "a", 1, ["Answer: up"] * 4, [1, 1, 1, 1]),
                                         (1, "b", 9, ["Answer: up left", "Answer: up down", "Answer: left", "x y"],
                                          [1, 0, 0, 0]),
                                         (2, "a", 1, ["Answer: down"] * 4, [0, 0, 0, 0])):
        recs += [{"step": step, "board_id": bid, "opt_len": L, "completion": t, "reward": float(r)}
                 for t, r in zip(texts, rewards)]
    rows = group_rows(recs, fmt)
    assert [(r["step"], r["board_id"], r["solved"], r["zero_std"]) for r in rows] == [
        (1, "a", 4, True), (1, "b", 1, False), (2, "a", 0, True)]
    assert [(r["distinct_texts"], r["distinct_first"]) for r in rows] == [(1, 1), (4, 3), (1, 1)]  # up, left, None
    s = group_summary(rows)
    assert (s["groups"], s["zero_std"], s["all_solved"], s["all_failed"]) == (3, round(2 / 3, 4), round(1 / 3, 4),
                                                                              round(1 / 3, 4))
    assert (s["zero_std_1_4"], s["groups_1_4"], s["zero_std_9_13"], s["zero_std_5_8"]) == (1.0, 2, 0.0, None)


@pytest.mark.split
@pytest.mark.tokenizer
def test_grpo_run_trainer_steps_on_cpu(tmp_path):
    """2 steps of 2 boards x 4 completions in micro-batches of 4: each step sees 2 new boards, 4 completions each;
    the lr stays constant; TRL's logged reward and zero-std share equal ours from the same rollouts."""
    tok = load_tokenizer()
    fmt = Format()
    recs = load_part("7x7-v1", "rl")[:6]
    log: list[dict] = []
    tr = grpo_run_trainer(_model(), tok, grpo_examples(tok, recs, fmt), reward_function(fmt, log=log), tmp_path,
                          prompts_per_step=2, num_generations=4, micro_batch=4, steps=2, lr=1e-3, bf16=False,
                          gradient_checkpointing=False, use_cpu=True, max_completion_length=8)
    a = tr.args
    assert (a.loss_type, a.beta, a.scale_rewards, a.num_iterations, a.epsilon) == ("dapo", 0.0, "group", 1, 0.2)
    assert (a.generation_batch_size, a.steps_per_generation, a.gradient_accumulation_steps) == (8, 2, 2)
    tr.train()
    by_step: dict[int, list[dict]] = {}
    for r in log:
        by_step.setdefault(r["step"], []).append(r)
    assert sorted(by_step) == [1, 2]
    boards = [sorted({r["board_id"] for r in by_step[s]}) for s in (1, 2)]
    assert all(len(b) == 2 for b in boards) and not set(boards[0]) & set(boards[1])
    assert all(sum(r["board_id"] == b for r in by_step[s]) == 4 for s in (1, 2) for b in boards[s - 1])
    logs = [h for h in tr.state.log_history if "loss" in h]
    assert [h["learning_rate"] for h in logs] == pytest.approx([1e-3, 1e-3])
    for s, h in zip((1, 2), logs):
        ours = group_summary(group_rows(by_step[s], fmt))
        assert h["frac_reward_zero_std"] == pytest.approx(ours["zero_std"])
        assert h["reward"] == pytest.approx(sum(r["reward"] for r in by_step[s]) / 8)
