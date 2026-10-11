"""The GRPO reward function (TRL's calling convention) and the SFT/GRPO example builders of sokolm.train."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from sokolm.formats import Format  # noqa: E402
from sokolm.lm import END_TOKENS  # noqa: E402
from sokolm.train import grpo_examples, reward_function, sft_examples  # noqa: E402

BOARD = "#####\n#@$.#\n#####"  # one move: right


def test_reward_follows_the_judging_rule():
    log = []
    fn = reward_function(Format(), END_TOKENS, log)
    texts = ["Answer: right", "Answer: right", "Answer: Right", "Answer: left"]
    ids = [[1, 2, END_TOKENS[0]], [1, 2, 3], [1, 2, END_TOKENS[1]], [1, 2, END_TOKENS[0]]]  # the second has no end
    n = len(texts)
    rewards = fn(prompts=["p"] * n, completions=texts, completion_ids=ids, board_text=[BOARD] * n, budget=[2] * n,
                 opt_len=[1] * n, board_id=["b"] * n, trainer_state=SimpleNamespace(global_step=0), log_metric=None)
    assert rewards == [1.0, 0.0, 0.0, 0.0]  # solved / truncated / parse failure / not solved
    assert [r["truncated"] for r in log] == [False, True, False, False]
    assert log[2]["parse_error"] == "bad_item" and all(r["step"] == 1 for r in log)


@pytest.fixture(scope="module")
def tok():
    from sokolm.assets import load_tokenizer

    try:
        return load_tokenizer()
    except SystemExit:
        pytest.skip("pinned tokenizer files not fetched (python -m sokolm fetch)")


def test_examples(tok):
    rec = {"board_id": "b", "text": BOARD, "solution": "R", "budget": 2, "opt_len": 1}
    (s,), (g,) = sft_examples(tok, [rec]), grpo_examples(tok, [rec])
    assert s["completion"] == "Answer: right" and s["prompt"] == g["prompt"]
    assert s["prompt"].endswith("<|im_start|>assistant\n")
    assert (g["board_text"], g["budget"], g["opt_len"]) == (BOARD, 2, 1)
