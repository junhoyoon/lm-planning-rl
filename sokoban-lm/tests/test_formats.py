"""Prompt rendering, the strict answer parser, and (when the pinned tokenizer files are present) the token facts the
format relies on."""

import random

import pytest

from sokolm.formats import (BOARD_STYLES, VOCABS, Format, answer_text, judge_completion, parse_answer, prompt_text,
                            render_board)
from sokolm.sokoban import Board, parse_actions

BOARD = """\
#######
#.   ##
#  #$ #
#  # ##
##    #
##  @ #
#######"""


def test_spaced_board_has_one_symbol_per_cell():
    rows = render_board(BOARD, "spaced").split("\n")
    assert all(len(r) == 13 and r[1::2] == " " * 6 for r in rows)
    assert "".join(r[::2] for r in rows) == BOARD.replace("\n", "").replace(" ", "_")
    assert render_board(BOARD, "xsb") == BOARD


@pytest.mark.parametrize("vocab", list(VOCABS))
def test_answer_roundtrip(vocab):
    fmt, rng = Format("spaced", vocab), random.Random(0)
    for n in (1, 2, 17, 92):
        actions = [rng.randrange(4) for _ in range(n)]
        assert parse_answer(answer_text(actions, fmt), fmt) == (actions, None)
        assert parse_answer("\n " + answer_text(actions, fmt) + " \n", fmt) == (actions, None)


@pytest.mark.parametrize("text, error", [
    ("", "empty"),
    ("  \n ", "empty"),
    ("Let me think.\nAnswer: up", "not_one_line"),
    ("up left", "no_prefix"),
    ("answer: up", "no_prefix"),
    ("Answer:", "no_actions"),
    ("Answer:   ", "no_actions"),
    ("Answer:up", "bad_item"),
    ("Answer: Up", "bad_item"),
    ("Answer: up.", "bad_item"),
    ("Answer: up,left", "bad_item"),
    ("Answer: up  left", "bad_item"),
    ("Answer: U L", "bad_item"),
])
def test_parse_failures(text, error):
    assert parse_answer(text, Format("spaced", "words")) == (None, error)


def test_judge_completion_rules():
    fmt, board = Format("spaced", "words"), Board("#####\n#@$.#\n#####")
    ok = judge_completion(board, "Answer: right", fmt, budget=2, opt_len=1)
    assert ok["reward"] == 1 and ok["parse_ok"] and ok["parse_error"] is None
    extra = judge_completion(board, "Answer: right left left", fmt, budget=2, opt_len=1)
    assert extra["reward"] == 1 and extra["n_after_solve"] == 2
    bad = judge_completion(board, "Answer: Right", fmt, budget=2, opt_len=1)
    assert bad["reward"] == 0 and not bad["parse_ok"] and bad["parse_error"] == "bad_item"
    cut = judge_completion(board, "Answer: right left", fmt, budget=2, opt_len=1, truncated=True)
    assert cut["reward"] == 0 and cut["truncated"] and cut["solved"]  # solved, but the output never ended


@pytest.fixture(scope="module")
def tok():
    from sokolm.assets import load_tokenizer

    try:
        return load_tokenizer()
    except SystemExit:
        pytest.skip("pinned tokenizer files not fetched (python -m sokolm fetch)")


def test_every_action_is_one_token_after_the_prefix(tok):
    for vocab, words in VOCABS.items():
        fmt = Format("spaced", vocab)
        ids = tok(answer_text([0, 1, 2, 3, 3, 2, 1, 0], fmt), add_special_tokens=False)["input_ids"]
        assert len(ids) == 2 + 8 and ids[2:6] == ids[6:][::-1]  # same token for the same action anywhere
        assert [tok.decode([i]) for i in ids[2:6]] == [" " + w for w in words]


def test_spaced_board_tokens_are_cells(tok):
    for style, n_tokens in (("spaced", 49), ("xsb", 20)):
        ids = tok(render_board(BOARD, style), add_special_tokens=False)["input_ids"]
        assert len(ids) == n_tokens


def test_prompt_uses_the_default_system_prompt_and_ends_with_the_generation_prompt(tok):
    for style in BOARD_STYLES:
        p = prompt_text(tok, BOARD, Format(style, "words"))
        assert p.startswith("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.")
        assert p.endswith("<|im_end|>\n<|im_start|>assistant\n")
        assert render_board(BOARD, style) in p


def test_parse_actions_matches_answer_text():
    fmt = Format("spaced", "letters")
    assert parse_answer(answer_text(parse_actions("RRDLULLDDRULUR"), fmt), fmt)[0] == parse_actions("RRDLULLDDRULUR")


def test_generation_configs_leave_nothing_for_the_hub_defaults(tok):
    pytest.importorskip("torch")
    from sokolm.lm import END_TOKENS, end_tokens, generation_config

    assert end_tokens(tok) == END_TOKENS  # read from the pinned config files
    for mode in ("greedy", "sample"):
        cfg = generation_config(tok, mode, temperature=1.0)
        # the hub generation_config.json would otherwise supply temperature 0.7, top_p 0.8, top_k 20, penalty 1.1
        assert (cfg.temperature, cfg.top_k, cfg.top_p, cfg.repetition_penalty) == (1.0, 0, 1.0, 1.0)
        assert cfg.do_sample is (mode == "sample") and list(cfg.eos_token_id) == list(END_TOKENS)
        assert cfg.max_new_tokens == 128
