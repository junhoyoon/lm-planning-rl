"""The SFT run's pieces on the CPU: TRL's lr schedules as they run, the val loss against a one-batch computation, the
bf16 evaluation copy, the kept best weights. A tiny random Qwen2 with the real vocabulary; no GPU needed."""

import pytest
import torch

from sokolm.assets import load_tokenizer
from sokolm.pipeline import _copy_weights, _sft_val_loss
from sokolm.formats import Format, prompt_text
from sokolm.lm import END_TOKENS
from sokolm.splits import load
from sokolm.train import sft_examples, sft_run_trainer, sft_target_logprobs


def _model(dtype=torch.float32):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg).to(dtype)
    m.generation_config.eos_token_id = list(END_TOKENS)
    return m


@pytest.fixture(scope="module")
def examples():
    tok = load_tokenizer()
    recs = [r for r in load("7x7-v1", "train") if r["opt_len"] in (1, 4, 9)][:6]
    return tok, sft_examples(tok, recs, Format())


@pytest.mark.split
@pytest.mark.tokenizer
def test_warmup_schedule_as_trained(examples, tmp_path):
    """constant_with_warmup, warmup 2: steps 1, 2, 3 use lr x 0, 1/2, 1; step 1 leaves the weights unchanged."""
    from transformers import TrainerCallback

    tok, ex = examples
    model = _model()
    w0 = model.model.embed_tokens.weight.detach().clone()
    changed = []

    class Watch(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            changed.append(not torch.equal(model.model.embed_tokens.weight, w0))

    tr = sft_run_trainer(model, tok, ex, tmp_path, lr=1e-3, steps=3, warmup=2, batch_size=2, bf16=False,
                         gradient_checkpointing=False, use_cpu=True, callbacks=[Watch()])
    tr.train()
    logs = [h for h in tr.state.log_history if "loss" in h]
    assert [h["learning_rate"] for h in logs] == pytest.approx([0.0, 5e-4, 1e-3])
    assert changed == [False, True, True]
    assert tr.args.weight_decay == 0.0 and tr.args.max_grad_norm == 1.0


@pytest.mark.split
@pytest.mark.tokenizer
def test_linear_schedule_as_trained(examples, tmp_path):
    """linear, warmup 2, 4 steps: steps 1-4 use lr x 0, 1/2, 1, 1/2 (down to lr / (steps - warmup) at the last)."""
    tok, ex = examples
    tr = sft_run_trainer(_model(), tok, ex, tmp_path, lr=1e-3, steps=4, warmup=2, batch_size=2, bf16=False,
                         gradient_checkpointing=False, use_cpu=True, schedule="linear")
    tr.train()
    logs = [h for h in tr.state.log_history if "loss" in h]
    assert [h["learning_rate"] for h in logs] == pytest.approx([0.0, 5e-4, 1e-3, 5e-4])


@pytest.mark.split
@pytest.mark.tokenizer
def test_val_loss_matches_one_batch_reference(examples):
    """Over several batches, the loss is still -(1/M) sum log p over all target tokens at once."""
    tok, ex = examples
    model = _model().eval()
    ref = sft_target_logprobs(model, tok, ex)
    v = _sft_val_loss(model, tok, ex, batch_size=4, autocast=False)  # batches of 4 and 2
    assert v["M"] == ref["M"] and v["loss"] == pytest.approx(ref["loss"], abs=1e-4)
    assert sum(v["share"].values()) == pytest.approx(1.0, abs=1e-3)
    n = [len(t) for t in ref["target_ids"]]
    end_nll = -sum(float(ref["logp"][b, k - 1]) for b, k in enumerate(n)) / len(n)
    assert v["nll_mean"]["end"] == pytest.approx(end_nll, abs=1e-4)


@pytest.mark.split
@pytest.mark.tokenizer
@pytest.mark.parametrize("bucket", ["in_dist", "long", "two_box"])
def test_val_prompts_have_one_length(bucket):
    """sft_target_logprobs needs one prompt length per batch: true for the 300 val boards of each bucket (sft scores
    in_dist, val-loss all three)."""
    tok = load_tokenizer()
    val = [r for r in load("7x7-v1", "val") if r["bucket"] == bucket][:300]
    lengths = {len(tok(prompt_text(tok, r["text"], Format()), add_special_tokens=False)["input_ids"]) for r in val}
    assert len(lengths) == 1


def test_copy_weights_casts_to_bf16():
    src = _model()
    with torch.no_grad():
        for p in src.parameters():
            p.add_(1e-3 * torch.randn_like(p))  # values that bf16 has to round
    dst = _model(torch.bfloat16)
    _copy_weights(src, dst)
    for (k, a), (_, b) in zip(src.named_parameters(), dst.named_parameters()):
        assert b.dtype == torch.bfloat16 and torch.equal(b, a.to(torch.bfloat16)), k


def test_best_weights_keep_the_first_best(tmp_path):
    """Rates 0.1, 0.3, 0.3, 0.2 at steps 0..3: the weights of step 1 are kept (a tie keeps the earlier) and saved."""
    from transformers import AutoModelForCausalLM

    from sokolm.pipeline import _BestWeights

    model = _model(torch.bfloat16)
    best = _BestWeights(model)
    snapshots = []
    for step, rate in enumerate((0.1, 0.3, 0.3, 0.2)):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.01)
        snapshots.append({k: p.detach().clone() for k, p in model.named_parameters()})
        best.offer(step, rate, model)
    assert best.step == 1 and best.rate == 0.3
    best.save(model, tmp_path / "best")
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path / "best", dtype=torch.bfloat16)
    for k, p in loaded.named_parameters():
        assert torch.equal(p, snapshots[1][k]), k
