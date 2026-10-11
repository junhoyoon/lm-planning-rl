"""TRL wiring: datasets in TRL's formats, the GRPO reward function, trainer construction, and the token
log-probabilities of SFT targets.

Precision. The trainers get the model with fp32 weights, and `bf16=True` runs the forward and backward passes under
bf16 autocast (accelerate's mixed precision). Weights, gradients and AdamW states stay fp32: stored in bf16, a weight
changes only when the update is at least about half its rounding step, and a first AdamW step moves each weight by
about the learning rate, which is far below that for most weights.

SFT (TRL SFTTrainer, prompt-completion format). prompt = the exact prompt string of the format (chat template applied,
ending with the generation prompt), completion = the answer line of the BFS solution. TRL appends the tokenizer's eos
token <|im_end|> to the completion, tokenizes prompt + completion, and puts only the completion tokens in the loss
(completion_only_loss): the answer line and <|im_end|>.

GRPO (TRL GRPOTrainer, prompt-only format). The prompt is the same string. TRL samples `num_generations` completions
per prompt with HF generate and cuts each one after its first end token (every eos id of the model's generation config:
<|im_end|> and <|endoftext|>). `reward_function` judges them (`formats.judge_completion`): 1 if solved within the
budget, else 0, and 0 for a completion without an end token within max_completion_length (truncated).

Decoding for GRPO rollouts is set explicitly (DECODING: the full distribution at temperature 1). The loss settings
(loss_type, beta, scale_rewards, epsilon, ...) stay at TRL's defaults and are recorded by the caller.
"""

import time
from typing import Optional

from sokolm.formats import COMPLETION_CAP, Format, answer_text, judge_completion, prompt_text
from sokolm.lm import END_TOKENS
from sokolm.sokoban import Board, parse_actions

DECODING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": None, "repetition_penalty": 1.0}


def sft_examples(tok, records: list[dict], fmt: Format = Format()) -> list[dict]:
    """Prompt-completion examples; the completion is the answer line of the stored BFS solution."""
    return [{"prompt": prompt_text(tok, r["text"], fmt), "completion": answer_text(parse_actions(r["solution"]), fmt),
             "board_id": r["board_id"]} for r in records]


def grpo_examples(tok, records: list[dict], fmt: Format = Format()) -> list[dict]:
    """Prompt-only examples; the other columns reach the reward function as keyword arguments."""
    return [{"prompt": prompt_text(tok, r["text"], fmt), "board_id": r["board_id"], "board_text": r["text"],
             "budget": r["budget"], "opt_len": r["opt_len"]} for r in records]


def reward_function(fmt: Format = Format(), end_ids: tuple[int, ...] = END_TOKENS, log: Optional[list] = None):
    """The binary reward as a TRL reward function. TRL passes the prompts, the decoded completions, their token ids
    (cut after the first end token) and the dataset columns. When `log` is a list, one record per completion is
    appended to it (with the optimizer step it was generated for)."""
    ends = set(end_ids)

    def sokoban_reward(prompts, completions, completion_ids, board_text, budget, opt_len, board_id,
                       trainer_state=None, **_):
        step = None if trainer_state is None else trainer_state.global_step + 1
        rewards = []
        for p, c, ids, text, b, o, bid in zip(prompts, completions, completion_ids, board_text, budget, opt_len,
                                              board_id, strict=True):
            truncated = not ids or ids[-1] not in ends
            judged = judge_completion(Board(text), c, fmt, b, o, truncated=truncated)
            rewards.append(float(judged["reward"]))
            if log is not None:
                log.append({"step": step, "board_id": bid, "prompt": p, "completion": c,
                            "completion_ids": list(ids), **judged})
        return rewards

    return sokoban_reward


def _common_args(output_dir, steps: int, seed: int, gradient_checkpointing: bool) -> dict:
    return dict(output_dir=str(output_dir), max_steps=steps, logging_steps=1, save_strategy="no", report_to="none",
                seed=seed, data_seed=seed, bf16=True, gradient_checkpointing=gradient_checkpointing, disable_tqdm=True,
                dataloader_num_workers=0)


SFT_SCHEDULES = {"constant": "constant_with_warmup", "linear": "linear"}


def sft_run_trainer(model, tok, examples: list[dict], output_dir, lr: float, steps: int, warmup: int, batch_size: int,
                    seed: int = 0, bf16: bool = True, gradient_checkpointing: bool = True, use_cpu: bool = False,
                    callbacks=None, schedule: str = "constant"):
    """SFTTrainer for an SFT training run. AdamW at `lr` after a linear warmup (schedule "constant",
    constant_with_warmup: optimizer step t = 1, 2, ... uses lr * min(1, (t - 1) / warmup), so step 1 uses 0 and step
    warmup + 1 the full lr). Schedule "linear" (transformers' linear schedule): the same warmup, then lr * (steps - t + 1) / (steps - warmup), down to lr / (steps - warmup) at the last step. TRL's other
    defaults (fused AdamW, weight decay 0, gradient clipping at norm 1). Every step takes `batch_size` examples, drawn
    without replacement in an order the Trainer's sampler fixes with `seed`. No evaluation dataset: the caller
    evaluates."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    args = SFTConfig(**(_common_args(output_dir, steps, seed, gradient_checkpointing) | {"bf16": bf16}),
                     use_cpu=use_cpu, learning_rate=lr, lr_scheduler_type=SFT_SCHEDULES[schedule],
                     warmup_steps=warmup, per_device_train_batch_size=batch_size, completion_only_loss=True)
    return SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(examples), processing_class=tok,
                      callbacks=callbacks)


def grpo_run_trainer(model, tok, examples: list[dict], reward_fn, output_dir, prompts_per_step: int,
                     num_generations: int, micro_batch: int, steps: int, lr: float, lr_schedule: str = "constant",
                     seed: int = 0, bf16: bool = True, gradient_checkpointing: bool = True, use_cpu: bool = False,
                     max_completion_length: int = COMPLETION_CAP, callbacks=None):
    """GRPOTrainer for a GRPO training run. Every optimizer step takes `prompts_per_step` boards of `examples`
    (drawn without replacement in an order TRL's RepeatSampler fixes with `seed`), generates `num_generations`
    completions for each in one generation batch, and computes the loss in micro-batches of `micro_batch` completions
    (gradient accumulation; with `dapo` every micro-batch is divided by the token count of the whole step).
    lr: `lr_schedule` ("constant", or TRL's default "linear" decay to 0 at `steps`), no warmup. Everything else stays
    at TRL's defaults (loss_type dapo, beta 0, scale_rewards group, epsilon 0.2, num_iterations 1, fused AdamW,
    weight decay 0, gradient clipping at norm 1) and is recorded by the caller."""
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    total = prompts_per_step * num_generations
    if total % micro_batch:
        raise ValueError(f"{total} completions per step do not split into micro-batches of {micro_batch}")
    args = GRPOConfig(**(_common_args(output_dir, steps, seed, gradient_checkpointing) | {"bf16": bf16}),
                      use_cpu=use_cpu, learning_rate=lr, lr_scheduler_type=lr_schedule,
                      per_device_train_batch_size=micro_batch, gradient_accumulation_steps=total // micro_batch,
                      num_generations=num_generations, max_completion_length=max_completion_length, **DECODING)
    return GRPOTrainer(model=model, reward_funcs=[reward_fn], args=args, train_dataset=Dataset.from_list(examples),
                       processing_class=tok, callbacks=callbacks)


def step_timer():
    """A TrainerCallback that records, for every optimizer step, its wall time and the peak GPU memory during it
    (allocated = live tensors, reserved = held by the caching allocator)."""
    import torch
    from transformers import TrainerCallback

    class StepTimer(TrainerCallback):
        def __init__(self):
            self.steps: list[dict] = []
            self._t0 = 0.0

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._t0 = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.steps.append({"step": state.global_step, "seconds": round(time.perf_counter() - self._t0, 3),
                               "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                               "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)})

    return StepTimer()


def token_logprobs(model, input_ids, attention_mask, T: int):
    """log p(y_t | x, y_<t) of the last T tokens of every row: (B, T) float32, with the gradient attached.

    input_ids (B, L): a prompt followed by T target slots; attention_mask (B, L): 1 on real tokens. The logits at
    position j score the token at j + 1, so the T slots are scored by the last T + 1 positions without the very last
    one; `logits_to_keep` computes only those. The log-softmax runs in float32 whatever the model's dtype."""
    import torch

    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=T + 1).logits  # (B, T+1, V)
    logits = logits[:, :-1].float()
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, input_ids[:, -T:].unsqueeze(-1)).squeeze(-1)


def sft_target_logprobs(model, tok, examples: list[dict]) -> dict:
    """Every SFT target token's log-probability under `model`, and the loss -(1/M) sum log p with M the number of
    target tokens in the batch.

    The target of an example is its completion (the answer line) followed by <|im_end|>, as TRL builds it. Rows are
    prompt + target, right-padded; every prompt has the same length P (a ValueError otherwise; true for the val boards
    of each bucket), so the targets sit at positions P .. P + T - 1.
    Returns, on the CPU: prompt_ids, target_ids (lists), logp (B, T) float32, mask (B, T) bool, M, loss."""
    import torch

    prompt_ids = [tok(e["prompt"], add_special_tokens=False)["input_ids"] for e in examples]
    target_ids = [tok(e["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id] for e in examples]
    P = len(prompt_ids[0])
    if any(len(p) != P for p in prompt_ids):
        raise ValueError("the prompts differ in length")
    T = max(map(len, target_ids))
    B = len(examples)
    ids = torch.full((B, P + T), tok.pad_token_id)
    am = torch.zeros((B, P + T), dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.bool)
    for b, (p, t) in enumerate(zip(prompt_ids, target_ids)):
        ids[b, :P + len(t)] = torch.tensor(p + t)
        am[b, :P + len(t)] = 1
        mask[b, :len(t)] = True
    ids, am = ids.to(model.device), am.to(model.device)
    with torch.no_grad():
        logp = token_logprobs(model, ids, am, T).float()
    m = mask.to(logp.device)
    M = int(m.sum())
    loss = float(-(logp * m).sum() / M)
    return {"prompt_ids": prompt_ids, "target_ids": target_ids, "logp": logp.cpu(), "mask": mask, "M": M,
            "loss": loss}
