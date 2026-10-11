"""Model loading and batched generation.

Every decoding setting is passed explicitly. transformers fills each field left unset from the model's
generation_config.json, which for Qwen2.5-0.5B-Instruct samples with temperature 0.7, top_p 0.8 and top_k 20 and
applies repetition_penalty 1.1, in greedy decoding too; fields unset there fall back to library defaults (top_k 50).
  greedy  do_sample=False, repetition_penalty 1: the argmax token at every step
  sample  do_sample=True, temperature T, top_k 0 (off), top_p 1 (off), repetition_penalty 1: at T = 1 every token is
          drawn from the model's own next-token distribution

A completion ends at its first end token: <|im_end|> closes the assistant turn of the chat template, and
<|endoftext|> is the second end id in generation_config.json (it is also the pad token). A completion without an end
token within max_new_tokens is truncated.
"""

import json
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, GenerationConfig

from sokolm.assets import MAIN, model_dir, verify
from sokolm.formats import COMPLETION_CAP

END_TOKENS = (151645, 151643)  # <|im_end|>, <|endoftext|> of the main model


def end_tokens(tokenizer, key: str = MAIN) -> tuple[int, ...]:
    """Ids that end a completion: the eos ids of the model's generation_config.json and config.json (and its
    text_config), then the tokenizer's eos token, without repeats. For the main model this gives END_TOKENS."""
    ids = []
    for name in ("generation_config.json", "config.json"):
        path = model_dir(key) / name
        cfg = json.loads(path.read_text()) if path.exists() else {}
        for c in (cfg, cfg.get("text_config") or {}):
            e = c.get("eos_token_id")
            ids += e if isinstance(e, list) else [] if e is None else [e]
    ids.append(tokenizer.eos_token_id)
    return tuple(dict.fromkeys(ids))


def load_model(key: str = MAIN, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
    model = AutoModelForCausalLM.from_pretrained(verify(key), dtype=dtype)
    return model.to(device).eval()


def generation_config(tokenizer, mode: str, temperature: float = 1.0, max_new_tokens: int = COMPLETION_CAP,
                      end_ids: tuple[int, ...] = END_TOKENS) -> GenerationConfig:
    """Greedy or sampling config with every field that the hub config would otherwise fill set explicitly."""
    if mode not in ("greedy", "sample"):
        raise ValueError(f"mode must be greedy or sample, got {mode!r}")
    return GenerationConfig(
        do_sample=mode == "sample", temperature=temperature if mode == "sample" else 1.0, top_k=0, top_p=1.0,
        repetition_penalty=1.0, num_beams=1, max_new_tokens=max_new_tokens, eos_token_id=list(end_ids),
        pad_token_id=tokenizer.pad_token_id)


@dataclass
class Completion:
    ids: list[int]  # generated tokens up to and including the first end token (all of them when truncated)
    text: str  # decoded without special tokens
    truncated: bool  # max_new_tokens reached without an end token
    end_token: Optional[int]  # the end token that stopped it, None when truncated


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], cfg: GenerationConfig, n: int = 1, batch_size: int = 64,
             seed: Optional[int] = None) -> list[list[Completion]]:
    """`n` completions for each prompt (prompt order kept; copies of a prompt are adjacent within the batches)."""
    tokenizer.padding_side = "left"  # all prompts then end at the same position, where generation starts
    if seed is not None:
        torch.manual_seed(seed)
    ends = set(cfg.eos_token_id)
    jobs = [i for i in range(len(prompts)) for _ in range(n)]
    out: list[list[Completion]] = [[] for _ in prompts]
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        enc = tokenizer([prompts[i] for i in chunk], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        seq = model.generate(**enc, generation_config=cfg)
        for i, g in zip(chunk, seq[:, enc["input_ids"].shape[1]:].tolist()):
            cut = next((j for j, t in enumerate(g) if t in ends), None)
            ids = g if cut is None else g[:cut + 1]
            out[i].append(Completion(ids, tokenizer.decode(ids, skip_special_tokens=True), cut is None,
                                     None if cut is None else g[cut]))
    return out
