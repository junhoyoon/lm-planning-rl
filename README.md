# lm-planning-rl

Training a small language model to solve planning puzzles with supervised fine-tuning (SFT) and reinforcement
learning (GRPO), on puzzles where a program can check every answer and find the optimal one.

## Results

### [SFT vs. GRPO on single-turn Sokoban](results/sft-vs-grpo/report.md)

Qwen2.5-0.5B-Instruct on 7×7 boards, generating a complete move sequence in one response. SFT targets are
BFS-optimal solutions.

- **The SFT schedule changed the conclusion.** GRPO raised the in-distribution greedy solve rate on validation boards
  from 87.0% to 93.7%, against 84.7% for continued SFT at a constant learning rate. Continued SFT with learning-rate
  decay also reached 93.7%, closing the initial gap.
- **After the decay, GRPO showed no advantage on new boards.** Against SFT with learning-rate decay, GRPO's sampling
  pass@1 differed by 0.0 and +2.6 percentage points on the 300 validation boards, depending on the starting
  checkpoint, but by −2.3 and +0.3 points on 2,000 new boards that no run trained on and no choice was based on. On
  the new boards, SFT with the decay had the higher greedy solve rate and pass@16 from both starting checkpoints. On
  solution lengths beyond the training range (validation boards only), GRPO showed no clear advantage.
- **Compute:** GRPO took approximately 14× as long per training step as SFT in this setup.
- **Evaluation scope:** One seed per run. Validation boards, and 2,000 new in-distribution boards evaluated after the
  validation results as a check; the test split is untouched.

![Animation: 300 validation boards as squares in two rows, colored by how many of 16 sampled answers solve each board, from the base model through SFT, then GRPO from the SFT model (top) and from SFT with the learning rate decayed to 0 (bottom)](results/sft-vs-grpo/board-solves.svg)

Each square is one in-distribution validation board, colored by how many of 16 sampled answers solve it. Blue squares
are boards that all 16 samples solve: one more SFT epoch on the same training boards with the learning rate decayed to
0 (bottom row, before GRPO) raised their number from 174 to 231, one GRPO epoch from the same SFT model (top row) to
245, and GRPO after the decay to 258. Only groups of answers with different rewards give GRPO a gradient
([report §5.3](results/sft-vs-grpo/report.md#53-mixed-reward-groups-during-grpo-training)), and such groups come from
the orange squares, boards solved sometimes but not every time: 112 when GRPO started from the SFT model and 58 when it
started after the decay. The counts come from the base model's evaluation run and from the in-training evaluations of
runs 1 and 6, every 100 GRPO steps; the report's tables use separate evaluations of the final checkpoints. Data and
script: [`board-solves.json`](results/sft-vs-grpo/board-solves.json),
[`board_solves.py`](results/sft-vs-grpo/board_solves.py).

## Layout

| path | contents |
|---|---|
| [`sokoban-lm/`](sokoban-lm/) | the Python package `sokolm` and its tests |
| [`results/sft-vs-grpo/`](results/sft-vs-grpo/) | report, recorded runs (no checkpoints), the 2,000 new boards, analysis script |

`sokolm` modules:

| module | what it does |
|---|---|
| `sokoban`, `solver`, `simulate` | board state and move rules, BFS for the optimal solution length, execution and the binary reward |
| `generate`, `symmetry`, `splits` | seeded board generator; the fixed train/val/test split by layout class (no problem in two splits) |
| `formats` | prompt rendering and the strict answer parser |
| `assets`, `lm` | the pinned model files (sha256-checked) and batched generation with every decoding setting explicit |
| `train` | TRL wiring: SFT and GRPO datasets, the reward function, trainers, SFT target log-probabilities |
| `evaluate` | greedy solve rate, pass@k, board-blind baselines, GRPO group statistics |
| `pipeline` | the run commands: `build-split`, `new-boards`, `eval`, `sft`, `grpo`, `val-loss` |
| `record`, `cli`, `guard`, `paths` | run directories with provenance, the command line, a resource guard, file locations |

## Setup

Linux with an NVIDIA GPU, [uv](https://docs.astral.sh/uv/) and Python 3.11. The recorded runs used one RTX 5090
(32 GB) under WSL2; training reserved up to 11.5 GB of GPU memory.

```sh
cd sokoban-lm
uv sync                                   # torch 2.11 (CUDA 12.8 wheels), transformers 5.19, TRL 1.14.2
uv run pytest                             # tests that need the files below are skipped until they exist
uv run python -m sokolm fetch             # Qwen2.5-0.5B-Instruct at a pinned commit, into data/hf/
uv run python -m sokolm build-split       # the 7x7-v1 split into data/splits/, checked against splits/7x7-v1.json
```

The commands for every run of the report are in its [reproduction section](results/sft-vs-grpo/report.md#7-reproduction).
Each run writes `runs/<timestamp>-<name>/` with its manifest (code commit, library versions, config), metrics and
answers.

Environment variables:

| variable | default | effect |
|---|---|---|
| `SOKOLM_DATA` | `sokoban-lm/data` | model files and split files |
| `SOKOLM_RUNS` | `sokoban-lm/runs` | run directories |
| `SOKOLM_THREADS` | `4` | CPU thread pools of numpy and torch |
| `SOKOLM_GUARD` | `1` | `0` turns the resource guard off |

The resource guard (`sokolm.guard`) samples Windows and WSL memory and the GPU every 2 s during a run and stops the
run when a limit is crossed. It reads Windows memory through PowerShell, so it only works under WSL2; elsewhere set
`SOKOLM_GUARD=0`.

## Author's contributions

The author designed and directed the study and is responsible for its conclusions. The author set its goal, scope
and compute budget, designed the task and the evaluation, reviewed the implementation and the experimental results
during development, and decided the direction of the experiments and when to stop. Specific contributions include:

- **Task.** The author chose the 7×7 board size and defined what counts as one problem: boards with the same walls and
  targets, up to rotation and reflection. The train/val/test split is built on this definition, so no problem appears
  in two splits.
- **Data and evaluation.** The author divided the training boards into an SFT half and an RL half with the same share
  of every optimal length, and added two measurements next to the greedy solve rate: pass@k at temperature 1 with 16
  samples per board, and the solve rate on training boards after SFT, to tell fitting apart from generalization.
- **SFT.** The author extended the planned 500-step learning-rate sweep with larger learning rates and 3-epoch runs,
  and set the SFT configuration from their results (3 epochs, learning rate 2e-5). The author also posed the question
  of whether SFT should train on boards with short optimal lengths and set its test at three runs per condition.
- **GRPO.** Before the TRL runs, a from-scratch GRPO update was written and checked against TRL's GRPO loss (not part
  of this release). The author reviewed its functions, and in a smaller policy-gradient experiment before it wrote the
  exact gradient in closed form.
- **Comparisons.** The author added run 7, a comparison to assess whether one epoch of training on the additional
  boards alone could explain GRPO's improvement (§5.2 of the report), and decided to stop the follow-up runs after it.
- **Follow-up on new boards.** The author questioned whether run 6's in-distribution advantage over run 7 showed a
  GRPO gain that survives the learning-rate decay, and made it the follow-up's question. The author revised the test
  design with SFT controls that skip run 7's learning-rate re-warmup, to separate GRPO's objective from run 7's
  schedule, and chose to evaluate on new boards first, with further runs only if the advantage held there. When it
  did not, the author extended the evaluation to every model of the report and revised the report and this README
  (§5.5 of the report).
- **Review.** The author reviewed the results before deciding the next steps and corrected conclusions that went
  beyond the evidence. In the report, the author set how the summary is framed and required it to state the setting's
  bias toward SFT, that the GRPO–SFT difference depends on the starting model, and the cost as a per-step ratio rather
  than one run's total.
- **This release.** The author decided what the release contains and how this repository relates to the development
  repository.

## Use of AI tools

Under the author's direction, Claude Code (Anthropic, Claude Opus 5.5) wrote the code in this repository and the
analysis scripts, ran the experiments on the author's workstation, and drafted the documents, including this README and
the report. It also proposed settings and follow-up runs, which the author chose among or changed. The author also used
ChatGPT (GPT-6 Astra, GPT-6.1 Sol) and Claude (Opus 5.5), separately from Claude Code, to help write the working-method
statement for the collaboration and to review intermediate results.

SFT targets were generated with a BFS solver. Rewards and evaluation were computed programmatically; external LLMs
were not used to generate target solutions or judge model outputs. Training uses TRL's `SFTTrainer` and
`GRPOTrainer`, with repository code providing data preparation, reward computation, evaluation, and experiment
records.

## License

MIT, see [LICENSE](LICENSE).

Qwen2.5-0.5B-Instruct is released by the Qwen team under the Apache License 2.0; this repository does not contain its
weights. The code uses PyTorch, Hugging Face Transformers and TRL, each under its own license.
