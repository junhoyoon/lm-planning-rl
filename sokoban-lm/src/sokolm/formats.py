"""Prompt and answer formats.

Used: board `spaced`, actions `words`, the strict answer line below, completion cap COMPLETION_CAP tokens. The
alternatives (`xsb`, `letters`) stay selectable.

The prompt is one user message passed through the model's chat template. No system message is given, so Qwen2.5's
template inserts its default one; `prompt_text` returns the exact string the model sees, ending with the
generation prompt "<|im_start|>assistant\\n".

Board styles:
  xsb     the XSB rows as stored ('#' wall, ' ' floor, '.' target, '$' box, '*' box on target, '@' player,
          '+' player on target); BPE merges neighbouring cells into tokens of 1 to 7 cells
  spaced  the same symbols with '_' for floor and one space between cells, so that every cell is its own token

Answer: one line "Answer: a1 a2 ...", the actions separated by single spaces, from one of two vocabularies:
  words    up down left right
  letters  U D L R
After "Answer:" every action is a single token that starts with a space (" up", " R"), so an action has the same
token id wherever it stands in the answer. Without separators BPE merges actions ("URRDL" -> "UR" "R" "DL").

`parse_answer` is strict: after stripping surrounding whitespace the completion must be exactly that line, with
at least one action, each written exactly as in the vocabulary. Anything else is a parse failure with a reason.
"""

from dataclasses import dataclass
from typing import Optional

from sokolm.simulate import judge, simulate
from sokolm.sokoban import Board

BOARD_STYLES = ("xsb", "spaced")
VOCABS = {"words": ("up", "down", "left", "right"), "letters": ("U", "D", "L", "R")}  # index = action (sokoban.ACTIONS)
ANSWER_PREFIX = "Answer:"
LEGEND = {
    "xsb": "# wall, (space) floor, . target, $ box, * box on a target, @ player, + player on a target",
    "spaced": "# wall, _ floor, . target, $ box, * box on a target, @ player, + player on a target",
}
PARSE_ERRORS = ("empty", "not_one_line", "no_prefix", "no_actions", "bad_item")
COMPLETION_CAP = 128  # max_new_tokens; the answer for the longest budget (92 actions) takes 95 tokens with the end token
EXAMPLE_ACTIONS = (2, 0, 0, 3)  # the answer shown in the prompt: left up up right


@dataclass(frozen=True)
class Format:
    board: str = "spaced"  # the defaults are the format used in every run
    vocab: str = "words"

    @property
    def name(self) -> str:
        return f"{self.board}-{self.vocab}"


def render_board(text: str, style: str) -> str:
    if style == "xsb":
        return text
    if style == "spaced":
        return "\n".join(" ".join(row.replace(" ", "_")) for row in text.split("\n"))
    raise ValueError(f"board style must be one of {BOARD_STYLES}, got {style!r}")


def user_message(text: str, fmt: Format) -> str:
    words = VOCABS[fmt.vocab]
    example = " ".join(words[a] for a in EXAMPLE_ACTIONS)
    return (
        "Solve this Sokoban puzzle.\n\n"
        f"{render_board(text, fmt.board)}\n\n"
        f"Symbols: {LEGEND[fmt.board]}.\n"
        "Goal: push every box onto a target.\n"
        f"Moves: {', '.join(words)}. Moving into a box pushes it one cell if the cell behind it is floor or a target; "
        "otherwise nothing happens. Walls block you, and boxes cannot be pulled.\n"
        f'Reply with one line: "{ANSWER_PREFIX}" followed by your moves separated by spaces, for example '
        f'"{ANSWER_PREFIX} {example}".'
    )


def messages(text: str, fmt: Format) -> list[dict]:
    return [{"role": "user", "content": user_message(text, fmt)}]


def prompt_text(tokenizer, text: str, fmt: Format, **template_kwargs) -> str:
    """The exact prompt string; `template_kwargs` go to the chat template."""
    return tokenizer.apply_chat_template(messages(text, fmt), tokenize=False, add_generation_prompt=True,
                                         **template_kwargs)


def answer_text(actions, fmt: Format) -> str:
    """The completion text for an action sequence (e.g. the BFS solution as an SFT target), without the end token."""
    return f"{ANSWER_PREFIX} " + " ".join(VOCABS[fmt.vocab][a] for a in actions)


def parse_answer(text: str, fmt: Format) -> tuple[Optional[list[int]], Optional[str]]:
    """(actions, None) for a well-formed completion, else (None, reason) with reason in PARSE_ERRORS."""
    s = text.strip()
    if not s:
        return None, "empty"
    if "\n" in s:
        return None, "not_one_line"
    if not s.startswith(ANSWER_PREFIX):
        return None, "no_prefix"
    body = s[len(ANSWER_PREFIX):]
    if not body.strip():
        return None, "no_actions"
    if not body.startswith(" "):
        return None, "bad_item"
    words = VOCABS[fmt.vocab]
    items = body[1:].split(" ")
    if any(w not in words for w in items):  # wrong case, punctuation, a double space, an unknown word
        return None, "bad_item"
    return [words.index(w) for w in items], None


def judge_completion(board: Board, completion: str, fmt: Format, budget: int, opt_len: Optional[int] = None,
                     truncated: bool = False) -> dict:
    """Parse, execute and judge one completion. `truncated` = generation hit the token limit without the end token;
    the text is still parsed and executed so that its record is complete, but the reward follows the rule for
    truncated outputs."""
    actions, error = parse_answer(completion, fmt)
    o = simulate(board, actions, budget) if actions is not None else None
    return judge(o, budget, opt_len, parse_ok=actions is not None, truncated=truncated) | {"parse_error": error}
