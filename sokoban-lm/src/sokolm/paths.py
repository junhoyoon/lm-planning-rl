"""Locations shared by all sokolm modules. Runs and data default to runs/ and data/ next to this project's
pyproject.toml; SOKOLM_RUNS and SOKOLM_DATA override them."""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parent
RUNS = Path(os.environ.get("SOKOLM_RUNS", PROJECT / "runs"))
DATA = Path(os.environ.get("SOKOLM_DATA", PROJECT / "data"))
