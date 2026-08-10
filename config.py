"""Config access. Parameters live in config.toml, never in code."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).with_name("config.toml")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else _CONFIG_PATH
    with open(p, "rb") as fh:
        return tomllib.load(fh)


CONFIG = load_config()
