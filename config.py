"""Config access. Parameters live in config.toml, never in code.

PATHS ARE NOT PORTABLE and must not be hard-coded, so they are resolved here
rather than written out in the TOML. Three sources, in order of precedence:

  1. the environment — LIGHTRACK_DATA and LIGHTRACK_OUT
  2. the value in config.toml, if it is absolute
  3. the value in config.toml read as RELATIVE TO THE REPOSITORY ROOT, which is
     the directory this package lives in

So a clone works with no edit at all as long as the recordings sit where the
default expects them, and moving the data anywhere else needs an environment
variable rather than a change to a tracked file.

    setx LIGHTRACK_DATA "D:\\recordings\\poc1"        (Windows, once)
    export LIGHTRACK_DATA=/mnt/recordings/poc1        (Linux/macOS)
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).with_name("config.toml")
_ROOT = Path(__file__).resolve().parent          # the package == the repo root

_ENV = {"data_dir": "LIGHTRACK_DATA", "out_dir": "LIGHTRACK_OUT"}


def _resolve(key: str, value: str) -> str:
    """Absolute path for one entry of [paths]."""
    env = os.environ.get(_ENV.get(key, ""), "")
    if env:
        return str(Path(env).expanduser().resolve())
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = _ROOT / p
    return str(p.resolve())


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else _CONFIG_PATH
    with open(p, "rb") as fh:
        cfg = tomllib.load(fh)
    paths = cfg.get("paths", {})
    for key in ("data_dir", "out_dir"):
        if key in paths:
            paths[key] = _resolve(key, paths[key])
    return cfg


CONFIG = load_config()
