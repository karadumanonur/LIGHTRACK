"""Reading interrogator records from disk, and feeding them in blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .config import CONFIG

FS = CONFIG["acquisition"]["fs_hz"]
N_CH = 12
PAD_OF_CHANNEL = CONFIG["channel_to_pad"]["pad_of_channel"]
CH_OF_PAD = {pad: i for i, pad in enumerate(PAD_OF_CHANNEL)}


class FormatError(RuntimeError):
    pass


def sniff_decimal(path: str | Path) -> str:
    """Return '.' or ',' — the decimal mark this file actually uses.

    It varies file to file and is not a function of date. Reading a
    comma-decimal file as if it used a point does not raise; it yields values a
    thousand times too large, silently.
    """
    with open(path, "r", encoding="ascii") as fh:
        fh.readline()
        first = fh.readline()
    if not first:
        raise FormatError(f"no data rows in {path}")
    fields = first.rstrip("\r\n").split("\t")
    comma = any("," in f for f in fields)
    dot = any("." in f for f in fields)
    if comma and dot:
        raise FormatError(f"{path}: both ',' and '.' present")
    if comma:
        return ","
    if dot:
        return "."
    raise FormatError(f"{path}: no decimal mark found")


def sniff_column_order(path: str | Path, decimal: str) -> str:
    """Return 'desc' or 'asc' — the spectral order of the twelve columns.

    Reading an ascending file as descending maps every sensor to the wrong pad
    and nothing raises, so this is checked rather than assumed.
    """
    with open(path, "r", encoding="ascii") as fh:
        fh.readline()
        first = fh.readline()
    vals = first.rstrip("\r\n").split("\t")[1:]
    conv = (lambda s: float(s.replace(",", "."))) if decimal == "," else float
    w = [conv(v) for v in vals if v.strip()]
    if len(w) < 2:
        raise FormatError(f"{path}: cannot determine column order")
    return "desc" if w[0] > w[-1] else "asc"


def read_channels(path: str | Path) -> np.ndarray:
    """Load one record as an (n, 12) array of wavelengths in nm, in pad order.

    Column 0 is PAD 1. The stored order differs between deliveries, so it is
    normalised here.
    """
    path = Path(path)
    dec = sniff_decimal(path)
    rows = []
    with open(path, "r", encoding="ascii") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < N_CH + 1:
                continue
            if dec == ",":
                parts = [p.replace(",", ".") for p in parts]
            rows.append(parts[1:N_CH + 1])
    if not rows:
        raise FormatError(f"{path}: no usable rows")
    w = np.asarray(rows, dtype=float)
    if sniff_column_order(path, dec) == "asc":
        w = w[:, ::-1]
    return w[:, [CH_OF_PAD[p] for p in range(1, N_CH + 1)]]


def blocks(signal: np.ndarray, size: int | None = None) -> Iterator[np.ndarray]:
    """Cut a channel into the blocks a live feed would deliver."""
    size = int(size or CONFIG["realtime"]["block_samples"])
    for i in range(0, len(signal), size):
        yield signal[i:i + size]
