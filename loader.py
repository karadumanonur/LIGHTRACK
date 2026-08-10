"""File loading for Lightrack POC1 .txt records.

Everything in here is defensive for one reason: the on-disk format has two
properties that make a naive ``pd.read_csv`` silently wrong rather than loudly
broken.

1. The header row is malformed. It reads ``Time [sec]\\tWavelength 1[nm]\\t``
   — two field names — while every data row carries 13 fields. Any reader that
   trusts the header will mis-align or drop columns.
2. The decimal separator is '.' in some files and ',' in others, and it is NOT
   a function of date. Reading a comma-decimal file with decimal='.' does not
   raise; pandas yields object dtype or, worse, a plausible-looking integer
   100000x too large. Sniff it per file. Always.

See ASSUMPTIONS.md A-02 and A-03.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG

# 20260514T142617,366Z.txt  ->  the ',' stands in for the ISO '.' before ms.
_FNAME_RE = re.compile(
    r"^(?P<Y>\d{4})(?P<m>\d{2})(?P<d>\d{2})T"
    r"(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2}),(?P<ms>\d{3})Z\.txt$"
)

N_CHANNELS = 12
TIME_COL = "time_s"
CH_COLS = [f"ch{i:02d}" for i in range(1, N_CHANNELS + 1)]

# Channel <-> PAD mapping. The interrogator names its channels by wavelength
# slot; the installation drawing names the sensors by pad. The two run in
# opposite directions, so every result reported against a pad has to come
# through here. Read from config, never hard-coded.
PAD_OF_CHANNEL = dict(zip(CH_COLS, CONFIG["channel_to_pad"]["pad_of_channel"]))
CH_OF_PAD = {pad: ch for ch, pad in PAD_OF_CHANNEL.items()}
COLUMNS = [TIME_COL] + CH_COLS


class FormatError(RuntimeError):
    """Raised when a file does not match the format established in Phase 0."""


def parse_filename_utc(name: str) -> datetime:
    """Filename -> timezone-aware UTC datetime.

    The trailing 'Z' is taken at face value: these are UTC, and Belgian local
    time is UTC+2 in this period (briefing §10). Off-by-two-hours silently
    ruins any timetable correlation, so local time is only ever produced by
    :func:`to_local`, never by accident.
    """
    m = _FNAME_RE.match(Path(name).name)
    if m is None:
        raise FormatError(f"filename does not match the POC1 pattern: {name!r}")
    g = m.groupdict()
    return datetime(
        int(g["Y"]), int(g["m"]), int(g["d"]),
        int(g["H"]), int(g["M"]), int(g["S"]),
        int(g["ms"]) * 1000,
        tzinfo=timezone.utc,
    )


def to_local(dt_utc: datetime) -> datetime:
    """UTC -> Belgian local wall clock, using the fixed offset from config.

    Fixed +2 is correct for 2026-05-14..2026-07-03 (CEST). This function is
    NOT a general tz converter and will be wrong across the October DST edge.
    """
    off = CONFIG["filename"]["local_utc_offset_hours"]
    return dt_utc.astimezone(timezone(timedelta(hours=off)))


def sniff_decimal(path: str | Path) -> str:
    """Return '.' or ',' — the decimal separator actually used in this file.

    Decided from the first data line only. The file is TAB-delimited, so a ','
    can never be a field separator here; if a ',' appears inside a field it is
    unambiguously the decimal mark.
    """
    with open(path, "r", encoding=CONFIG["format"]["encoding"]) as fh:
        fh.readline()  # malformed header
        first = fh.readline()
    if not first:
        raise FormatError(f"no data rows in {path}")
    fields = first.rstrip("\r\n").split("\t")
    has_comma = any("," in f for f in fields)
    has_dot = any("." in f for f in fields)
    if has_comma and has_dot:
        raise FormatError(
            f"{path}: both ',' and '.' present in a data row — separator ambiguous"
        )
    if has_comma:
        return ","
    if has_dot:
        return "."
    raise FormatError(f"{path}: no decimal mark found in first data row")


def sniff_column_order(path: str | Path, decimal: str | None = None) -> str:
    """Return 'desc' or 'asc' — the spectral order of the wavelength columns.

    POC1's 93 records list the twelve FBGs in DESCENDING centre wavelength
    (1566.7 -> 1516.0). The July 2026 delivery lists them ASCENDING, with a
    complete header (`Wavelength 1[nm] ... Wavelength 12[nm]`) where POC1's was
    truncated to a single name — i.e. the export changed.

    Reading an ascending file as if it were descending silently maps every
    sensor to the wrong PAD: ch01 would be read as PAD 12 when it is PAD 1.
    Nothing would raise; the amplitudes are all plausible.

    This is a PARSING property of the file, in the same class as the decimal
    separator (A-03) — not a tuned parameter. Detecting it and normalising to
    POC1 order is required to read the file correctly, and does not constitute
    tuning on the data.
    """
    dec = decimal or sniff_decimal(path)
    with open(path, "r", encoding=CONFIG["format"]["encoding"]) as fh:
        fh.readline()
        first = fh.readline()
    vals = first.rstrip("\r\n").split("\t")[1:]
    conv = (lambda s: float(s.replace(",", "."))) if dec == "," else float
    w = [conv(v) for v in vals if v.strip()]
    if len(w) < 2:
        raise FormatError(f"{path}: cannot determine column order")
    return "desc" if w[0] > w[-1] else "asc"


def load_file(path: str | Path) -> pd.DataFrame:
    """Load one record into a DataFrame with columns time_s, ch01..ch12.

    Values stay in **nm as stored** (absolute wavelength). Conversion to a pm
    shift needs a reference and is therefore a separate, explicit step — doing
    it here would bake in a baseline choice that the safety and diagnostic
    chains must be free to make differently (briefing §7.5).
    """
    path = Path(path)
    dec = sniff_decimal(path)
    df = pd.read_csv(
        path,
        sep=CONFIG["format"]["delimiter"],
        skiprows=CONFIG["format"]["header_rows"],  # header is malformed, discard
        header=None,
        names=COLUMNS,
        decimal=dec,
        dtype="float64",
        engine="c",
    )
    if df.shape[1] != CONFIG["format"]["n_columns"]:
        raise FormatError(f"{path}: expected 13 columns, got {df.shape[1]}")
    if df.isna().any().any():
        bad = df.isna().sum()
        raise FormatError(f"{path}: NaNs after parse (decimal={dec!r}): {bad.to_dict()}")

    # Normalise to POC1's descending-wavelength order so that ch01..ch12 mean
    # the same physical gratings in every delivery. See sniff_column_order.
    if sniff_column_order(path, dec) == "asc":
        df[CH_COLS] = df[CH_COLS[::-1]].to_numpy()
    return df


@dataclass
class FileMeta:
    """One row of the file index. Pure facts read off disk — no inference."""

    name: str
    path: str
    bytes: int
    utc: datetime
    local: datetime
    decimal: str
    n_rows: int
    t_first: float
    t_last: float
    duration_s: float
    dt_median: float
    dt_min: float
    dt_max: float
    fs_hz: float
    n_dt_outliers: int
    clock_offset_s: float  # instrument clock t_first minus filename epoch

    def as_dict(self) -> dict:
        d = asdict(self)
        d["utc"] = self.utc.isoformat()
        d["local"] = self.local.isoformat()
        return d


def scan_file(path: str | Path) -> tuple[FileMeta, np.ndarray]:
    """Full read of one file -> (metadata, wavelengths_nm[n, 12])."""
    path = Path(path)
    dec = sniff_decimal(path)
    df = load_file(path)
    t = df[TIME_COL].to_numpy()
    wl = df[CH_COLS].to_numpy()

    dt = np.diff(t)
    dt_med = float(np.median(dt))
    # An outlier here means a sample drop mid-file. At 500 Hz a single dropped
    # sample doubles dt; tolerance is deliberately tight (1%).
    outliers = int(np.count_nonzero(np.abs(dt - dt_med) > 0.01 * dt_med))

    utc = parse_filename_utc(path.name)
    # Instrument clock is a free-running counter, not an epoch. Its offset from
    # the filename timestamp is the quantity that reveals resets/drift.
    epoch_s = utc.timestamp()

    meta = FileMeta(
        name=path.name,
        path=str(path),
        bytes=path.stat().st_size,
        utc=utc,
        local=to_local(utc),
        decimal=dec,
        n_rows=len(t),
        t_first=float(t[0]),
        t_last=float(t[-1]),
        duration_s=float(t[-1] - t[0]),
        dt_median=dt_med,
        dt_min=float(dt.min()),
        dt_max=float(dt.max()),
        fs_hz=float(1.0 / dt_med),
        n_dt_outliers=outliers,
        clock_offset_s=float(t[0]) - epoch_s,
    )
    return meta, wl


def list_files(data_dir: str | Path | None = None) -> list[Path]:
    """POC1 records in the data directory, in filename order.

    Filters on the NAMING PATTERN, not just the extension. Globbing `*.txt`
    alone picks up any unrelated text file that happens to sit in the folder —
    which happened, and produced a UnicodeDecodeError deep inside the parser
    rather than a clear "this is not a record" at the listing step.
    """
    d = Path(data_dir or CONFIG["paths"]["glob"] and CONFIG["paths"]["data_dir"])
    return sorted((p for p in d.glob(CONFIG["paths"]["glob"])
                   if _FNAME_RE.match(p.name)), key=lambda p: p.name)
