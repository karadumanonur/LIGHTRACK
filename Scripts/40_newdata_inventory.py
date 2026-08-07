"""Steps 1-2 — inventory and identity check on Valid_data_new. NO MERGING.

Step 1 reads only headers and a tail slice per file — enough for the decimal
separator, row count and dt without loading full records.

Step 2 is the question that decides what the data can be used for. The WDM comb
is the installation's fingerprint: if the twelve centre wavelengths match the
POC1 values within a few hundred pm, it is the same gratings on the same fibre
and the channel->PAD mapping and commissioning baselines carry over. If they are
shifted or reordered, the sensors were re-spliced or re-installed, the mapping
must be re-derived from scratch, and NO degradation curve may cross the
boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, parse_filename_utc, to_local, sniff_decimal  # noqa: E402
from lightrack.loader import _FNAME_RE, CH_COLS                            # noqa: E402

NEW = Path(r"C:\Users\onurk\Desktop\data\Valid_data_new")
OUT = Path(CONFIG["paths"]["out_dir"])
FS = CONFIG["acquisition"]["fs_hz"]
pd.set_option("display.width", 250)
LINE = "=" * 112

print(f"{LINE}\nSTEP 1 — INVENTORY (no records merged, no full loads)\n{LINE}")
allf = sorted(NEW.rglob("*"))
files = [p for p in allf if p.is_file()]
txt = [p for p in files if p.suffix.lower() == ".txt"]
recs = [p for p in txt if _FNAME_RE.match(p.name)]
skipped = [p for p in files if p not in recs]

print(f"directory      : {NEW}")
print(f"files total    : {len(files)}")
print(f".txt files     : {len(txt)}")
print(f"match POC1 name pattern: {len(recs)}")
print(f"SKIPPED by the loader's naming filter: {len(skipped)}")
for p in skipped[:20]:
    print(f"    {p.relative_to(NEW)}  ({p.stat().st_size} B)")
if len(skipped) > 20:
    print(f"    ... and {len(skipped)-20} more")

rows = []
for p in recs:
    utc = parse_filename_utc(p.name)
    with open(p, "rb") as fh:
        head = fh.readline()
        first = fh.readline()
        fh.seek(max(0, p.stat().st_size - 400))
        tail = fh.read().split(b"\n")
    tail = [t for t in tail if t.strip()][-1].decode("ascii", "replace")
    dec = sniff_decimal(p)
    conv = (lambda s: float(s.replace(",", "."))) if dec == "," else float
    t0 = conv(first.decode().split("\t")[0])
    t1 = conv(first.decode().split("\t")[0])
    tend = conv(tail.split("\t")[0])
    ncol = len(first.decode().rstrip().split("\t"))
    rows.append({"name": p.name, "rel": str(p.relative_to(NEW)),
                 "utc": utc, "local": to_local(utc), "bytes": p.stat().st_size,
                 "decimal": dec, "ncol": ncol, "t_first": t0, "t_last": tend,
                 "duration_s": tend - t0,
                 "header_ok": head.decode("ascii", "replace").rstrip("\r\n")
                 == "Time [sec]\tWavelength 1[nm]\t"})
inv = pd.DataFrame(rows).sort_values("utc").reset_index(drop=True)
inv["est_rows"] = (inv["duration_s"] / (1 / FS)).round().astype(int) + 1
inv.to_csv(OUT / "newdata_inventory.csv", index=False)

print(f"\nrecords            : {len(inv)}")
print(f"UTC range          : {inv['utc'].min()}  ->  {inv['utc'].max()}")
print(f"LOCAL range        : {inv['local'].min()}  ->  {inv['local'].max()}")
print(f"size min/max       : {inv['bytes'].min():,} / {inv['bytes'].max():,}")
print(f"decimal separator  : {inv['decimal'].value_counts().to_dict()}")
print(f"columns per row    : {inv['ncol'].value_counts().to_dict()}")
print(f"header byte-identical to POC1: {int(inv['header_ok'].sum())}/{len(inv)}")
print(f"duration s min/med/max: {inv['duration_s'].min():.3f} / "
      f"{inv['duration_s'].median():.3f} / {inv['duration_s'].max():.3f}")
print(f"implied rows  min/med/max: {inv['est_rows'].min()} / "
      f"{int(inv['est_rows'].median())} / {inv['est_rows'].max()}")

print(f"\nfiles per LOCAL day:")
for d, n in inv.groupby(inv["local"].dt.date).size().items():
    dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][
        pd.Timestamp(d).weekday()]
    print(f"   {d}  {dow}  {n:>3}")

# ---- overlap with the existing 93 -----------------------------------------
old = sorted(Path(CONFIG["paths"]["data_dir"]).glob("*.txt"))
old = [p for p in old if _FNAME_RE.match(p.name)]
old_names = {p.name for p in old}
old_utc = {parse_filename_utc(p.name) for p in old}
dup_name = sorted(set(inv["name"]) & old_names)
dup_time = sorted(set(inv["utc"]) & old_utc)
print(f"\nOVERLAP WITH THE EXISTING 93")
print(f"   identical filenames : {len(dup_name)}  {dup_name[:5]}")
print(f"   identical timestamps: {len(dup_time)}  {[str(t) for t in dup_time[:5]]}")
oldmax = max(old_utc)
print(f"   existing corpus ends: {oldmax}")
print(f"   new data starts     : {inv['utc'].min()}")
gap_days = (inv['utc'].min() - oldmax).total_seconds() / 86400
print(f"   gap between sets    : {gap_days:+.2f} days")
OUTAGE = (pd.Timestamp("2026-06-02", tz="UTC"), pd.Timestamp("2026-07-03", tz="UTC"))
before = (inv["utc"] < OUTAGE[0]).sum()
inside = ((inv["utc"] >= OUTAGE[0]) & (inv["utc"] <= OUTAGE[1])).sum()
after = (inv["utc"] > OUTAGE[1]).sum()
print(f"\n   relative to the 02/06-03/07 outage:")
print(f"      before 02/06 : {before}")
print(f"      inside window: {inside}")
print(f"      after 03/07  : {after}")

print(f"\n{LINE}\nSTEP 2 — IDENTITY CHECK: is this the same installation?\n{LINE}")
old_c = pd.read_csv(OUT / "channel_centres.csv", index_col=0).iloc[:, 0]
sample = inv.sample(min(8, len(inv)), random_state=0).sort_values("utc")
mats = []
for _, r in sample.iterrows():
    p = NEW / r["rel"]
    dec = r["decimal"]
    df = pd.read_csv(p, sep="\t", skiprows=1, header=None,
                     names=["time_s"] + CH_COLS, decimal=dec,
                     dtype="float64", engine="c", nrows=20000)
    mats.append(df[CH_COLS].median().values)
new_c = pd.Series(np.median(np.array(mats), axis=0), index=CH_COLS)

print(f"centres from {len(sample)} sampled new records vs POC1 "
      f"(out/channel_centres.csv)\n")
print(f"{'ch':>6}{'POC1 nm':>12}{'NEW nm':>12}{'delta pm':>11}   verdict")
deltas = []
for ch in CH_COLS:
    d = (new_c[ch] - old_c[ch]) * 1000
    deltas.append(d)
    v = "same grating" if abs(d) < 500 else ("SHIFTED" if abs(d) < 5000 else "DIFFERENT")
    print(f"{ch:>6}{old_c[ch]:>12.4f}{new_c[ch]:>12.4f}{d:>+11.1f}   {v}")
deltas = np.array(deltas)
print(f"\nmax |delta| {np.abs(deltas).max():.1f} pm, "
      f"median {np.median(deltas):+.1f} pm, sd {deltas.std():.1f} pm")
if np.abs(deltas).max() < 500:
    print("\nVERDICT: SAME INSTALLATION. All twelve centres within 500 pm of the")
    print("POC1 values and in the same spectral order. The channel->PAD mapping")
    print("and the commissioning baselines carry over.")
elif np.abs(deltas).max() < 5000:
    print("\nVERDICT: SAME GRATINGS, SHIFTED. Order preserved but centres moved")
    print("by more than 500 pm - consistent with a temperature offset (10 pm/degC")
    print(f"=> ~{np.median(deltas)/10:.0f} degC) or a re-tensioning. Mapping likely")
    print("carries over; commissioning AMPLITUDE baselines must be re-checked.")
else:
    print("\nVERDICT: DIFFERENT OR RE-SPLICED. Mapping must be re-derived from")
    print("scratch and NO degradation curve may cross the boundary.")
print(f"\nwrote {OUT/'newdata_inventory.csv'}")
