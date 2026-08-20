"""Run both branches over the same records and print what each one saw.

    python compare/run.py path/to/records
    python compare/run.py path/to/records --limit 20 --sensor 12 --pad 1

The two branches are checked out beside this one as git worktrees, so nothing
is copied or vendored and neither side can drift from what is on GitHub. The
first run creates them; later runs reuse them.

Each record produces one line per side and a verdict:

    both saw a train, the axle counts agree
    both saw a train, the axle counts differ
    only one of them saw a train
    neither saw a train

The verdicts are counted at the end. A disagreement is not a scoreboard — the
two sides answer slightly different questions, and the point of putting them on
one page is to see exactly where the answers part company.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIDES = ROOT / ".sides"
BRANCHES = {"onur": "Onur-Branch", "asude": "Asude-Branch"}


def worktree(name: str, branch: str) -> Path:
    """Check the branch out beside this one, detached so it stays available."""
    path = SIDES / name
    if (path / ".git").exists():
        return path
    # A half-removed worktree leaves either its registration or its checkout
    # behind, and git then refuses to add the same path again. Both are ours
    # and both are a checkout of a branch, so clearing them costs nothing.
    subprocess.run(["git", "worktree", "prune"], cwd=ROOT, capture_output=True)
    if path.exists():
        print(f"   clearing a leftover checkout in {path.relative_to(ROOT)}")
        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            sys.exit(f"{path} is in the way and could not be removed — a file "
                     "in it is open somewhere. Close it and run again.")
    ref = f"origin/{branch}"
    if subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref],
                      cwd=ROOT, capture_output=True).returncode:
        ref = branch
    print(f"   checking out {ref} into {path.relative_to(ROOT)}")
    done = subprocess.run(["git", "worktree", "add", "--detach", str(path), ref],
                          cwd=ROOT, capture_output=True, text=True)
    if done.returncode:
        sys.exit(f"could not check out {ref}:\n{done.stderr.strip()}\n"
                 "fetch the branches first: git fetch origin")
    return path


def load(name: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return importlib.import_module(f"sides.{name}")


def pad_of_column(tree: Path, column: int, path: Path) -> str:
    """Which PAD the other side's fixed column lands on, for this file."""
    try:
        sys.path.insert(0, str(tree))
        from analysis.config import CONFIG                      # noqa: E402
        from analysis.reader import sniff_column_order, sniff_decimal
        order = sniff_column_order(path, sniff_decimal(path))
        pads = CONFIG["channel_to_pad"]["pad_of_channel"]
        i = column - 1
        return f"PAD {pads[i] if order == 'desc' else pads[len(pads) - 1 - i]}"
    except Exception:
        return "unknown"


def verdict(a: dict | None, b: dict | None, names: list[str]) -> str:
    if a is None or b is None:
        return "one side could not read the file"
    saw_a, saw_b = a["axles"] > 0, b["axles"] > 0
    if not saw_a and not saw_b:
        return "neither saw a train"
    if saw_a != saw_b:
        return f"only {names[0] if saw_a else names[1]} saw a train"
    if a["axles"] == b["axles"]:
        return "both saw a train, the axle counts agree"
    return "both saw a train, the axle counts differ"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="a folder of .txt recordings")
    ap.add_argument("--limit", type=int, default=0, help="stop after N records")
    ap.add_argument("--pad", type=int, default=1, help="PAD for the Onur side")
    ap.add_argument("--sensor", type=int, default=12,
                    help="wavelength column for the Asude side")
    ap.add_argument("--out", type=Path, default=ROOT / "compare" / "out")
    args = ap.parse_args()

    files = sorted(args.data.glob("*.txt"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no .txt recordings in {args.data}")

    print("preparing the two branches")
    trees = {n: worktree(n, b) for n, b in BRANCHES.items()}
    mods = {n: load(n) for n in BRANCHES}
    for n, m in mods.items():
        m.prepare(trees[n])
        print(f"   {m.NAME:<6} {trees[n].relative_to(ROOT)}  :  {m.DESCRIPTION}")

    NAMES = [mods["onur"].NAME, mods["asude"].NAME]
    mapped = pad_of_column(trees["onur"], args.sensor, files[0])
    print(f"\n{len(files)} records from {args.data}")
    print(f"{NAMES[0]} reads PAD {args.pad}; {NAMES[1]} reads column "
          f"{args.sensor}, which on this file is {mapped}\n")

    head = (f"{'record':<26}{NAMES[0]:>8}{NAMES[1]:>8}   verdict")
    print(head)
    print("-" * len(head))

    rows, counts = [], {}
    for path in files:
        got = {}
        for n, m in mods.items():
            try:
                got[n] = m.run(path, args.pad if n == "onur" else args.sensor)
            except Exception as exc:
                got[n] = None
                print(f"{path.name[:24]:<26}{'':>8}{'':>8}   "
                      f"{m.NAME} could not read it: "
                      f"{str(exc).splitlines()[0][:60]}")
        a, b = got["onur"], got["asude"]
        v = verdict(a, b, NAMES)
        counts[v] = counts.get(v, 0) + 1
        if a is None or b is None:
            continue
        print(f"{path.name[:24]:<26}{a['axles']:>8}{b['axles']:>8}   {v}")
        rows.append({
            "record": path.name,
            f"{NAMES[0]} axles": a["axles"], f"{NAMES[1]} axles": b["axles"],
            f"{NAMES[0]} trains": a["trains"], f"{NAMES[0]} detail": a["detail"],
            f"{NAMES[0]} speed_kmh": a["speed_kmh"],
            f"{NAMES[0]} first_s": a["first_s"], f"{NAMES[1]} first_s": b["first_s"],
            "difference": a["axles"] - b["axles"], "verdict": v,
        })

    print(f"\n{'':<26}{'':>8}{'':>8}   summary")
    print("-" * len(head))
    for v, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{n:>5}  {v}")
    both = [r for r in rows if r["verdict"].startswith("both")]
    if both:
        agree = sum(1 for r in both if r["difference"] == 0)
        print(f"\nof the {len(both)} records where both saw a train, "
              f"{agree} agree on the axle count ({100 * agree / len(both):.0f} %)")
        off = sorted(both, key=lambda r: -abs(r["difference"]))[:5]
        if off and off[0]["difference"]:
            print("the widest disagreements")
            for r in off:
                if r["difference"]:
                    print(f"   {r['record'][:24]:<26}{NAMES[0]} "
                          f"{r[NAMES[0] + ' axles']:>4}   {NAMES[1]} "
                          f"{r[NAMES[1] + ' axles']:>4}")

    if rows:
        import csv
        args.out.mkdir(parents=True, exist_ok=True)
        target = args.out / "comparison.csv"
        with open(target, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten to {target}")


if __name__ == "__main__":
    main()
