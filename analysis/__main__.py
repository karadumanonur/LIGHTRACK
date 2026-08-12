"""Count axles in one or more recordings.

    python -m analysis record.txt
    python -m analysis folder/*.txt --pad 1 --partner 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import CONFIG
from .counter import StreamingCounter, compare_heads
from .reader import blocks, read_channels

FS = CONFIG["acquisition"]["fs_hz"]
SEP_M = 15.0


def expand(paths: list[Path]) -> list[Path]:
    """Accept files, folders and wildcards alike."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("*.txt")))
        elif "*" in p.name or "?" in p.name:
            out.extend(sorted((p.parent or Path(".")).glob(p.name)))
        else:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--partner", type=int, default=4,
                    help="second sensor on the same head, confirms the count")
    ap.add_argument("--direction-pad", type=int, default=5,
                    help="sensor on the next head, gives direction and speed")
    ap.add_argument("--block", type=int, default=None)
    args = ap.parse_args()
    files = expand(args.files)

    total = 0
    for path in files:
        try:
            w = read_channels(path)
        except Exception as exc:
            print(f"{path.name}: cannot read — {exc}")
            continue

        counters = {}
        for pad in (args.pad, args.partner, args.direction_pad):
            if pad and pad not in counters:
                c = StreamingCounter(block=args.block or
                                     CONFIG["realtime"]["block_samples"])
                for b in blocks(w[:, pad - 1], args.block):
                    c.push(b)
                counters[pad] = c

        main_c = counters[args.pad]
        other = counters.get(args.direction_pad)
        heading = compare_heads(main_c, other, SEP_M) if other else None
        trains = main_c.flush()
        partner = counters.get(args.partner)
        partner_trains = partner.flush() if partner else []

        print(f"{path.name}")
        for i, t in enumerate(trains):
            confirmed = (i < len(partner_trains)
                         and partner_trains[i].axles == t.axles)
            mark = "confirmed" if confirmed else "UNCONFIRMED"
            line = (f"   t={t.first_index / FS:7.1f} s  {t.axles:4d} axles"
                    f"   X {t.n_x} / D {t.n_d}"
                    f"   {t.speed_kmh:5.0f} km/h   {mark}")
            if heading and i == 0:
                line += f"   head {heading.head_first} first"
            print(line)
            total += 1
        if not trains:
            print("   no train")
    print(f"\n{total} train(s) in {len(args.files)} file(s)")


if __name__ == "__main__":
    main()
