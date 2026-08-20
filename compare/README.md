# Comparing the two branches

Two people wrote axle counters for the same rail from the same recordings.
This runs both over the same files and prints what each one saw, so the two can
be read against each other instead of against a description of each other.

```powershell
python compare/run.py path/to/records
python compare/run.py path/to/records --limit 20
```

The first run checks the branches out beside this one as git worktrees, under
`.sides/`. Nothing is copied into `main` and nothing is vendored: what runs is
the code as it stands on `Onur-Branch` and `Asude-Branch`. Later runs reuse the
worktrees; `git worktree remove .sides/onur` undoes it.

## What comes out

```
record                        Onur   Asude   verdict
----------------------------------------------------
20260515T032854,182Z.txt        24      24   both saw a train, the axle counts agree
20260515T053253,794Z.txt        24      25   both saw a train, the axle counts differ
20260515T092351,183Z.txt         0       0   neither saw a train

    55  both saw a train, the axle counts agree
     4  both saw a train, the axle counts differ
     1  neither saw a train

of the 59 records where both saw a train, 55 agree on the axle count (93 %)
```

and `compare/out/comparison.csv` with a row per record: both counts, the
difference, how many trains the Onur side split the record into, its speed, and
when each side saw its first axle.

## Two things to know before reading the numbers

**The two sides pick their channel differently.** The Asude side takes the
wavelength column by position — `--sensor 12` is the twelfth column of the file.
The Onur side asks for a PAD and works out which column that is, because the
column order is not the same in every delivery. The harness prints the mapping
for the first file it reads, so `Onur reads PAD 1; Asude reads column 12, which
on this file is PAD 1` means both are on the same rail. If a delivery arrives in
the other order, that line will say so.

**A peak is not a train.** The Asude side numbers peaks over the whole
recording; it has no notion of a train, so a record holding two passages gives
one number. The Onur side finds trains and this harness adds their axles up, so
that the two numbers answer the same question. The `Onur trains` and
`Onur detail` columns in the CSV say when that mattered.

Neither of these is a fault. They are the reason the comparison is worth
running: a disagreement usually points at one of them rather than at a
miscounted axle.

## Adding a third side

Drop a module in `compare/sides/` with a `NAME`, a `DESCRIPTION`, a
`prepare(worktree)` that puts its branch on `sys.path`, and a `run(path, ...)`
returning `axles`, `trains`, `detail`, `first_s`, `speed_kmh`, `sensor`. Then
add its branch to `BRANCHES` in `run.py`.

## Installation

```powershell
python -m pip install numpy pandas scipy matplotlib
```
