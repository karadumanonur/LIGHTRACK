# LIGHTRACK

FBG axle counting between Aiseau and Farciennes. An internship at UMONS for
B-sens.

The code lives on the branches, one per person:

| branch | what is on it |
|---|---|
| `Onur-Branch` | the live counter — a streaming package that counts axles block by block |
| `Asude-Branch` | the peak-algorithm and matching studies |

`main` holds only the harness that runs both of them over the same recordings
and prints the two answers side by side. Nothing is copied or vendored: the
branches are checked out beside `main` as git worktrees, so what runs is the
code as it stands on them.

## Running it

**1. Get the project.** A clone brings all three branches with it.

```powershell
git clone https://github.com/karadumanonur/LIGHTRACK.git
cd LIGHTRACK
```

If you cloned it before, update `main` and make sure both branches are fetched
— the harness runs them from `origin/Onur-Branch` and `origin/Asude-Branch`, so
it cannot find them otherwise.

```powershell
git pull origin main
git fetch origin
```

**2. Install what both sides need.**

```powershell
python -m pip install numpy pandas scipy matplotlib
```

**3. Put the recordings somewhere.** Either inside the clone or beside it —
`.gitignore` excludes everything but the harness, so a `detected/` folder sitting
in the clone stays out of git.

**4. Run it**, pointing at the folder of `.txt` recordings.

```powershell
python compare/run.py detected
python compare/run.py ../detected --limit 20
```

The first run checks both branches out into `.sides/`, which takes a few
seconds; later runs reuse them. `--limit` is worth using the first time. The
whole `detected/` set is about half an hour, since every file is read twice.

Other options: `--pad 4` chooses the PAD the Onur side reads, `--sensor 9` the
wavelength column the Asude side reads, and `--out somewhere` moves the CSV.

## What comes out

```
record                        Onur   Asude   verdict
----------------------------------------------------
20260515T032854,182Z.txt        24      24   both saw a train, the axle counts agree
20260519T113644,219Z.txt       121     136   both saw a train, the axle counts differ
20260515T092351,183Z.txt         0       0   neither saw a train

  189  both saw a train, the axle counts agree
   10  both saw a train, the axle counts differ
    1  neither saw a train

of the 199 records where both saw a train, 189 agree on the axle count (95 %)
```

and `compare/out/comparison.csv` with a row per record: both counts, the
difference, how many trains the Onur side split the record into, its speed, and
when each side saw its first axle.

## Two things to know before reading the numbers as a score

**The two sides pick their channel differently.** The Asude side takes the
wavelength column by position — `--sensor 12` is the twelfth column of the file.
The Onur side asks for a PAD and works out which column that is, because the
column order is not the same in every delivery. The harness prints the mapping
for the first file it reads, so `Onur reads PAD 1; Asude reads column 12, which
on this file is PAD 1` means both are on the same rail. If a delivery arrives in
the other order, that line will say so.

**A peak is not a train.** The Asude side numbers peaks over the whole
recording; it has no notion of a train, so a record holding two passages gives
one number. The Onur side finds trains and the harness adds their axles up, so
that the two numbers answer the same question. The `Onur trains` and
`Onur detail` columns in the CSV say when that mattered.

Neither of these is a fault. They are the reason the comparison is worth
running: a disagreement usually points at one of them rather than at a
miscounted axle.

## Adding a third side

Drop a module in `compare/sides/` with a `NAME`, a `DESCRIPTION`, a
`prepare(worktree)` that puts its branch on `sys.path`, and a `run(path, ...)`
returning `axles`, `trains`, `detail`, `first_s`, `speed_kmh`, `sensor`. Then
add its branch to `BRANCHES` in `compare/run.py`.

## If it will not start

- `could not check out origin/Onur-Branch` — run `git fetch origin`.
- `.sides/onur is in the way` — delete that folder and run again.
- `no .txt recordings in ...` — the path must hold the `.txt` files directly.
