# LIGHTRACK

FBG axle counting between Aiseau and Farciennes. An internship at UMONS for
B-sens.

The code lives on the branches, one per person:

| branch | what is on it |
|---|---|
| `Onur-Branch` | the live counter — a streaming package that counts axles block by block |
| `Asude-Branch` | the peak-algorithm and matching studies |

`main` holds only the harness that runs both of them over the same recordings
and prints the two answers side by side.

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
```

and `compare/out/comparison.csv` with a row per record. Before reading the
numbers as a score, read [`compare/README.md`](compare/README.md): the two sides
pick their channel differently and one counts peaks where the other counts
trains, and those two facts explain most of the disagreements.

## If it will not start

- `could not check out origin/Onur-Branch` — run `git fetch origin`.
- `.sides/onur is in the way` — delete that folder and run again.
- `no .txt recordings in ...` — the path must hold the `.txt` files directly.
