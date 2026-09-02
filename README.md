# LIGHTRACK

FBG axle counting between Aiseau and Farciennes. An internship at UMONS for
B-sens.

## Get the program

[**Download LIGHTRACK.exe**](https://github.com/karadumanonur/LIGHTRACK/releases/latest)
— one file, Windows, about 60 MB. No install, no administrator rights and no
Python: double-click it and the browser opens on the window. A console window
stays behind it saying where the program is listening; closing that window
stops the program.

On a Mac take `LIGHTRACK-apple-silicon.dmg`, or
`LIGHTRACK-intel.dmg` on a Mac made before 2020. Open it and drag LIGHTRACK to
Applications. The first time, **right-click the app and choose Open** rather
than double-clicking it: the program is not signed by a registered Apple
developer, and a plain double-click refuses it outright with no way forward.
If macOS says it is damaged, that is the same thing said badly, and

```bash
xattr -dr com.apple.quarantine /Applications/LIGHTRACK.app
```

answers it. On a Mac the store goes to
`~/Library/Application Support/LIGHTRACK`, so replacing the app leaves the work
where it is.

This repository is private, so that link asks you to sign in and works only for
people who have been given access to it. Anyone else has to be added to the
repository first, or handed the file directly.

Point it at a folder of `.txt` recordings with **Select folder** and it reads
them in, counts them and measures all twelve sensors. Its store is a single
`lightrack.db` file beside the program, so copying that folder carries the work
with it; put the program somewhere you can write to and it will keep it there.

Windows will say the file is not signed, because it is not. What it is built
from is this branch, and the recipe is `packaging/lightrack.spec`.

## What is on which branch

| branch | what is on it |
|---|---|
| `main` | the program — the window, the store, the counting package, and the harness that runs both engines over the same records |
| `Onur-Branch` | the counter on its own, as a streaming package |
| `Asude-Branch` | the peak-algorithm and matching studies |

The harness copies nothing: the branches are checked out beside `main` as git
worktrees, so what runs is the code as it stands on them.

## The window

`lightrack.app` is a small application over the same recordings: a store of
every record on one side, and on the other the raw signal, its spectra, its
modes, and — the point of it — the picture of how the axles were actually
found. It runs on the Python you already have; there is no web framework to
install, because the person who most needs to look at these records should not
have to fight a package manager first.

```powershell
python -m lightrack.app scan detected Valid_data_new poc1
python -m lightrack.app measure --engine onur --pad 1
python -m lightrack.app serve
```

`scan` reads only the file names, so it is quick and can be rerun whenever a
delivery arrives. `measure` is the slow one — it reads every record once, about
twenty minutes for eighteen hundred of them — and it can be interrupted and
resumed; the window shows whatever is already measured and leaves the rest
blank. `serve` opens the browser at `127.0.0.1:8765`.

One head is COUNTED and all twelve are MEASURED. Counting is a job for the head
that works; power is how you find out which heads still do. The passage window
comes from the counting head and is then used for every sensor, so the twelve
numbers describe the same train over the same rail at the same moment.

What is in the window:

| | |
|---|---|
| Select folder | point it at any folder of records; it reads them in and keeps them |
| the engine switch | Onur's counter or Asude's peak tracker, over the same record |
| the sensor row | click a grating to ADD it to every chart, click again to take it away |
| raw signal | with the baseline taken out so the gratings sit on top of each other; one button puts it back |
| axle count | what the engine looks at, its thresholds, and every mark it made |
| rate of change | how fast the signal is moving, in picometres per second |
| PSD, Fourier, EMD | of the raw signal |

**Sensors add up rather than replace.** Every chart draws each selected grating
in its own colour, so PAD 9 beside PAD 1 beside PAD 11 is one picture and not
three visits. That is the whole reason to have twelve of them: a sensor is only
loud or quiet compared to the one next to it on the same train.

**The rate of change** sits under the axle count and over the same window,
because that is the only way to read the two together. An axle arriving is a
steep climb and an axle leaving is a steep fall, and the derivative crosses
zero exactly where the signal peaks — which is what the counter's second chain
looks for, and why a peak the hysteresis chain is too timid to arm on can
still be found. It is a twenty-millisecond slope rather than a difference of
neighbouring samples: differencing multiplies every frequency by itself and
the noise lives at the top of the band, so a raw difference is mostly noise.
Twenty milliseconds sits comfortably inside an axle peak, which is about
eighty milliseconds wide at a hundred kilometres an hour, so the rise and the
fall of one wheel stay separate events.

**The SNR of every passage** is no longer in the window; it is its own page,
built by `analysis/out/snr.html`, because it answers a question about the whole
campaign rather than about the record in front of you. It carries its own data
and needs no server.

**Every chart carries the same controls**, in the strip at the right of its
title: save it as a picture, drag-to-zoom or drag-to-pan, in, out, show
everything, mark a place, clear the marks, fill the window. The wheel zooms
about the cursor, shift inverts the drag mode, and a double-click on the chart
shows everything again.

**The cursor reads the chart.** Wherever it is, a crosshair follows it and the
value under it is written out in both directions — `89.342 s   54.4 pm`,
`29.69 Hz   0.108`, `29 May, 05:05`. Turn marking on and a click leaves that
reading pinned to the chart; click a mark to take it off, or clear them all.

**Zooming asks for more, not bigger.** The raw signal, the axle count and the
rate of change fetch that window again from the server, so going in shows MORE
signal, down to every individual sample.

**One panel at a time.** The `⤢` button fills the window with a panel, and so
does double-clicking its title bar; the same gesture or Escape brings the rest
back.

**Getting about.** Up and down arrows step through the records.

Onur's engine is read from the working tree, so the window shows the counter as
it stands right now, uncommitted edits included. Asude's is read from the branch
worktree that `lightrack.compare` lays down. The built program carries both,
because a switch that cannot be switched is not worth putting on the screen.

**Twelve gratings, six colours.** No set of twelve colours stays apart from each
other for every reader — under the usual red-green deficiencies twelve hues
collapse into far fewer — so the window spends six well-separated hues and
draws PADs 7 to 12 dashed. A grating keeps its colour and its stroke whatever
else is on the chart, and the legend shows both.

## Building it yourself

The checkout has to be called `lightrack`, because that is the name the code
imports itself by.

```powershell
git clone https://github.com/karadumanonur/LIGHTRACK.git lightrack
cd lightrack
python -m pip install -r requirements.txt -r packaging/requirements-build.txt
python -m PyInstaller --noconfirm packaging/lightrack.spec
```

`dist/LIGHTRACK.exe` is the whole program. The spec carries the static pages,
the `analysis` package and Asude's tracker as SOURCE rather than as compiled
modules, because the window shows the algorithm it ran and a byte-compiled copy
has nothing to show.

Asude's tracker is read from `.sides/asude`, which `python -m lightrack.compare`
lays down from her branch. Build without it and you get a program with one
engine and a switch that cannot be switched, so lay it down first.

**The Mac program cannot be built on Windows.** PyInstaller does not
cross-compile. `.github/workflows/macos.yml` runs the same spec on a Mac
runner, wraps the bundle in a disk image and attaches it to a release; start it
by hand from the Actions tab with the tag to attach to. On a Mac of your own it
is the same three commands as above, and then

```bash
hdiutil create -volname LIGHTRACK -srcfolder staging -ov -format UDZO LIGHTRACK.dmg
```

with `dist/LIGHTRACK.app` and a symlink to `/Applications` inside `staging/`.

## Running the comparison

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
